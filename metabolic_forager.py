"""
The private-metabolism task: task 3 done right (the obligate bearer test).

Built to satisfy the Generic-Policy Escape criterion (CONSOLIDATION_ESCAPE.md):
every bearer-neutral policy must fail for some bearer, while a bearer-correct
policy keeps every bearer viable.

Design. Eating gives bearer-NEUTRAL calories, and a continuous maintenance cost
forces the agent to keep eating (no eat-less escape, no timing escape). But
viability also requires the DIET COMPOSITION to match a private per-body demand
vector r_i (a point on the simplex over K nutrient types). r_i differs per bearer
with no universal answer (no fixed-mixture escape), and misalignment penalizes
BOTH deficiency and surplus (no quantity escape). The agent observes its own
recent diet but NOT r_i, so it must LEARN its needs through its own outcomes,
which routes the bearer-specific behavior through the incorporated pathway.

Bearer manipulation. V_self scores the diet against the OWN r_i; V_wrong scores
the SAME diet against an ALT body's r_i. A wrong-bearer learner drives its diet
toward the alt demand, which is misaligned with its own body, so its own
viability suffers. Prediction: self beats wrong on the own-body metric.

Mediator check FIRST. Before reading self-vs-wrong, verify the learned policy
actually matches its diet to its own r_i (else a null is ambiguous, the lesson of
the obligate-poison run).

Usage:  python metabolic_forager.py --gens 80 --pop 768
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse, json, time
import numpy as np
import torch

import gpu_batched as G
from cca_milestone1 import Config, PlasticAgent, A_CONSUME, A_WAIT

K = 6  # nutrient types (matches cfg.K)
EMA = 0.04          # diet decay: recent-diet histogram
W_MIS = 1.0         # weight of metabolic misalignment in eps
CAL = 0.06          # calories per meal (bearer-neutral)
GAIN = 1.0          # Lever A: amplifies net energy of above-uniform selection.
                    # net = CAL*GAIN*(r[k]*K - 1). The zero-crossing stays at
                    # uniform (r*K=1) so the escape is NOT reopened, but a larger
                    # GAIN lets a modest own-demand alignment (e.g. the curriculum's
                    # 0.656) accrue enough energy to hold the band -> viable, so the
                    # ablation gate becomes informative. Set from --gain-scale.


class MetabolicForagerEnv(G.BatchedForagerEnv):
    """Nutrient-balance forager. Reuses vision/movement/respawn/drift from the
    base env; replaces the energy/integrity homeostasis with an energy +
    diet-composition-vs-private-demand model."""

    def __init__(self, cfg, P, seed, device):
        super().__init__(cfg, P, seed, device)
        rng = np.random.default_rng(seed + 555)
        # private demand r_i: a simplex over K types, made diverse by const_sigma
        logits = rng.normal(0, cfg.const_sigma, (cfg.K,))
        r = np.exp(logits - logits.max()); r = r / r.sum()
        self.r_own = self._t(r)                              # [K]
        self.r_alt = None                                   # set for wrong-bearer
        self.diet = torch.zeros(P, cfg.K, device=device)     # eaten-type histogram
        # OBLIGATE energy model: net energy from eating type k is CAL*(r[k]*K - 1),
        # positive only for above-average-demand types, negative (toxic) for
        # below-average ones. Uniform eating nets ~0 against a continuous
        # maintenance cost and starves, so it is foreclosed. Only selecting one's
        # own high-demand types is viable, and that requires knowing own r_i.
        self.E = torch.full((P,), 0.6, device=device)
        self.E_alt = self.E.clone()                          # shadow for wrong-bearer
        # curriculum: per-agent blindness. When blind, the E and diet channels of
        # the observation are zeroed for the whole episode, so the agent cannot
        # sense its state through the recurrent shortcut and must rely on the
        # incorporated outcome-driven pathway. Ramped from 0 to 1 over training.
        self.blind = torch.zeros(P, device=device)
        self.eps_prev = self._band(self.E, cfg.band_E)
        self.eps_prev_alt = None

    def set_blind(self, prob):
        self.blind = (torch.rand(self.P, device=self.dev) < prob).float()

    def set_alt_demand(self, seed):
        rng = np.random.default_rng(seed + 1 + 555)
        logits = rng.normal(0, self.cfg.const_sigma, (self.cfg.K,))
        r = np.exp(logits - logits.max()); r = r / r.sum()
        self.r_alt = self._t(r)
        self.E_alt = torch.full((self.P,), 0.6, device=self.dev)
        self.eps_prev_alt = self._band(self.E_alt, self.cfg.band_E)

    def _diet_norm(self):
        return self.diet / (self.diet.sum(1, keepdim=True) + 1e-8)

    def _eps(self):                                          # own-body eps
        if not hasattr(self, "r_own"):                       # during base __init__
            return torch.zeros(self.P, device=self.dev)
        return self._band(self.E, self.cfg.band_E)

    @property
    def obs_dim(self):
        base = (self.cfg.K + self.cfg.H + 1) * self.cfg.view ** 2
        # + energy (1) + own diet composition (K). Blind to r_own and to eps.
        return base + 1 + self.cfg.K

    def obs(self):
        P, G_, v2 = self.P, self.cfg.grid, self.cfg.view ** 2
        C = self.cfg.K + self.cfg.H + 1
        wx = (self.pos[:, 0:1] + self.off[0].unsqueeze(0)) % G_
        wy = (self.pos[:, 1:2] + self.off[1].unsqueeze(0)) % G_
        idx = wx * G_ + wy
        rk = self.res.view(P, -1).gather(1, idx)
        hk = self.haz.view(P, -1).gather(1, idx)
        win = torch.zeros(P, C, v2, device=self.dev)
        win.scatter_(1, (rk - 1).clamp(min=0).unsqueeze(1),
                     (rk > 0).float().unsqueeze(1))
        win.scatter_(1, (self.cfg.K + (hk - 1).clamp(min=0)).unsqueeze(1),
                     (hk > 0).float().unsqueeze(1))
        win[:, -1, v2 // 2] = 1.0
        vis = 1.0 - self.blind.unsqueeze(1)                  # 0 for blind agents
        state = torch.cat([self.E.unsqueeze(1), self._diet_norm()], 1) * vis
        return torch.cat([win.reshape(P, -1), state], 1)

    def step(self, a):
        cfg, G_, P = self.cfg, self.cfg.grid, self.P
        self.t += 1
        eps_before, eps_before_alt = self.eps_prev, self.eps_prev_alt
        deg = self._degradation()
        slip = torch.rand(P, device=self.dev, generator=self.g) < 0.3 * deg
        a = torch.where(slip, torch.full_like(a, A_WAIT), a)
        F_gen = torch.zeros(P, device=self.dev)

        is_move = a < 8
        delta = torch.where(is_move.unsqueeze(1), self.moves[a.clamp(max=7)],
                            torch.zeros_like(self.pos))
        self.pos = (self.pos + delta) % G_

        flat = self._flat(self.pos)
        res_flat = self.res.view(P, -1)
        k_here = res_flat.gather(1, flat.unsqueeze(1)).squeeze(1)
        eating = (a == A_CONSUME) & (k_here > 0)
        ki = (k_here - 1).clamp(min=0)
        ate = eating.float()
        # net energy = CAL*(r[k]*K - 1): nourishing for own high-demand types,
        # toxic for below-average ones. Uniform selection nets ~0 -> starves.
        gain_own = CAL * GAIN * (self.r_own[ki] * cfg.K - 1.0) * ate
        self.E = (self.E + gain_own).clamp(0.0, 1.0)
        onehot = torch.zeros(P, cfg.K, device=self.dev)
        onehot.scatter_(1, ki.unsqueeze(1), ate.unsqueeze(1))
        self.diet = self.diet * (1 - EMA) + onehot                  # eaten-type hist
        F_gen = F_gen + CAL * ate
        self.last_raw = ate
        self.last_taste = ate
        res_flat.scatter_(1, flat.unsqueeze(1),
                          torch.where(eating, torch.zeros_like(k_here), k_here).unsqueeze(1))

        self.E = (self.E - cfg.passive_decay).clamp(0.0, 1.0)       # maintenance
        self._respawn()
        if cfg.drift_every and self.t % cfg.drift_every == 0:
            self._drift()

        eps = self._band(self.E, cfg.band_E)
        V_self = eps_before - eps
        self.eps_prev = eps

        V_wrong = None
        if self.r_alt is not None:                                  # same event, alt body
            gain_alt = CAL * GAIN * (self.r_alt[ki] * cfg.K - 1.0) * ate
            self.E_alt = (self.E_alt + gain_alt - cfg.passive_decay).clamp(0.0, 1.0)
            eps_alt = self._band(self.E_alt, cfg.band_E)
            V_wrong = eps_before_alt - eps_alt
            self.eps_prev_alt = eps_alt
        return self.obs(), F_gen, V_self, eps, V_wrong


# --- ES training (compact, held-out) using the metabolic env ---------------

def build(cfg, device):
    obs_dim = MetabolicForagerEnv(cfg, 1, 0, device).obs_dim
    spec = G.param_spec(cfg, obs_dim)
    torch.manual_seed(0)
    base = {n: p.detach().to(device) for n, p in PlasticAgent(cfg, obs_dim).named_parameters()}
    return obs_dim, spec, base


@torch.no_grad()
def eval_pop(theta, cfg, spec, base, obs_dim, lifetime, seed, device, blind=0.0):
    P = theta.shape[0]
    env = MetabolicForagerEnv(cfg, P, seed, device)
    if blind > 0:
        env.set_blind(blind)
    ag = G.BatchedAgent(cfg, G.unflatten_batch(theta, spec, "all", base), obs_dim, device)
    obs = env.obs(); acc = torch.zeros(P, device=device); n = 0; t0 = int(lifetime * .75)
    for t in range(lifetime):
        a = ag.act(obs)
        obs, F, Vs, eps, Vw = env.step(a)
        ag.plastic_update(Vs, eps, env._degradation())
        if t >= t0: acc -= eps; n += 1
    return torch.where(ag.last_dP <= 0, torch.full_like(acc, -1e6), acc / max(n, 1))


def train(cfg, pop, gens, lifetime, obs_dim, spec, base, device, log=print, curriculum=False):
    D = G.spec_dim(spec, "all")
    theta = torch.cat([base[n].reshape(-1) for n, _, _ in spec if G._in_scope(n, "all")]).clone()
    hold = [90000 + i for i in range(6)]
    # under the curriculum the deployment target is FULLY blind, so select and
    # baseline there; training blindness ramps 0 -> 1 to push the agent off the
    # observable shortcut onto the incorporated pathway after it can forage.
    eval_blind = 1.0 if curriculum else 0.0
    def ramp(gi):
        if not curriculum:
            return 0.0
        return float(np.clip((gi / gens - 0.2) / 0.5, 0.0, 1.0))
    def randbase():
        vs = []
        for i in range(12):
            torch.manual_seed(555 + i); a = PlasticAgent(cfg, obs_dim)
            vs.append(torch.cat([p.detach().reshape(-1).to(device) for nm, p in a.named_parameters() if G._in_scope(nm, "all")]))
        stack = torch.stack(vs)
        per = np.zeros(12)
        for s in hold:
            per += eval_pop(stack, cfg, spec, base, obs_dim, lifetime, s, device, blind=eval_blind).cpu().numpy()
        return float((per / len(hold)).mean())
    rand = randbase(); log(f"[metab] D={D} rand baseline={rand:+.4f} curriculum={curriculum} eval_blind={eval_blind}")
    gen = torch.Generator(device=device); gen.manual_seed(99)
    best, best_theta = -1e9, theta.clone()
    for gi in range(gens):
        t0 = time.time()
        bt = ramp(gi)
        noise = torch.randn(pop // 2, D, device=device, generator=gen)
        eps_all = torch.cat([noise, -noise], 0)
        fits = eval_pop(theta.unsqueeze(0) + 0.02 * eps_all, cfg, spec, base, obs_dim, lifetime, 10000 + gi, device, blind=bt)
        theta = theta + 0.05 * (G.centered_ranks(fits).unsqueeze(1) * eps_all).sum(0) / (pop * 0.02)
        log(f"[gen {gi:03d}] train={fits.mean():+.3f} blind_train={bt:.2f} ({time.time()-t0:.1f}s)")
        if gi % 20 == 0 or gi == gens - 1:
            ho = float(np.mean([float(eval_pop(theta.unsqueeze(0), cfg, spec, base, obs_dim, lifetime, s, device, blind=eval_blind)[0]) for s in hold]))
            if ho > best: best, best_theta = ho, theta.clone()
            log(f"          HELD-OUT={ho:+.3f} (vs rand {rand:+.3f}, {ho-rand:+.3f})")
    return best_theta, best, rand


# --- mediator check: does the diet match own r_i? --------------------------

@torch.no_grad()
def mediator_check(theta, cfg, spec, base, obs_dim, device, seeds=8, lifetime=1200, blind=0.0):
    """SHARP check: does the diet CONCENTRATE on the agent's own high-demand
    types? Alignment = diet_norm . r_own. A uniform diet gives 1/K exactly (the
    escape). Perfect selection of the single top type gives max(r_own). The
    champion is a real mediator only if its alignment exceeds 1/K clearly.
    (The earlier L1-vs-random check was fooled by uniform eating.)"""
    def align(th, seed):
        env = MetabolicForagerEnv(cfg, 1, seed, device)
        if blind > 0: env.set_blind(blind)
        ag = G.BatchedAgent(cfg, G.unflatten_batch(th.unsqueeze(0), spec, "all", base), obs_dim, device)
        obs = env.obs()
        for t in range(lifetime):
            a = ag.act(obs); obs, F, Vs, eps, Vw = env.step(a)
            ag.plastic_update(Vs, eps, env._degradation())
        d = env._diet_norm()[0]
        return float((d * env.r_own).sum()), float(env.r_own.max())
    champ = [align(theta, 90000 + s) for s in range(seeds)]
    champ_align = np.mean([c[0] for c in champ])
    ceil = np.mean([c[1] for c in champ])       # best possible (eat only top type)
    unif = 1.0 / cfg.K                          # uniform-diet alignment (the escape)
    return champ_align, unif, ceil


@torch.no_grad()
def gate(theta, cfg, spec, base, obs_dim, device, seeds=8, life=1500, blind=0.0):
    def run(plastic, seed):
        env = MetabolicForagerEnv(cfg, 1, seed, device)
        if blind > 0: env.set_blind(blind)
        ag = G.BatchedAgent(cfg, G.unflatten_batch(theta.unsqueeze(0), spec, "all", base), obs_dim, device)
        obs = env.obs(); acc = 0; n = 0
        for t in range(life):
            a = ag.act(obs); obs, F, Vs, eps, Vw = env.step(a)
            if plastic: ag.plastic_update(Vs, eps, env._degradation())
            if t >= life // 2: acc -= float(eps.mean()); n += 1
        return acc / max(n, 1)
    on = np.mean([run(True, 90000 + s) for s in range(seeds)])
    off = np.mean([run(False, 90000 + s) for s in range(seeds)])
    return on - off


@torch.no_grad()
def bearer_sweep(theta, cfg, spec, base, obs_dim, device, seeds=20, life=1500,
                 lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), blind=0.0):
    """self vs wrong on the own-body metabolic eps. Paired within-world."""
    nl = len(lambdas)
    allp = np.zeros((seeds, nl, 2))
    lam = torch.tensor([lambdas[i] for i in range(nl) for _ in range(2)], device=device)
    is_w = torch.tensor([j for _ in range(nl) for j in range(2)], device=device).float()
    Pn = nl * 2
    for s in range(seeds):
        env = MetabolicForagerEnv(cfg, Pn, 90000 + s, device)
        if blind > 0: env.set_blind(blind)
        env.set_alt_demand(90000 + s)
        ag = G.BatchedAgent(cfg, G.unflatten_batch(theta.unsqueeze(0).expand(Pn, -1).contiguous(), spec, "all", base), obs_dim, device)
        obs = env.obs(); acc = torch.zeros(Pn, device=device); n = 0; t0 = int(life * .75)
        for t in range(life):
            a = ag.act(obs); obs, F, Vs, eps, Vw = env.step(a)
            bearer = torch.where(is_w.bool(), Vw, Vs)
            ag.plastic_update((1 - lam) * F + lam * bearer, eps, env._degradation())
            if t >= t0: acc -= eps; n += 1
        allp[s] = (acc / max(n, 1)).cpu().numpy().reshape(nl, 2)
    mean = allp.mean(0); dP = mean[:, 0] - mean[:, 1]
    dP_ci = 1.96 * (allp[:, :, 0] - allp[:, :, 1]).std(0, ddof=1) / np.sqrt(seeds)
    return mean, dP, dP_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--pop", type=int, default=768)
    ap.add_argument("--lifetime", type=int, default=1200)
    ap.add_argument("--const-sigma", type=float, default=1.5)
    ap.add_argument("--K", type=int, default=6,
                    help="number of nutrient types. K=2 is the learnable-obligate "
                         "regime: uniform still starves, but the private demand is "
                         "~1 bit, tractable to learn blind (threads the valley)")
    ap.add_argument("--sweep-seeds", type=int, default=20)
    ap.add_argument("--gain-scale", type=float, default=1.0,
                    help="Lever A: amplify net energy of above-uniform selection "
                         "(keeps the uniform zero-crossing, so no escape reopens). "
                         ">1 lets modest own-demand alignment reach viability so the "
                         "gate becomes informative (BEARER_OBSERVABILITY lever A)")
    ap.add_argument("--curriculum", action="store_true",
                    help="ramp observation-blindness 0->1 during training, so the "
                         "agent is pushed off the observable shortcut onto the "
                         "incorporated pathway; eval fully blind (BEARER_OBSERVABILITY)")
    ap.add_argument("--learned-update", action="store_true",
                    help="Lever B: give the plasticity controller a 4-term learned "
                         "update rule instead of gated Hebbian, so the incorporated "
                         "pathway can acquire blind inference more strongly")
    ap.add_argument("--out", default="metabolic_result.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global GAIN
    GAIN = args.gain_scale
    cfg = Config(seed=0, lambda_b=1.0, drift_every=200, d=64, bearer_dim=64,
                 const_sigma=args.const_sigma, K=args.K,
                 learned_update=args.learned_update)
    obs_dim, spec, base = build(cfg, device)
    eb = 1.0 if args.curriculum else 0.0
    print(f"[metabolic] K={cfg.K} const_sigma={args.const_sigma} obs_dim={obs_dim} "
          f"gens={args.gens} pop={args.pop} curriculum={args.curriculum} gain_scale={GAIN} "
          f"learned_update={cfg.learned_update}")

    theta, best, rand = train(cfg, args.pop, args.gens, args.lifetime, obs_dim, spec, base, device, curriculum=args.curriculum)
    adv = best - rand
    print(f"\n[TRAIN] champion held-out {best:+.3f} vs random {rand:+.3f} -> advantage {adv:+.3f} "
          f"({'TRAINED' if adv > 0.05 else 'NOT TRAINED'})")

    champ_align, unif, ceil = mediator_check(theta, cfg, spec, base, obs_dim, device, blind=eb)
    present = champ_align > unif + 0.05
    print(f"\n[MEDIATOR CHECK] diet.r_own alignment: champion={champ_align:.3f} | "
          f"uniform-escape={unif:.3f} | ceiling={ceil:.3f}")
    print(f"  {'MEDIATOR PRESENT: diet concentrates on own high-demand types' if present else 'MEDIATOR ABSENT (near uniform escape) -> sweep uninterpretable'}")

    g = gate(theta, cfg, spec, base, obs_dim, device, blind=eb)
    print(f"[GATE] plastic ON-OFF = {g:+.3f} ({'PASS' if g > 0.03 else 'FAIL'})")

    mean, dP, dP_ci = bearer_sweep(theta, cfg, spec, base, obs_dim, device, seeds=args.sweep_seeds, life=args.lifetime, blind=eb)
    print("\n[BEARER SWEEP] self vs wrong on own-body metabolic viability:")
    print(f"{'lambda':>7} {'self':>10} {'wrong':>10} {'deltaP':>16}")
    for i, l in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        print(f"{l:>7.2f} {mean[i,0]:>+10.4f} {mean[i,1]:>+10.4f} {dP[i]:>+8.4f}+/-{dP_ci[i]:.4f}")
    sig = [i for i in range(1, 5) if abs(dP[i]) > dP_ci[i]]
    verdict = (f"BEARER EFFECT at lambda {[[0,.25,.5,.75,1][i] for i in sig]}, "
               f"{'self>wrong (predicted)' if all(dP[i]>0 for i in sig) else 'mixed'}") if sig else "no effect above noise"
    print(f"  falsifier dP(0)={dP[0]:+.4f}+/-{dP_ci[0]:.4f} | VERDICT: {verdict}")

    json.dump({"advantage": adv, "mediator_align": champ_align, "mediator_uniform": unif,
               "mediator_ceiling": ceil, "mediator_present": bool(present), "gate": g,
               "deltaP": dP.tolist(), "deltaP_ci": dP_ci.tolist(),
               "self": mean[:, 0].tolist(), "wrong": mean[:, 1].tolist()},
              open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
