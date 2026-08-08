"""
Testbed extensions: the two Chapter 9.9 items the CCA build was missing, plus the
Prediction 3, 4, and 8 machinery they unlock.

9.9 item 7 (independent external logs): EventLog records every event and update in
an object held SEPARATELY from the bearer. The agent never reads it. It is an
auditor's record, not part of the state the agent acts from. This is exactly the
manuscript's discipline that persistence of information is not persistence of the
bearer.

9.9 item 8 (ablation access to constitution and record): exchange, corrupt, and
remove the two candidate stores independently. We designate the fast-weight store
P as the external record m and the bearer state b as the incorporated bearer
state, following Chapter 5's external-record vs changed-actor split. Exchanging
them independently is the manuscript's Prediction 4 four-condition design, and it
answers the storage-vs-incorporation question directly: does later behavior follow
the record P or the bearer state b?

Also included:
- Prediction 3 (delayed counterfactual): omit one outcome mid-history, then test
  divergence AFTER the cue is gone.
- Constitution exchange: swap the private body, holding the developed state fixed.

Usage:  python testbed_extensions.py --champion champion_gpu_drift.npz
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import numpy as np
import torch

import gpu_batched as G
from cca_milestone1 import Config, PlasticAgent, N_ACTIONS


# ---------------------------------------------------------------------------
# 9.9 item 7: an independent per-event audit log, external to the bearer
# ---------------------------------------------------------------------------

class EventLog:
    """Auditor's record of every interaction. Held outside the agent; the agent's
    act() and plastic_update() never take it as input, so by construction it is a
    record, not bearer state. Records per step: action, resource eaten, hazard
    hit, the outcome components, and the update magnitudes |dP| and |db|."""

    def __init__(self):
        self.rows = []  # each: (t, action, raw_food, taste_food, eps, dP, db)

    def record(self, t, action, env, ag, b_prev):
        db = float((ag.b - b_prev).norm())
        self.rows.append((
            int(t), int(action),
            float(env.last_raw.sum()), float(env.last_taste.sum()),
            float(env._eps().mean()), float(ag.last_dP.mean()), db,
        ))

    def as_array(self):
        return np.array(self.rows, dtype=np.float64)

    def save(self, path):
        a = self.as_array()
        np.savez(path, log=a,
                 columns=np.array(["t", "action", "raw", "taste", "eps", "dP", "db"]))
        return a.shape


# ---------------------------------------------------------------------------
# Loading a champion and developing donor states
# ---------------------------------------------------------------------------

def load_champion(path, device):
    npz = np.load(path)
    d = int(npz["d"]) if "d" in npz else 64
    drift = int(npz["drift_every"]) if "drift_every" in npz else 0
    homeo = bool(int(npz["observe_homeo"])) if "observe_homeo" in npz else True
    theta = torch.from_numpy(npz["theta"].astype(np.float32)).to(device)
    cfg = Config(seed=0, lambda_b=1.0, drift_every=drift, d=d, bearer_dim=d,
                 observe_homeo=homeo)
    obs_dim = G.BatchedForagerEnv(cfg, 1, 0, device).obs_dim
    spec = G.param_spec(cfg, obs_dim)
    torch.manual_seed(0)
    base = {n: p.detach().to(device) for n, p in PlasticAgent(cfg, obs_dim).named_parameters()}
    return theta, cfg, spec, base, obs_dim


def set_body(env, seed):
    """Overwrite the env's private constitution with a body drawn from `seed`."""
    r = np.random.default_rng(seed + 555)
    K, H = env.cfg.K, env.cfg.H
    env.m_dev = env._t(r.normal(0, env.cfg.const_sigma, K))
    env.v_dev = env._t(r.normal(0, env.cfg.const_sigma, H))
    env.kappa = env._t(np.clip(r.normal(0.002, 0.001, N_ACTIONS), 2e-4, 6e-3))


@torch.no_grad()
def develop(theta, cfg, spec, base, obs_dim, world_seed, body_seed, steps,
            device, log=False):
    """Run one agent to a developed state. Returns a checkpoint of its record P,
    bearer state b, hidden h, and optionally an EventLog."""
    env = G.BatchedForagerEnv(cfg, 1, world_seed, device)
    set_body(env, body_seed)
    params = G.unflatten_batch(theta.unsqueeze(0), spec, "all", base)
    ag = G.BatchedAgent(cfg, params, obs_dim, device)
    ev = EventLog() if log else None
    obs = env.obs()
    for t in range(steps):
        b_prev = ag.b.clone()
        a = ag.act(obs)
        obs, F, Vs, eps, Vw = env.step(a)
        ag.plastic_update((1 - cfg.lambda_b) * F + cfg.lambda_b * Vs, eps,
                          env._degradation())
        if ev is not None:
            ev.record(t, int(a.item()), env, ag, b_prev)
    ckpt = {"P": {k: v.clone() for k, v in ag.P_fast.items()},
            "b": ag.b.clone(), "h": ag.h.clone()}
    return ckpt, ev


def fresh_agent(theta, cfg, spec, base, obs_dim, P, device):
    params = G.unflatten_batch(theta.unsqueeze(0).expand(P, -1).contiguous(),
                               spec, "all", base)
    return G.BatchedAgent(cfg, params, obs_dim, device)


def inject(ag, cond, P_src, b_src, h_src):
    """Set condition `cond` of a batched agent from chosen record/bearer sources."""
    for k in ag.LAYERS:
        ag.P_fast[k][cond] = P_src[k][0]
    ag.b[cond] = b_src[0]
    ag.h[cond] = h_src[0]


# ---------------------------------------------------------------------------
# Prediction 4: record exchange vs bearer-state exchange
# ---------------------------------------------------------------------------

@torch.no_grad()
def prediction4(theta, cfg, spec, base, obs_dim, device, steps=400, test_seed=90123):
    """Four conditions in one shared test world:
       0: (P_A, b_A)  intact A
       1: (P_B, b_A)  foreign record, own bearer state
       2: (P_A, b_B)  foreign bearer state, own record
       3: (P_B, b_B)  intact B
    Record = fast weights P. Bearer state = b. If later coupling follows P, the
    system is storage-dominant; if it follows b, it is incorporation-dominant."""
    A, _ = develop(theta, cfg, spec, base, obs_dim, 91001, body_seed=11, steps=800, device=device)
    B, _ = develop(theta, cfg, spec, base, obs_dim, 91002, body_seed=22, steps=800, device=device)

    env = G.BatchedForagerEnv(cfg, 4, test_seed, device)
    ag = fresh_agent(theta, cfg, spec, base, obs_dim, 4, device)
    inject(ag, 0, A["P"], A["b"], A["h"])
    inject(ag, 1, B["P"], A["b"], A["h"])
    inject(ag, 2, A["P"], B["b"], B["h"])
    inject(ag, 3, B["P"], B["b"], B["h"])

    # Freeze plasticity during the test so the injected record P and bearer state
    # b are the only causes of behavior. If they kept updating, the agent would
    # re-develop to its setpoint and wash out the exchange.
    obs = env.obs(); acc = torch.zeros(4, device=device); n = 0
    for t in range(steps):
        a = ag.act(obs)
        obs, F, Vs, eps, Vw = env.step(a)
        if t >= steps // 4:
            acc -= eps; n += 1
    return (acc / max(n, 1)).cpu().numpy()


# ---------------------------------------------------------------------------
# Record corruption / removal vs bearer-state removal (item 8 signature)
# ---------------------------------------------------------------------------

@torch.no_grad()
def corruption_removal(theta, cfg, spec, base, obs_dim, device, steps=400, test_seed=90123):
    """Take one developed agent and test four interventions in a shared world:
       intact / record-corrupted (P<-donor) / record-removed (P=0) / bearer-removed (b=0)."""
    A, _ = develop(theta, cfg, spec, base, obs_dim, 91001, body_seed=11, steps=800, device=device)
    D, _ = develop(theta, cfg, spec, base, obs_dim, 91003, body_seed=33, steps=800, device=device)
    env = G.BatchedForagerEnv(cfg, 4, test_seed, device)
    ag = fresh_agent(theta, cfg, spec, base, obs_dim, 4, device)
    inject(ag, 0, A["P"], A["b"], A["h"])                    # intact
    inject(ag, 1, D["P"], A["b"], A["h"])                    # record corrupted (donor P)
    inject(ag, 2, A["P"], A["b"], A["h"]); [ag.P_fast[k][2].zero_() for k in ag.LAYERS]  # record removed
    inject(ag, 3, A["P"], A["b"], A["h"]); ag.b[3].zero_()   # bearer state removed

    obs = env.obs(); acc = torch.zeros(4, device=device); n = 0
    for t in range(steps):
        a = ag.act(obs)
        obs, F, Vs, eps, Vw = env.step(a)
        ag.plastic_update((1 - cfg.lambda_b) * F + cfg.lambda_b * Vs, eps, env._degradation())
        if t >= steps // 2:
            acc -= eps; n += 1
    return (acc / max(n, 1)).cpu().numpy()


# ---------------------------------------------------------------------------
# Prediction 3: delayed counterfactual divergence from one omitted outcome
# ---------------------------------------------------------------------------

@torch.no_grad()
def prediction3(theta, cfg, spec, base, obs_dim, device, T_omit=200, win=25,
                gap=300, world_seed=91004, body_seed=44):
    """Two matched histories differing only in whether the updates in a short
    window [T_omit, T_omit+win) are applied. Measure bearer-state divergence AFTER
    the cue is gone (gap steps later): delayed incorporation, not immediate
    context. A single step can be a null outcome; a window ensures a real
    manipulation. The action stream is held identical across the two runs so the
    divergence is caused by incorporation, not by different sampled actions."""
    def run(omit):
        env = G.BatchedForagerEnv(cfg, 1, world_seed, device); set_body(env, body_seed)
        params = G.unflatten_batch(theta.unsqueeze(0), spec, "all", base)
        ag = G.BatchedAgent(cfg, params, obs_dim, device)
        obs = env.obs()
        gen = torch.Generator(device=device); gen.manual_seed(777)
        for t in range(T_omit + win + gap):
            # deterministic-but-shared action draw so both runs act identically
            with torch.no_grad():
                z = torch.tanh(ag._lin("enc1", obs)); z = torch.tanh(ag._lin("enc2", z))
                film = G._bmv(ag.p["film.weight"], ag.b) + ag.p["film.bias"]
                gm, bt = film.chunk(2, 1); z = z * (1 + gm) + bt
                ag.h = ag._gru("gru", z, ag.h)
                logits = ag._lin("pi_head", ag.h)
                a = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1)
                for name in ag.LAYERS:
                    x, y = ag._cache[name]; outer = torch.einsum("po,pi->poi", y, x)
                    nrm = outer.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-8
                    ag.elig[name].mul_(ag.cfg.fw_gamma_e).add_(outer / nrm)
            obs, F, Vs, eps, Vw = env.step(a)
            o = (1 - cfg.lambda_b) * F + cfg.lambda_b * Vs
            if omit and T_omit <= t < T_omit + win:
                continue                          # omit these updates entirely
            ag.plastic_update(o, eps, env._degradation())
        return ag.b.clone()
    b_full = run(False)
    b_omit = run(True)
    return float((b_full - b_omit).norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="champion_gpu_drift.npz")
    ap.add_argument("--out-log", default="event_log.npz")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta, cfg, spec, base, obs_dim = load_champion(args.champion, device)
    print(f"[testbed] champion={args.champion} d={cfg.d} drift={cfg.drift_every} "
          f"observe_homeo={cfg.observe_homeo}")

    # item 7: independent audit log
    _, ev = develop(theta, cfg, spec, base, obs_dim, 91000, body_seed=11, steps=600,
                    device=device, log=True)
    shape = ev.save(args.out_log)
    print(f"\n[item 7] EventLog: {shape[0]} events x {shape[1]} fields "
          f"-> {args.out_log} (external to the bearer; never read by act/update)")

    # Prediction 4: does behavior follow the record P or the bearer state b?
    p4 = prediction4(theta, cfg, spec, base, obs_dim, device)
    print("\n[Prediction 4] record (P) vs bearer-state (b) exchange, coupling (-eps):")
    print(f"  (P_A,b_A)={p4[0]:+.4f}  (P_B,b_A)={p4[1]:+.4f}  "
          f"(P_A,b_B)={p4[2]:+.4f}  (P_B,b_B)={p4[3]:+.4f}")
    same_P = abs(p4[0] - p4[1]) + abs(p4[2] - p4[3])       # small if P drives
    same_b = abs(p4[0] - p4[2]) + abs(p4[1] - p4[3])       # small if b drives
    verdict = ("incorporation-dominant (behavior follows the bearer state b)"
               if same_b < same_P else
               "storage-dominant (behavior follows the record P)")
    print(f"  spread when P held fixed: {same_b:.4f} | when b held fixed: {same_P:.4f}")
    print(f"  -> {verdict}")

    # item 8 signature: corruption / removal
    cr = corruption_removal(theta, cfg, spec, base, obs_dim, device)
    print("\n[item 8] corruption/removal, coupling (-eps):")
    print(f"  intact={cr[0]:+.4f}  record-corrupted={cr[1]:+.4f}  "
          f"record-removed={cr[2]:+.4f}  bearer-removed={cr[3]:+.4f}")
    print(f"  record sensitivity (intact - removed): {cr[0]-cr[2]:+.4f} | "
          f"bearer sensitivity (intact - removed): {cr[0]-cr[3]:+.4f}")

    # Prediction 3: delayed counterfactual divergence
    d3 = prediction3(theta, cfg, spec, base, obs_dim, device)
    print(f"\n[Prediction 3] delayed bearer divergence from one omitted outcome "
          f"(300 steps after the cue): ||b_full - b_omit|| = {d3:.5f}")
    print("  (nonzero after the gap = the specific outcome was historically incorporated)")


if __name__ == "__main__":
    main()
