"""
Bearer-indexing sweep on the GPU, run on a TRAINED champion.

This is the experiment the whole CCA package exists for, and it is only
interpretable now that we have an agent that beats random (see
RESULTS_GPU_TRAIN.md). Sweep #1 measured an untrained agent and its null was
meaningless.

The question: as the outcome an agent updates on shifts from a world fact
(F_generic) to a bearer-relative fact (V_self, the agent's own homeostatic
change), does it matter WHOSE body the consequence is valued through?

Conditions, per lambda (the world-fact vs bearer-fact mix, cfg.lambda_b):
  self          outcome = (1-lam) F + lam V_self       (valued through own body)
  wrong_bearer  outcome = (1-lam) F + lam V_wrong       (SAME event, valued through
                a different constitution: same action, same dice, wrong body)

Headline: Delta P(lambda) = perf(self) - perf(wrong_bearer).
Falsifier: at lambda=0 the bearer term has zero weight, so the two conditions
are identical BY CONSTRUCTION and Delta P(0) must be ~0. The script checks this
and refuses to interpret the curve if it fails.

Batching: for each held-out world seed, the champion is instantiated once per
(lambda, mode) cell as a batch that SHARES the world. All cells start from
identical weights and the same world; they diverge only through the outcome
rule each one applies. So within a world the comparison is paired, and world
variance cancels in Delta P. Seeds are looped for a confidence interval.

Not included here: the 'stream' control (classic yoked, wrong event AND wrong
bearer). It needs a donor agent in a second world and is a secondary control;
the self-vs-wrong_bearer contrast is the clean isolation. Add it in a second
pass if wanted.

Usage:
    python bearer_sweep_gpu.py --champion champion_gpu_big.npz \
        --lifetime 2000 --seeds 24
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json

import numpy as np
import torch

import gpu_batched as G
from cca_milestone1 import Config, N_ACTIONS

MODES = ("self", "wrong_bearer")


def set_alt_constitution(env, seed):
    """Give the env a wrong-bearer body: a constitution drawn from a different
    seed, matching the CPU Constitution.draw order (m_dev, v_dev, kappa)."""
    r = np.random.default_rng(seed + 1 + 555)
    K, H = env.cfg.K, env.cfg.H
    env.alt_m = env._t(r.normal(0, env.cfg.const_sigma, K))
    env.alt_v = env._t(r.normal(0, env.cfg.const_sigma, H))
    env.alt_kappa = env._t(np.clip(r.normal(0.002, 0.001, N_ACTIONS), 2e-4, 6e-3))


# the scoring yardsticks, in the order eval_world returns them
#   diet_quality = taste_food/raw_food is the clean selectivity signal: the
#   average own-body value of the resources the agent CHOSE to eat. Self (learns
#   own taste) should eat higher-own-value than wrong_bearer (learns alt taste),
#   controlling for how much was eaten. This is the sharp bearer-INDEXED test.
METRICS = ("homeostasis", "raw_food", "taste_food", "diet_quality")


@torch.no_grad()
def eval_world(theta_vec, cfg, spec, base, lambdas, lifetime, seed, device):
    """One world. Returns perf [n_lambda, n_mode, 3] for the champion under each
    (lambda, mode) cell, all sharing this world, scored on three yardsticks:
      homeostasis = -eps                         (reactive-regulatable via obs)
      raw_food    = sum base_value               (bearer-NEUTRAL: any body, same)
      taste_food  = sum base_value*(1+own m_dev) (bearer-INDEXED: own criterion)
    The paper's criterion-internal result predicts a bearer effect on taste_food
    and none on raw_food."""
    nl, nm = len(lambdas), len(MODES)
    P = nl * nm
    lam_grid = torch.tensor([lambdas[i] for i in range(nl) for _ in range(nm)],
                            device=device)                       # [P]
    mode_grid = torch.tensor([j for _ in range(nl) for j in range(nm)],
                             device=device)                      # [P]
    is_wrong = (mode_grid == 1).float()

    env = G.BatchedForagerEnv(cfg, P, seed, device)
    set_alt_constitution(env, seed)
    params = G.unflatten_batch(theta_vec.unsqueeze(0).expand(P, -1).contiguous(),
                               spec, "all", base)
    ag = G.BatchedAgent(cfg, params, env.obs_dim, device)

    obs = env.obs()
    tail_start = int(lifetime * 0.75)
    acc = torch.zeros(P, device=device)          # -eps, steady-state (tail)
    tcnt = 0
    raw = torch.zeros(P, device=device)          # raw food, WHOLE life
    taste = torch.zeros(P, device=device)        # own-taste food, WHOLE life
    for t in range(lifetime):
        a = ag.act(obs)
        obs, F_gen, V_self, eps, V_wrong = env.step(a)
        bearer = torch.where(is_wrong.bool(), V_wrong, V_self)
        outcome = (1 - lam_grid) * F_gen + lam_grid * bearer
        ag.plastic_update(outcome, eps, env._degradation())
        # food over the WHOLE lifetime: the agent forages to a setpoint then
        # coasts, so a tail-only window sees almost no eating. Selectivity (does
        # the agent preferentially eat resources good for ITS OWN body?) shows
        # up in cumulative intake, not steady-state.
        raw += env.last_raw
        taste += env.last_taste
        if t >= tail_start:
            acc -= eps; tcnt += 1
    diet = taste / (raw + 1e-6)                   # own-value per unit eaten
    out = torch.stack([acc / max(tcnt, 1), raw, taste, diet], dim=1)   # [P, 4]
    return out.cpu().numpy().reshape(nl, nm, len(METRICS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="champion_gpu_big.npz")
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--lifetime", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=90_000)  # held-out world band
    ap.add_argument("--out", default="results_bearer_gpu.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz = np.load(args.champion)
    d = int(npz["d"]) if "d" in npz else 64
    # match the world regime the champion was trained under
    drift = int(npz["drift_every"]) if "drift_every" in npz else 0
    homeo = bool(int(npz["observe_homeo"])) if "observe_homeo" in npz else True
    csig = float(npz["const_sigma"]) if "const_sigma" in npz else 0.5
    obp = float(npz["obligate_poison"]) if "obligate_poison" in npz else 0.0
    theta = torch.from_numpy(npz["theta"].astype(np.float32)).to(device)
    cfg = Config(seed=0, lambda_b=1.0, drift_every=drift, d=d, bearer_dim=d,
                 observe_homeo=homeo, const_sigma=csig, obligate_poison=obp)
    print(f"[world] drift_every={drift} ({'STATIC' if drift == 0 else 'DRIFTING'})"
          f" | observe_homeo={homeo} | const_sigma={csig}")
    obs_dim = G.BatchedForagerEnv(cfg, 1, 0, device).obs_dim
    spec = G.param_spec(cfg, obs_dim)
    torch.manual_seed(cfg.seed)
    from cca_milestone1 import PlasticAgent
    base = {n: p.detach().to(device) for n, p in PlasticAgent(cfg, obs_dim).named_parameters()}

    print(f"[bearer] champion={args.champion} d={d} lifetime={args.lifetime} "
          f"seeds={args.seeds} lambdas={args.lambdas}")

    nl, nm = len(args.lambdas), len(MODES)
    # perf[seed, lambda, mode, metric]
    allp = np.zeros((args.seeds, nl, nm, len(METRICS)))
    for s in range(args.seeds):
        allp[s] = eval_world(theta, cfg, spec, base, args.lambdas,
                             args.lifetime, args.seed0 + s, device)
        if (s + 1) % 4 == 0:
            print(f"  {s + 1}/{args.seeds} worlds")

    result = {"champion": args.champion, "d": d, "drift_every": drift,
              "lifetime": args.lifetime, "seeds": args.seeds,
              "lambdas": args.lambdas, "metrics": {}}
    for mi, metric in enumerate(METRICS):
        m = allp[:, :, :, mi]                            # [seed, lambda, mode]
        mean = m.mean(0)
        ci = 1.96 * m.std(0, ddof=1) / np.sqrt(args.seeds)
        dP = mean[:, 0] - mean[:, 1]                     # self - wrong
        dP_ci = 1.96 * (m[:, :, 0] - m[:, :, 1]).std(0, ddof=1) / np.sqrt(args.seeds)
        neutral = {"raw_food": "bearer-NEUTRAL",
                   "taste_food": "bearer-indexed (confounded w/ intake)",
                   "diet_quality": "bearer-INDEXED (clean selectivity)",
                   "homeostasis": "reactive-regulatable"}[metric]
        print(f"\n=== metric: {metric} ({neutral}) ===")
        print(f"{'lambda':>7} {'self':>18} {'wrong_bearer':>18} {'deltaP':>16}")
        for i, lam in enumerate(args.lambdas):
            print(f"{lam:>7.2f} {mean[i,0]:>+8.4f}+/-{ci[i,0]:.4f} "
                  f"{mean[i,1]:>+8.4f}+/-{ci[i,1]:.4f} "
                  f"{dP[i]:>+8.4f}+/-{dP_ci[i]:.4f}")
        ok0 = abs(dP[0]) <= max(3 * dP_ci[0], 1e-3)
        sig = [i for i in range(1, nl) if abs(dP[i]) > dP_ci[i]]
        if not ok0:
            verdict = f"falsifier FAIL (dP(0)={dP[0]:+.4f}); do not interpret"
        elif sig:
            allpos = all(dP[i] > 0 for i in sig)
            verdict = (f"bearer effect at lambda {[args.lambdas[i] for i in sig]}"
                       f"{' (self > wrong, predicted direction)' if allpos else ''}")
        else:
            verdict = "no effect above noise (dP within CI of 0 everywhere)"
        print(f"  falsifier dP(0)={dP[0]:+.4f}+/-{dP_ci[0]:.4f} | VERDICT: {verdict}")
        result["metrics"][metric] = {
            "mean_self": mean[:, 0].tolist(), "mean_wrong": mean[:, 1].tolist(),
            "deltaP": dP.tolist(), "deltaP_ci": dP_ci.tolist(),
            "falsifier_ok": bool(ok0)}

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
