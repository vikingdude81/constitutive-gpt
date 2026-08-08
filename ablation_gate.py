"""
Ablation gate: is the outcome-driven plastic/bearer pathway load-bearing?

This MUST pass before any bearer sweep on a champion is interpretable. The
bearer manipulation (self vs wrong_bearer) acts only through the outcome ->
fast-weight/bearer-state pathway. If freezing that pathway does not hurt
performance, the agent solved the task some other way (base recurrent policy),
the manipulation has no surface to act on, and a null Delta P is an artifact.

Bearer sweep #1 failed exactly this: on the static-world champion, plastic ON
-0.4265 vs OFF -0.4218 (diff 0.005). See RESULTS_BEARER_SWEEP1.md.

Reports plastic ON vs OFF on held-out worlds, using the SAME drift setting the
champion was trained under (read from the npz). Verdict:
  ON - OFF > ~0.03 and above the CIs  -> pathway load-bearing, sweep is valid
  else                                -> pathway idle, do NOT run/trust a sweep

Usage:  python ablation_gate.py --champion champion_gpu_drift.npz --seeds 12
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import numpy as np
import torch

import gpu_batched as G
from cca_milestone1 import Config, PlasticAgent


@torch.no_grad()
def run(theta, cfg, spec, base, obs_dim, seed, plastic, life, device):
    env = G.BatchedForagerEnv(cfg, 1, seed, device)
    params = G.unflatten_batch(theta.unsqueeze(0), spec, "all", base)
    ag = G.BatchedAgent(cfg, params, obs_dim, device)
    obs = env.obs(); acc = 0.0; cnt = 0; tail = int(life * 0.75)
    for t in range(life):
        a = ag.act(obs)
        obs, F, Vs, eps, Vw = env.step(a)
        if plastic:
            o = (1 - cfg.lambda_b) * F + cfg.lambda_b * Vs
            ag.plastic_update(o, eps, env._degradation())
        if t >= tail:
            acc -= float(eps.mean()); cnt += 1
    pnorm = sum(float(ag.P_fast[k].norm()) for k in ag.LAYERS)
    return acc / max(cnt, 1), pnorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=90_000)
    ap.add_argument("--lifetime", type=int, default=2000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz = np.load(args.champion)
    d = int(npz["d"]) if "d" in npz else 64
    drift = int(npz["drift_every"]) if "drift_every" in npz else 0
    homeo = bool(int(npz["observe_homeo"])) if "observe_homeo" in npz else True
    csig = float(npz["const_sigma"]) if "const_sigma" in npz else 0.5
    obp = float(npz["obligate_poison"]) if "obligate_poison" in npz else 0.0
    theta = torch.from_numpy(npz["theta"].astype(np.float32)).to(device)
    cfg = Config(seed=0, lambda_b=1.0, drift_every=drift, d=d, bearer_dim=d,
                 observe_homeo=homeo, const_sigma=csig, obligate_poison=obp)
    obs_dim = G.BatchedForagerEnv(cfg, 1, 0, device).obs_dim
    spec = G.param_spec(cfg, obs_dim)
    torch.manual_seed(0)
    base = {n: p.detach().to(device) for n, p in PlasticAgent(cfg, obs_dim).named_parameters()}

    print(f"[gate] champion={args.champion} d={d} drift_every={drift} "
          f"lifetime={args.lifetime} seeds={args.seeds}")
    seeds = range(args.seed0, args.seed0 + args.seeds)
    on = np.array([run(theta, cfg, spec, base, obs_dim, s, True, args.lifetime, device) for s in seeds])
    off = np.array([run(theta, cfg, spec, base, obs_dim, s, False, args.lifetime, device) for s in seeds])
    on_p, off_p = on[:, 0], off[:, 0]
    ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    diff = on_p.mean() - off_p.mean()
    diff_ci = 1.96 * (on_p - off_p).std(ddof=1) / np.sqrt(len(on_p))
    print(f"  plastic ON  : {on_p.mean():+.4f} +/- {ci(on_p):.4f}")
    print(f"  plastic OFF : {off_p.mean():+.4f} +/- {ci(off_p):.4f}")
    print(f"  ON - OFF    : {diff:+.4f} +/- {diff_ci:.4f}")
    print(f"  ||P_fast||  : {on[:,1].mean():.3f} (moves; but only matters if it helps)")
    passed = diff > 0.03 and diff > diff_ci
    print(f"\n  GATE: {'PASS - pathway load-bearing, bearer sweep is valid' if passed else 'FAIL - pathway idle, a bearer sweep would be uninterpretable'}")
    return passed


if __name__ == "__main__":
    main()
