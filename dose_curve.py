"""
The constitution-strength dose curve: the culminating test of Prediction 2.

Hypothesis (from RESULTS_BEARER_NOHOMEO and the Bearer Theory audit): the
self-vs-wrong_bearer gap is absent because, at const_sigma = 0.5, eating the
wrong resources is not harmful enough, so bearer-specific selectivity is never
incentivized. Raise const_sigma so some resources are strong own-poison. Then
avoiding own-poison and seeking own-good becomes load-bearing for survival, and a
self-coupled agent (which learns its OWN body) should beat a wrong_bearer agent
(which learns a DIFFERENT body, whose good resources may be its own poison).

For each const_sigma the driver runs the full chain in one process:
  1. train a drift-world champion at that const_sigma (held-out selection),
  2. run the ablation gate (must pass, else the sweep is uninterpretable),
  3. run the bearer sweep and read diet_quality (the clean bearer-indexed metric).

It reports, per const_sigma: the gate gap, the self diet_quality absolute (does
it rise above 1.0 = selectivity emerging), and dP = self - wrong on diet_quality
(does the bearer effect appear and grow). Prediction: dP grows with const_sigma.

Usage:  python dose_curve.py --sigmas 0.5 1.0 2.0 --gens 90
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time

import numpy as np
import torch

import gpu_batched as G
from cca_milestone1 import Config, PlasticAgent
import train_gpu
import ablation_gate
import bearer_sweep_gpu as BS


def build(cfg, device):
    obs_dim = G.BatchedForagerEnv(cfg, 1, 0, device).obs_dim
    spec = G.param_spec(cfg, obs_dim)
    torch.manual_seed(0)
    base = {n: p.detach().to(device) for n, p in PlasticAgent(cfg, obs_dim).named_parameters()}
    return obs_dim, spec, base


def gate(theta, cfg, spec, base, obs_dim, device, seeds=10, life=1500):
    s0 = 90_000
    on = np.array([ablation_gate.run(theta, cfg, spec, base, obs_dim, s, True, life, device)[0]
                   for s in range(s0, s0 + seeds)])
    off = np.array([ablation_gate.run(theta, cfg, spec, base, obs_dim, s, False, life, device)[0]
                    for s in range(s0, s0 + seeds)])
    return float(on.mean() - off.mean())


def sweep_diet(theta, cfg, spec, base, device, lambdas, seeds=16, life=1500):
    allp = np.zeros((seeds, len(lambdas), 2, len(BS.METRICS)))
    for s in range(seeds):
        allp[s] = BS.eval_world(theta, cfg, spec, base, lambdas, life, 90_000 + s, device)
    di = BS.METRICS.index("diet_quality")
    m = allp[:, :, :, di]                      # [seed, lambda, mode]
    self_dq = m[:, :, 0].mean(0)               # self diet_quality per lambda
    dP = (m[:, :, 0] - m[:, :, 1]).mean(0)      # self - wrong per lambda
    dP_ci = 1.96 * (m[:, :, 0] - m[:, :, 1]).std(0, ddof=1) / np.sqrt(seeds)
    return self_dq, dP, dP_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--pop", type=int, default=768)
    ap.add_argument("--gens", type=int, default=90)
    ap.add_argument("--lifetime", type=int, default=1000)
    ap.add_argument("--drift-every", type=int, default=200)
    ap.add_argument("--sweep-seeds", type=int, default=16)
    ap.add_argument("--obligate-poison", type=float, default=0.0)
    ap.add_argument("--out", default="dose_curve_results.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"[dose] sigmas={args.sigmas} pop={args.pop} gens={args.gens} "
          f"lifetime={args.lifetime} drift={args.drift_every}")

    results = {"meta": vars(args), "lambdas": lambdas, "points": []}
    for csig in args.sigmas:
        t0 = time.time()
        cfg = Config(seed=0, lambda_b=1.0, drift_every=args.drift_every, d=64,
                     bearer_dim=64, observe_homeo=True, const_sigma=csig, obligate_poison=args.obligate_poison)
        obs_dim, spec, base = build(cfg, device)
        out = f"champion_sigma{csig}.npz"
        print(f"\n########## const_sigma={csig} ##########")
        theta_np, champ, rand = train_gpu.train(
            cfg, args.pop, args.gens, args.lifetime, 0.02, 0.05,
            holdout_every=20, n_holdout=6, out_path=out, device=device)
        theta = torch.from_numpy(theta_np.astype(np.float32)).to(device)
        g = gate(theta, cfg, spec, base, obs_dim, device)
        self_dq, dP, dP_ci = sweep_diet(theta, cfg, spec, base, device, lambdas,
                                        seeds=args.sweep_seeds, life=1500)
        sig_dP = [i for i in range(len(lambdas)) if abs(dP[i]) > dP_ci[i]]
        pt = {"const_sigma": csig, "trained_adv": champ - rand, "gate_gap": g,
              "self_diet_quality": self_dq.tolist(), "deltaP_diet": dP.tolist(),
              "deltaP_ci": dP_ci.tolist()}
        results["points"].append(pt)
        print(f"[sigma {csig}] trained_adv={champ-rand:+.3f} gate_gap={g:+.3f} "
              f"| self diet_q (lam1)={self_dq[-1]:.3f} "
              f"| dP_diet={[f'{x:+.3f}' for x in dP]} "
              f"| signif at lam {[lambdas[i] for i in sig_dP] or 'none'} "
              f"({time.time()-t0:.0f}s)")
        json.dump(results, open(args.out, "w"), indent=2)

    print("\n===== DOSE CURVE SUMMARY =====")
    print(f"{'sigma':>6} {'gate':>7} {'self_dq@1':>10} {'dP_diet@1':>10} {'max|dP|':>9}")
    for pt in results["points"]:
        dP = pt["deltaP_diet"]
        print(f"{pt['const_sigma']:>6.2f} {pt['gate_gap']:>+7.3f} "
              f"{pt['self_diet_quality'][-1]:>10.3f} {dP[-1]:>+10.3f} "
              f"{max(abs(x) for x in dP):>9.3f}")
    print(f"\nsaved {args.out}")
    print("PREDICTION: self_dq rises above 1.0 and dP_diet grows positive with sigma")


if __name__ == "__main__":
    main()
