"""
Constitutive GPT: seed replicates of the self versus wrong-bearer effect (M3).

The M3 headline rests on one trained policy and evolutionary search was noisy late,
so a reviewer's first question is whether the effect is robust to training
randomness. This trains R independent policies with different ES seeds and reports
the self versus wrong-bearer gap and the falsifier isolation across seeds, with mean
and standard deviation, so the result carries error bars.

Each policy is trained from the same sensible init but a different ES noise stream,
then evaluated with the M3 bearer sweep on a common held-out instance set.

Usage:  python constitutive_gpt_replicates.py --reps 6 --gens 130
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse, json, time
import numpy as np
import torch

from constitutive_gpt_m2 import make_spec, es_train
from constitutive_gpt_m3 import rollout_bearer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--N", type=int, default=1500, help="shorter horizon for cheaper replicates")
    ap.add_argument("--M", type=int, default=48)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=130)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--evalM", type=int, default=256)
    ap.add_argument("--out", default="results_replicates.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = make_spec(args.K, args.d)
    chance = 1.0 / args.K

    print(f"[REPLICATES] {args.reps} seeds  K={args.K} eps={args.eps} d={args.d} N={args.N} "
          f"gens={args.gens}  chance={chance:.3f}  device={device}")
    rows = []
    for r in range(args.reps):
        t0 = time.time()
        theta, best = es_train(spec, args.K, args.d, args.eps, args.N, args.M, args.pop,
                               args.gens, device, log=lambda *a: None, seed=1000 + r)
        th = theta.unsqueeze(0)
        self0 = rollout_bearer(th, spec, args.K, args.d, args.eps, args.N, args.evalM, 555, device, 0.0, "indep")
        wrong1 = rollout_bearer(th, spec, args.K, args.d, args.eps, args.N, args.evalM, 555, device, 1.0, "indep")
        fals1 = rollout_bearer(th, spec, args.K, args.d, args.eps, args.N, args.evalM, 555, device, 1.0, "same")
        gap = self0 - wrong1
        isolate = fals1 - wrong1
        rows.append({"seed": 1000 + r, "self": self0, "wrong": wrong1, "falsifier": fals1,
                     "gap": gap, "isolate": isolate, "trained": best})
        print(f"  seed {1000+r}: self(0)={self0:.3f}  wrong(1)={wrong1:.3f}  gap={gap:+.3f}  "
              f"falsifier(1)={fals1:.3f}  isolate={isolate:+.3f}  (train {best:.3f}, {time.time()-t0:.0f}s)")

    def ms(key):
        v = np.array([row[key] for row in rows])
        return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
    gm, gs = ms("gap"); im, isd = ms("isolate")
    sm, ss = ms("self"); wm, ws = ms("wrong")
    n = len(rows)
    print(f"\n[SUMMARY over {n} seeds]")
    print(f"  self (lambda=0)          {sm:.3f} +/- {ss:.3f}")
    print(f"  wrong-bearer (lambda=1)  {wm:.3f} +/- {ws:.3f}   (chance {chance:.3f})")
    print(f"  gap (self - wrong)       {gm:.3f} +/- {gs:.3f}")
    print(f"  falsifier isolation      {im:.3f} +/- {isd:.3f}")
    all_positive = all(row["gap"] > 0.2 for row in rows)
    ci95 = 1.96 * gs / np.sqrt(n) if n > 1 else 0.0
    print(f"  gap 95% CI (normal approx): {gm:.3f} +/- {ci95:.3f}  -> [{gm-ci95:.3f}, {gm+ci95:.3f}]")
    print(f"  every seed shows self >> wrong (gap > 0.2): {'YES' if all_positive else 'no'}")

    json.dump({"rows": rows, "gap_mean": gm, "gap_std": gs, "gap_ci95": ci95,
               "isolate_mean": im, "isolate_std": isd, "self_mean": sm, "wrong_mean": wm,
               "chance": chance, "reps": n, "all_positive": bool(all_positive)},
              open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
