"""
CCA Milestone 1 -- Full Experiment Driver
=========================================

One command from nothing to the headline figure:

  # end-to-end pipeline validation (a few minutes, CPU):
  python run_experiment.py --env forager --quick

  # the real thing (overnight on a many-core desktop):
  python run_experiment.py --env forager --train \
      --generations 100 --pop 64 --lifetime 2000 \
      --seeds 20 --steps 200000 --workers 24

  # PD-social variant (x-axis becomes lambda_social):
  python run_experiment.py --env social --train --seeds 20 --steps 200000

Outputs in --out-dir (default results_<env>/):
  champion.npz        ES-meta-trained parameters
  results.csv         one row per (lambda, mode, seed) cell
  headline_figure.png self vs wrong-bearer (+stream) with 95% CI,
                      plus the Delta P(lambda) existence-proof panel

Conditions per cell (B is always the measured agent):
  self          B updates on its own bearer-relative outcome
  wrong_bearer  PRIMARY yoked: same event, same dice, consequence valued
                through the wrong constitution / wrong social identity
  stream        classic yoked control (wrong event AND wrong bearer)

Built-in sanity: at lambda = 0, self and wrong_bearer are identical by
construction in the forager, so Delta P(0) must be ~0.  If it isn't,
something is broken -- the driver checks and warns.

GPU note: these nets are tiny; throughput comes from CPU parallelism
(--workers), not CUDA.  The GPU pays off when you port eval to the
batched-population pattern (hoffman_pytorch_scale.py style).
"""

from __future__ import annotations

# CPU-only guard (see cca_milestone1.py): a per-worker CUDA context on a large
# GPU reserves its VRAM in system RAM under Windows/WDDM and OOMs the box.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import torch

torch.set_num_threads(1)

MODES = ("self", "wrong_bearer", "stream")


# ----------------------------------------------------------------------------
# One cell = one (lambda, mode, seed) lifetime pair
# ----------------------------------------------------------------------------

def run_cell(args) -> dict:
    env_name, lam, mode, seed, steps, theta, scope = args
    torch.set_num_threads(1)

    if env_name == "forager":
        from cca_milestone1 import Config, run_pair
        cfg = Config(seed=seed, lambda_b=lam)
        runner = lambda **kw: run_pair(cfg, steps, **kw)
    else:
        from pd_social_env import SocialConfig, run_pair_social
        cfg = SocialConfig(seed=seed, lambda_b=1.0, lambda_social=lam)
        runner = lambda **kw: run_pair_social(cfg, steps, **kw)

    init_fn = None
    if theta is not None:
        from es_outer_loop import set_flat_params
        t_theta = torch.from_numpy(theta).float()
        init_fn = lambda ag: set_flat_params(ag, t_theta, scope)

    log = runner(mode=mode, log_every=max(1, steps // 200), init_fn=init_fn)
    tail = max(1, int(len(log.perf_B) * 0.25))
    return {
        "env": env_name, "lambda": lam, "mode": mode, "seed": seed,
        "perf_B": float(np.mean(log.perf_B[-tail:])),
        "perf_A": float(np.mean(log.perf_A[-tail:])),
        "dP_B_last": float(log.dP_B[-1]),
    }


# ----------------------------------------------------------------------------
# Aggregation + figure
# ----------------------------------------------------------------------------

def summarize(rows: list[dict], lambdas, modes):
    out = {}
    for lam, mode in product(lambdas, modes):
        v = np.array([r["perf_B"] for r in rows
                      if r["lambda"] == lam and r["mode"] == mode])
        if len(v):
            out[(lam, mode)] = (v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))
                                if len(v) > 1 else 0.0)
    return out


def make_figure(rows, lambdas, modes, env_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summarize(rows, lambdas, modes)
    xlab = "lambda_b (bearer indexing)" if env_name == "forager" \
        else "lambda_social (relationship indexing)"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = {"self": "tab:blue", "wrong_bearer": "tab:red",
              "stream": "tab:gray"}
    for mode in modes:
        xs = [l for l in lambdas if (l, mode) in s]
        ys = [s[(l, mode)][0] for l in xs]
        es = [s[(l, mode)][1] for l in xs]
        ax1.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                     label=mode, color=colors.get(mode))
    ax1.set_xlabel(xlab)
    ax1.set_ylabel("homeostatic performance (-eps, tail mean)")
    ax1.set_title(f"{env_name}: B's performance by coupling condition")
    ax1.legend()

    if "self" in modes and "wrong_bearer" in modes:
        xs = [l for l in lambdas
              if (l, "self") in s and (l, "wrong_bearer") in s]
        dp = [s[(l, "self")][0] - s[(l, "wrong_bearer")][0] for l in xs]
        de = [np.hypot(s[(l, "self")][1], s[(l, "wrong_bearer")][1]) for l in xs]
        ax2.errorbar(xs, dp, yerr=de, marker="s", capsize=3, color="tab:purple")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xlabel(xlab)
        ax2.set_ylabel("Delta P = self - wrong_bearer")
        ax2.set_title("Existence-proof panel: Delta P(lambda)")
        if xs and abs(xs[0]) < 1e-9 and abs(dp[0]) > max(3 * de[0], 1e-3):
            print(f"[warn] Delta P(0) = {dp[0]:+.4f} != 0 -- at lambda=0 self "
                  "and wrong_bearer should coincide; investigate before "
                  "trusting the rest of the curve.")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[fig] wrote {path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["forager", "social"], default="forager")
    ap.add_argument("--out-dir", default=None)
    # meta-training
    ap.add_argument("--train", action="store_true",
                    help="run ES meta-training first (else use --champion)")
    ap.add_argument("--champion", default=None,
                    help="path to existing champion.npz (skips training)")
    ap.add_argument("--scope", choices=["meta", "all"], default="all")
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--lifetime", type=int, default=2000)
    # sweep
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true",
                    help="tiny end-to-end validation of the whole pipeline")
    args = ap.parse_args()

    out_dir = args.out_dir or f"results_{args.env}"
    os.makedirs(out_dir, exist_ok=True)

    if args.quick:
        args.train = True
        args.generations, args.pop, args.lifetime = 2, 8, 300
        args.lambdas = [0.0, 1.0]
        args.modes = ["self", "wrong_bearer"]
        args.seeds, args.steps = 2, 1500
        args.workers = min(4, os.cpu_count() or 1)

    # --- champion -----------------------------------------------------------
    theta = None
    if args.champion:
        theta = np.load(args.champion)["theta"].astype(np.float32)
        print(f"[champion] loaded {args.champion} (dim={theta.size})")
    elif args.train:
        from es_outer_loop import run_es
        champ_path = os.path.join(out_dir, "champion.npz")
        print("[train] ES meta-training ...")
        theta = run_es(args.env, args.scope, args.generations, args.pop,
                       args.lifetime, sigma=0.02, lr=0.05,
                       workers=args.workers, out_path=champ_path
                       ).astype(np.float32)
        print(f"[train] champion -> {champ_path}")
    else:
        print("[warn] no --train / --champion: sweeping with RANDOM weights "
              "(pipeline test only; results scientifically meaningless)")

    # --- sweep --------------------------------------------------------------
    jobs = [(args.env, lam, mode, 1000 + s, args.steps, theta, args.scope)
            for lam, mode, s in product(args.lambdas, args.modes,
                                        range(args.seeds))]
    print(f"[sweep] {len(jobs)} cells "
          f"({len(args.lambdas)} lambdas x {len(args.modes)} modes x "
          f"{args.seeds} seeds), {args.steps} steps each, "
          f"{args.workers} workers")
    t0 = time.time()
    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, r in enumerate(pool.map(run_cell, jobs)):
                rows.append(r)
                if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                    print(f"  {i + 1}/{len(jobs)} cells "
                          f"({time.time() - t0:.0f}s)")
    else:
        for i, j in enumerate(jobs):
            rows.append(run_cell(j))
            print(f"  {i + 1}/{len(jobs)}")

    # --- outputs ------------------------------------------------------------
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {csv_path}")

    make_figure(rows, args.lambdas, args.modes, args.env,
                os.path.join(out_dir, "headline_figure.png"))

    s = summarize(rows, args.lambdas, args.modes)
    print("\nsummary (perf_B mean +/- 95% CI):")
    for lam in args.lambdas:
        line = f"  lambda={lam:4.2f}  "
        for mode in args.modes:
            if (lam, mode) in s:
                m, e = s[(lam, mode)]
                line += f"{mode}={m:+.4f}±{e:.4f}  "
        print(line)


if __name__ == "__main__":
    main()
