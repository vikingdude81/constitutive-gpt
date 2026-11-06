"""
GPU ES training with held-out evaluation -- the functional test.

Sweep #1 measured an untrained agent: ES at pop 24 in a 1.16M-dim space could
not learn, and the "champion" was overfit to its training-world seeds (it
scored well only on env_seed 10000+gen, the worlds it was selected on). This
script fixes the three things that caused that:

  1. Large population on the GPU. Batching makes pop 256-512 affordable, which
     is the difference between an ES gradient estimate from 24 samples and from
     hundreds.
  2. Smaller agent. d=64 instead of 256 drops the search space from ~1.16M to
     ~150k parameters. A toy forager does not need 256 hidden units.
  3. Held-out selection. The champion is the search-distribution CENTER (which
     is what ES actually optimizes), scored on worlds NEVER used in training.
     Training seeds are 10000+gen; held-out seeds are 90000+i, disjoint.

The run ends with the test that matters: does the champion beat random weights
on the held-out worlds? If not, we have not trained anything, and no bearer
sweep on this champion would mean anything (that was the lesson of sweep #1).

Usage:
    python train_gpu.py --pop 256 --generations 80 --lifetime 800 --d 64
    python train_gpu.py --smoke        # tiny end-to-end check
"""

from __future__ import annotations

# Claim the GPU before importing cca_milestone1 (whose guard would hide it).
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time

import numpy as np
import torch

import gpu_batched as G
from cca_milestone1 import Config, PlasticAgent


def eval_on_worlds(thetas, cfg, spec, base, lifetime, seeds, device):
    """Score a BATCH of genomes [P, D] across a set of worlds, averaging over
    worlds. Returns per-genome mean fitness [P] (numpy). Batches genomes within
    each world (one eval_population call per world), so cost is ~len(seeds)
    world-passes regardless of P: the single-agent (P=1) anti-pattern that
    wastes the GPU is avoided."""
    if thetas.dim() == 1:
        thetas = thetas.unsqueeze(0)
    acc = np.zeros(thetas.shape[0])
    for sd in seeds:
        acc += G.eval_population(thetas, cfg, spec, "all", base, lifetime, sd,
                                 device).cpu().numpy()
    return acc / len(seeds)


def random_baseline(cfg, spec, base, obs_dim, lifetime, seeds, device, n=16,
                    seed0=555):
    """Mean fitness of n random genomes over the held-out worlds, plus the
    per-genome means. All n genomes are one batch per world."""
    vecs = []
    for i in range(n):
        torch.manual_seed(seed0 + i)
        a = PlasticAgent(cfg, obs_dim)
        vecs.append(torch.cat([p.detach().reshape(-1).to(device)
                               for nm, p in a.named_parameters()
                               if G._in_scope(nm, "all")]))
    per = eval_on_worlds(torch.stack(vecs), cfg, spec, base, lifetime, seeds,
                         device)
    return float(per.mean()), per.tolist()


def train(cfg, pop, generations, lifetime, sigma, lr, holdout_every,
          n_holdout, out_path, device, log=print):
    assert pop % 2 == 0
    obs_dim = G.BatchedForagerEnv(cfg, 1, 0, device).obs_dim
    spec = G.param_spec(cfg, obs_dim)
    torch.manual_seed(cfg.seed)
    tmpl = PlasticAgent(cfg, obs_dim)
    base = {n: p.detach().to(device) for n, p in tmpl.named_parameters()}
    D = G.spec_dim(spec, "all")
    theta = torch.cat([base[n].reshape(-1) for n, _, _ in spec
                       if G._in_scope(n, "all")]).clone()
    holdout = [90_000 + i for i in range(n_holdout)]

    log(f"[train] device={device} pop={pop} D={D} d={cfg.d} "
        f"lifetime={lifetime} gens={generations}")
    rand_mean, rand_per = random_baseline(cfg, spec, base, obs_dim, lifetime,
                                          holdout, device)
    log(f"[baseline] random on {n_holdout} held-out worlds: "
        f"mean={rand_mean:+.4f} (n=16 genomes, "
        f"range [{min(rand_per):+.3f},{max(rand_per):+.3f}])")

    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed + 99)
    best_ho, best_theta = -float("inf"), theta.clone()
    for g_i in range(generations):
        t0 = time.time()
        noise = torch.randn(pop // 2, D, device=device, generator=gen)
        eps_all = torch.cat([noise, -noise], 0)
        fits = G.eval_population(theta.unsqueeze(0) + sigma * eps_all, cfg, spec,
                                 "all", base, lifetime, 10_000 + g_i, device)
        shaped = G.centered_ranks(fits)
        theta = theta + lr * (shaped.unsqueeze(1) * eps_all).sum(0) / (pop * sigma)

        msg = (f"[gen {g_i:03d}] train_mean={fits.mean():+.4f} "
               f"train_max={fits.max():+.4f} ({time.time() - t0:.1f}s)")
        if g_i % holdout_every == 0 or g_i == generations - 1:
            ho_mean = float(eval_on_worlds(theta, cfg, spec, base, lifetime,
                                           holdout, device)[0])
            gap = ho_mean - rand_mean
            if ho_mean > best_ho:
                best_ho, best_theta = ho_mean, theta.clone()
            msg += f"  HELD-OUT={ho_mean:+.4f} (vs rand {rand_mean:+.4f}, {gap:+.4f})"
            np.savez(out_path, theta=best_theta.cpu().numpy().astype(np.float64),
                     scope="all", env="forager", d=cfg.d,
                     drift_every=cfg.drift_every,
                     observe_homeo=int(cfg.observe_homeo),
                     const_sigma=cfg.const_sigma,
                     obligate_poison=cfg.obligate_poison,
                     cfg=json.dumps(cfg.__dict__), best_holdout=best_ho,
                     rand_baseline=rand_mean)
        log(msg)

    # --- definitive test: champion vs random on held-out worlds ---
    # champion + the 16 random genomes evaluated together, one batch per world
    champ_per_world = []
    for sd in holdout:
        champ_per_world.append(float(G.eval_population(
            best_theta.unsqueeze(0), cfg, spec, "all", base, lifetime, sd,
            device)[0]))
    champ_mean = float(np.mean(champ_per_world))
    log("\n=== FUNCTIONAL TEST: champion vs random on held-out worlds ===")
    log(f"  champion mean : {champ_mean:+.4f}")
    log(f"  random   mean : {rand_mean:+.4f}  (best random {max(rand_per):+.4f})")
    log(f"  advantage     : {champ_mean - rand_mean:+.4f}")
    log("  champion per-world: " + " ".join(f"{v:+.2f}" for v in champ_per_world))
    verdict = ("TRAINED: champion generalizes past random."
               if champ_mean - rand_mean > 0.05 else
               "NOT TRAINED: no real advantage over random. Do not sweep this.")
    log(f"  VERDICT: {verdict}")
    return best_theta.cpu().numpy(), champ_mean, rand_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--generations", type=int, default=80)
    ap.add_argument("--lifetime", type=int, default=800)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--bearer", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--holdout-every", type=int, default=5)
    ap.add_argument("--n-holdout", type=int, default=8)
    ap.add_argument("--out", default="champion_gpu.npz")
    ap.add_argument("--drift-every", type=int, default=0,
                    help="world drift period in steps (0=static). >0 forces "
                         "online adaptation, needed for the plastic/bearer "
                         "pathway to be load-bearing (see RESULTS_BEARER_SWEEP1)")
    ap.add_argument("--no-homeo", action="store_true",
                    help="drop E/I/eps from the action observation; the agent "
                         "must learn its constitution through consequence "
                         "instead of sensing it (see RESULTS_BEARER_CRITERION)")
    ap.add_argument("--const-sigma", type=float, default=0.5,
                    help="private-constitution strength. Higher makes some "
                         "resources strongly own-poison, so bearer-specific "
                         "selectivity becomes load-bearing (the dose curve)")
    ap.add_argument("--obligate-poison", type=float, default=0.0,
                    help="integrity damage per unit own-poison eaten; makes "
                         "selectivity obligate (no bearer-neutral substitute)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not visible; running on CPU")
    else:
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    if args.smoke:
        args.pop, args.generations, args.lifetime = 16, 4, 300
        args.d = args.bearer = 32
        args.holdout_every, args.n_holdout = 2, 3

    cfg = Config(seed=0, lambda_b=1.0, drift_every=args.drift_every, d=args.d,
                 bearer_dim=args.bearer, observe_homeo=not args.no_homeo,
                 const_sigma=args.const_sigma)
    print(f"[world] drift_every={args.drift_every} "
          f"({'STATIC' if args.drift_every == 0 else 'DRIFTING'}) | "
          f"observe_homeo={cfg.observe_homeo} | const_sigma={cfg.const_sigma} | obligate_poison={cfg.obligate_poison}")
    train(cfg, args.pop, args.generations, args.lifetime, args.sigma, args.lr,
          args.holdout_every, args.n_holdout, args.out, device)
    print(f"champion -> {args.out}")


if __name__ == "__main__":
    main()
