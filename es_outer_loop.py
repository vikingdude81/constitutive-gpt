"""
CCA -- ES Outer Loop (meta-training)
====================================

Wires the missing piece: evolution-strategies meta-training of the plastic
machinery, runnable against BOTH environments:

    python es_outer_loop.py --env forager --scope all  --generations 100
    python es_outer_loop.py --env social  --scope meta --generations 100
    python es_outer_loop.py --smoke        # tiny end-to-end check

Scopes:
  meta : evolve only the plasticity controller + FiLM + bearer cell
         (the spec's default; base W must then be trained separately, e.g.
         PPO, or left to the fast weights)
  all  : deep-neuroevolution baseline -- evolve every nn.Parameter (W
         included).  Simplest way to get competent agents without writing
         the PPO inner loop; fine at this parameter count (~1M).

Fast-weight buffers (P, e) are never part of the genome -- they are reset
at birth and written only by the lived plasticity loop, as required.

ES details: antithetic (mirrored) sampling, centered-rank fitness shaping,
common random numbers (all genomes in a generation share the same env
seed) to cut variance, vanilla SGD on the natural-gradient estimate.

Port target for real runs: this loop is deliberately structured like the
batched-population evaluation in crypto-ml-trading-system's GPU evolution
scripts (hoffman_pytorch_scale.py / massive_scale_evolution_50k.py) --
replace the ProcessPool with a batched-agents-on-GPU evaluator there.

After meta-training, load the champion into a deployment agent:

    theta = np.load("champion.npz")["theta"]
    agent = PlasticAgent(cfg, env.obs_dim)
    set_flat_params(agent, torch.from_numpy(theta).float(), scope)
    # then run_pair / run_pair_social for the self-vs-yoked experiment
"""

from __future__ import annotations

# CPU-only guard (see cca_milestone1.py): a per-worker CUDA context on a large
# GPU reserves its VRAM in system RAM under Windows/WDDM and OOMs the box.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import torch

from cca_milestone1 import Config, Constitution, PCFEnv, PlasticAgent

torch.set_num_threads(1)


# ----------------------------------------------------------------------------
# Flat parameter plumbing
# ----------------------------------------------------------------------------

def _modules_for_scope(agent: PlasticAgent, scope: str):
    if scope == "meta":
        return [agent.controller, agent.film, agent.bearer_cell]
    if scope == "all":
        return [agent]
    raise ValueError(scope)


def get_flat_params(agent: PlasticAgent, scope: str) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1)
                      for m in _modules_for_scope(agent, scope)
                      for p in m.parameters()])


@torch.no_grad()
def set_flat_params(agent: PlasticAgent, flat: torch.Tensor, scope: str):
    i = 0
    for m in _modules_for_scope(agent, scope):
        for p in m.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n
    assert i == flat.numel(), "theta size mismatch for scope"


# ----------------------------------------------------------------------------
# Fitness: one lifetime, self-coupled, mean(-eps) over the tail
# ----------------------------------------------------------------------------

def _build(env_name: str, cfg_dict: dict, env_seed: int):
    if env_name == "forager":
        cfg = Config(**{k: v for k, v in cfg_dict.items()
                        if k in Config.__dataclass_fields__})
        rng = np.random.default_rng(env_seed + 555)
        env = PCFEnv(cfg, Constitution.draw(cfg, rng), seed=env_seed)
    elif env_name == "social":
        from pd_social_env import PDSocialEnv, SocialConfig
        cfg = SocialConfig(**{k: v for k, v in cfg_dict.items()
                              if k in SocialConfig.__dataclass_fields__})
        env = PDSocialEnv(cfg, focal_id="A", seed=env_seed)
    else:
        raise ValueError(env_name)
    return cfg, env


def eval_genome(args) -> float:
    """Top-level (picklable) worker: build fresh agent, load theta, live."""
    theta, cfg_dict, env_name, scope, lifetime, env_seed, agent_seed = args
    torch.set_num_threads(1)
    cfg, env = _build(env_name, cfg_dict, env_seed)
    torch.manual_seed(agent_seed)
    agent = PlasticAgent(cfg, env.obs_dim)
    set_flat_params(agent, torch.from_numpy(theta).float(), scope)

    obs = env.obs()
    tail_start = int(lifetime * 0.75)
    tail = []
    for t in range(lifetime):
        a = agent.act(obs)
        obs, Fg, Vs, eps, _ = env.step(a)
        o = (1 - cfg.lambda_b) * Fg + cfg.lambda_b * Vs
        agent.plastic_update(o, eps, env._degradation())
        if t >= tail_start:
            tail.append(-eps)
    # plasticity-death guard: a dead loop cannot win by luck
    if agent.last_dP <= 0:
        return -1e6
    return float(np.mean(tail))


# ----------------------------------------------------------------------------
# ES loop
# ----------------------------------------------------------------------------

def centered_ranks(x: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(x)
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


def run_es(env_name: str, scope: str, generations: int, pop: int,
           lifetime: int, sigma: float, lr: float, workers: int,
           out_path: str, base_cfg: Config | None = None,
           log_fn=print) -> np.ndarray:
    assert pop % 2 == 0, "pop must be even (antithetic pairs)"
    cfg = base_cfg or Config()
    cfg_dict = asdict(cfg)

    # template agent defines theta's shape
    _, env0 = _build(env_name, cfg_dict, env_seed=0)
    torch.manual_seed(cfg.seed)
    template = PlasticAgent(cfg, env0.obs_dim)
    theta = get_flat_params(template, scope).numpy().astype(np.float64)
    log_fn(f"[es] env={env_name} scope={scope} dim={theta.size} "
           f"pop={pop} lifetime={lifetime}")

    best_fit, best_theta = -np.inf, theta.copy()
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    rng = np.random.default_rng(cfg.seed + 99)

    try:
        for g in range(generations):
            t0 = time.time()
            env_seed = 10_000 + g                    # common random numbers
            noise = rng.standard_normal((pop // 2, theta.size))
            eps_all = np.concatenate([noise, -noise])          # antithetic
            jobs = [(theta + sigma * e, cfg_dict, env_name, scope,
                     lifetime, env_seed, cfg.seed + g) for e in eps_all]
            jobs = [tuple([j[0].astype(np.float32)] + list(j[1:])) for j in jobs]
            if pool:
                fits = np.array(list(pool.map(eval_genome, jobs)))
            else:
                fits = np.array([eval_genome(j) for j in jobs])

            shaped = centered_ranks(fits)
            grad = (shaped[:, None] * eps_all).sum(0) / (pop * sigma)
            theta = theta + lr * grad

            gi = int(fits.argmax())
            if fits[gi] > best_fit:
                best_fit = float(fits[gi])
                best_theta = (theta if fits.mean() >= best_fit
                              else jobs[gi][0].astype(np.float64)).copy()
            np.savez(out_path, theta=best_theta, scope=scope,
                     env=env_name, cfg=json.dumps(cfg_dict))
            log_fn(f"[gen {g:03d}] mean={fits.mean():+.4f} "
                   f"max={fits.max():+.4f} best={best_fit:+.4f} "
                   f"({time.time() - t0:.1f}s)")
    finally:
        if pool:
            pool.shutdown()
    return best_theta


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["forager", "social"], default="forager")
    ap.add_argument("--scope", choices=["meta", "all"], default="all")
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--lifetime", type=int, default=2000)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--out", default="champion.npz")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end check (pop 8, 2 gens, lifetime 300)")
    args = ap.parse_args()

    if args.smoke:
        run_es(args.env, "all", generations=2, pop=8, lifetime=300,
               sigma=0.02, lr=0.05, workers=min(4, os.cpu_count()),
               out_path="champion_smoke.npz")
        print("smoke ok -- champion_smoke.npz written")
        return

    run_es(args.env, args.scope, args.generations, args.pop, args.lifetime,
           args.sigma, args.lr, args.workers, args.out)


if __name__ == "__main__":
    main()
