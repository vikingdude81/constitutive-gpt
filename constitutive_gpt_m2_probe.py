"""
Constitutive GPT, M2 viability probe.

Before building the full incorporated transformer, answer the one de-risking
question: can ES learn to solve the obligate bandit-ID task (M1) THROUGH an
incorporated integrating state, beating the context-only baseline?

Minimal incorporated agent, deliberately stripped so only the incorporated state
can integrate (no recurrence, no context window at all):

  - incorporated state b in R^K, a per-arm value accumulator, reset each episode.
  - policy: a_t ~ softmax(beta * b), so the agent explores when b is flat and
    exploits as b separates.
  - online update from the agent's OWN outcome: b[a_t] += eta * (r_t - shift -
    rho * b[a_t]), a learned running estimate of each arm's payoff.
  - ES optimizes the update and policy scalars {eta, rho, shift, beta, b0}.

This is the cleanest test of the incorporated pathway: b is the only carrier of
information across steps, so any above-baseline accuracy is the incorporated state
doing the integration. A pass retires the task-and-ES risk and gives M2 a target;
the full M2 must then realize the same integration via fast-weight deltas and
attention rather than a hand-shaped accumulator. A fail is an early, cheap signal
that the update or the ES setup, not the transformer, is the bottleneck.

Usage:  python constitutive_gpt_m2_probe.py --K 8 --eps 0.06 --N 2000
"""
from __future__ import annotations

import argparse
import numpy as np

from constitutive_gpt_m1 import bayes_map_accuracy


def rollout_accuracy(params, K, eps, N, M, rng, tail=0.25):
    """Evaluate a batch of policies. params [P,5] -> tail accuracy [P].

    params columns: log_eta, rho, shift, log_beta, b0.  b is [P,M,K]; M shared
    instances per policy for low-variance comparison.
    """
    P = params.shape[0]
    eta = np.exp(params[:, 0]).reshape(P, 1, 1)
    rho = params[:, 1].reshape(P, 1, 1)
    shift = params[:, 2].reshape(P, 1, 1)
    beta = np.exp(params[:, 3]).reshape(P, 1, 1)
    b0 = params[:, 4].reshape(P, 1, 1)

    theta = rng.integers(0, K, size=M)                      # shared instances [M]
    theta_b = np.broadcast_to(theta, (P, M))
    b = np.full((P, M, K), 0.0) + b0                        # accumulator [P,M,K]
    tail_start = int(N * (1 - tail))
    hits = 0.0
    cnt = 0
    ar = np.arange(P)[:, None]
    am = np.arange(M)[None, :]
    for t in range(N):
        logits = beta * b                                   # [P,M,K]
        g = -np.log(-np.log(rng.random((P, M, K)) + 1e-12) + 1e-12)  # Gumbel
        a = np.argmax(logits + g, axis=-1)                  # sample softmax [P,M]
        correct = (a == theta_b)
        p = np.where(correct, 0.5 + eps, 0.5)
        r = (rng.random((P, M)) < p).astype(np.float64)
        # b[p,m,a] += eta*(r - shift - rho*b[p,m,a])
        cur = b[ar, am, a]
        b[ar, am, a] = cur + eta[:, :, 0] * (r - shift[:, :, 0] - rho[:, :, 0] * cur)
        if t >= tail_start:
            # per-policy accuracy of the CURRENT best-arm estimate, averaged over instances
            hits = hits + (np.argmax(b, axis=-1) == theta_b).mean(axis=1)
            cnt += 1
    return hits / max(cnt, 1)                               # [P]


def es_train(K, eps, N, M, pop, gens, seed, log=print):
    rng = np.random.default_rng(seed)
    dim = 5
    theta = np.array([np.log(0.1), 0.0, 0.5, np.log(2.0), 0.0])  # sensible init
    sigma, lr = 0.3, 0.2
    best, best_theta = -1.0, theta.copy()
    for g in range(gens):
        noise = rng.standard_normal((pop // 2, dim))
        eps_all = np.concatenate([noise, -noise], 0)
        cand = theta[None, :] + sigma * eps_all
        fits = rollout_accuracy(cand, K, eps, N, M, np.random.default_rng(1000 + g))
        # centered ranks
        ranks = np.empty(len(fits)); ranks[np.argsort(fits)] = np.arange(len(fits))
        adv = ranks / (len(fits) - 1) - 0.5
        theta = theta + lr * (adv[:, None] * eps_all).sum(0) / (len(fits) * sigma)
        ho = float(rollout_accuracy(theta[None, :], K, eps, N, M, np.random.default_rng(9000 + g))[0])
        if ho > best:
            best, best_theta = ho, theta.copy()
        if g % 5 == 0 or g == gens - 1:
            log(f"[gen {g:03d}] pop_mean={fits.mean():.3f} held-out={ho:.3f} best={best:.3f}")
    return best_theta, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--N", type=int, default=2000, help="probe horizon (shorter than the M1 N=4000 for speed)")
    ap.add_argument("--M", type=int, default=128, help="instances per policy")
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--L", type=int, default=32, help="context window, for the baseline comparison")
    args = ap.parse_args()

    chance = 1.0 / args.K
    ctx = bayes_map_accuracy(args.K, args.eps, args.L, 8000, 1)
    ceil = bayes_map_accuracy(args.K, args.eps, args.N, 8000, 3)
    print(f"[M2 probe] K={args.K} eps={args.eps} N={args.N}  chance={chance:.3f} "
          f"context-only(L={args.L})={ctx:.3f}  full-horizon(N={args.N})={ceil:.3f}")

    theta, best = es_train(args.K, args.eps, args.N, args.M, args.pop, args.gens, seed=0)
    eta, rho, shift, beta, b0 = np.exp(theta[0]), theta[1], theta[2], np.exp(theta[3]), theta[4]
    print(f"\n[BEST] incorporated-agent tail accuracy = {best:.3f}")
    print(f"       learned: eta={eta:.3f} rho={rho:.3f} shift={shift:.3f} beta={beta:.3f} b0={b0:.3f}")
    beat = best > ctx + 0.05
    strong = best > 0.5 * (ceil + chance)
    print(f"\n[VIABILITY] vs context-only {ctx:.3f}: {'BEATS' if beat else 'does NOT beat'}"
          f" | vs full-horizon ceiling {ceil:.3f}: {'strong' if strong else 'weak'}")
    if beat and strong:
        print("  PROBE PASS: an ES-tuned incorporated integrator solves the obligate task well "
              "above the context-only baseline. Task-and-ES risk retired; proceed to build M2 "
              "(fast-weight + attention realization) toward this target.")
    elif beat:
        print("  PROBE PARTIAL: beats the baseline but stays well below ceiling. Integrator works "
              "but ES tuning or horizon needs work before the full build.")
    else:
        print("  PROBE FAIL: does not beat the context-only baseline. The bottleneck is the update "
              "or the ES setup, not the transformer. Diagnose before building M2.")


if __name__ == "__main__":
    main()
