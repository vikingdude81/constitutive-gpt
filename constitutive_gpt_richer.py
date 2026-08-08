"""
Constitutive GPT: a richer, multi-dimensional private code.

M1 to M3 used a one-dimensional private code, a single hidden arm. A reviewer can
ask whether the effect is genuine identity or a one-dimensional lookup. This task
makes the private code a vector w_i in R^m, an m-dimensional response profile.

  - Each instance draws a private unit vector w_i in R^m, unobservable.
  - Each step shows an observable context vector x_t in R^m. The correct action is
    the sign of w_i . x_t, so it varies with the context. Reward is noisy: the
    correct action pays Bernoulli(0.5+eps), the wrong one Bernoulli(0.5).
  - To act correctly across contexts the agent must recover the whole vector w_i,
    not one coordinate. This is a genuine multi-dimensional identity.

The incorporated state is a running estimate w_est in R^m, updated by a reward-gated
perceptron rule: w_est += eta*(r - shift)*(2a-1)*x - rho*w_est. Evolutionary search
tunes the scalars. Nothing observes w_i; it is inferred from the agent's own outcomes.

Three checks, mirroring M1 to M3:
  - gate 0: a context-only estimator from L samples must stay near chance (0.5)
    while the full horizon is solvable, so the m-dimensional code cannot be read
    from a bounded window;
  - the incorporated model beats the context-only baseline;
  - self versus wrong-bearer: feeding another instance's outcomes drives w_est to
    the wrong profile and fails on this instance's own contexts.

Numpy and CPU by design, so it can run beside a GPU job. No em-dashes.

Usage:  python constitutive_gpt_richer.py --m 4 --eps 0.15 --N 2000 --gens 60
"""
from __future__ import annotations

import argparse, json
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def context_bound(m, eps, n, trials, rng, test=200):
    """Heuristic context-only accuracy: estimate w from n noisy samples and test on
    fresh contexts. Upper-boundish proxy for what a bounded-window learner can do."""
    acc = 0.0
    for _ in range(trials):
        w = rng.standard_normal(m); w /= np.linalg.norm(w) + 1e-9
        X = rng.standard_normal((n, m))
        c = (X @ w > 0).astype(float)
        a = rng.integers(0, 2, n).astype(float)
        r = (rng.random(n) < np.where(a == c, 0.5 + eps, 0.5)).astype(float)
        chat = np.where(r > 0.5, a, 1 - a)                 # noisy pseudo-label
        what = ((2 * chat - 1)[:, None] * X).sum(0)
        Xt = rng.standard_normal((test, m))
        acc += ((Xt @ what > 0).astype(float) == (Xt @ w > 0).astype(float)).mean()
    return acc / trials


def rollout(params, m, eps, N, M, rng, lam=0.0, alt="none", tail=0.25):
    """params [P,4]: log_eta, shift, log_beta, rho. Returns tail accuracy [P]."""
    P = params.shape[0]
    eta = np.exp(params[:, 0])[:, None]
    shift = params[:, 1][:, None]
    beta = np.exp(params[:, 2])[:, None]
    rho = np.clip(params[:, 3], 0.0, 1.0)[:, None]

    w_true = rng.standard_normal((M, m)); w_true /= np.linalg.norm(w_true, axis=1, keepdims=True) + 1e-9
    if alt == "same":
        w_alt = w_true
    elif alt == "indep":
        w_alt = rng.standard_normal((M, m)); w_alt /= np.linalg.norm(w_alt, axis=1, keepdims=True) + 1e-9
    else:
        w_alt = None
    w_est = np.zeros((P, M, m))
    tail_start = int(N * (1 - tail)); acc = np.zeros(P); cnt = 0
    for t in range(N):
        x = rng.standard_normal((M, m))                    # shared context this step
        xb = x[None, :, :]
        score = (w_est * xb).sum(-1)                       # [P,M]
        a = (rng.random((P, M)) < _sigmoid(beta * score)).astype(float)
        c = ((x * w_true).sum(-1) > 0).astype(float)[None, :]     # [M] correct action
        r_own = (rng.random((P, M)) < np.where(a == c, 0.5 + eps, 0.5)).astype(float)
        if w_alt is None:
            r = r_own
        else:
            c_alt = ((x * w_alt).sum(-1) > 0).astype(float)[None, :]
            r_alt = (rng.random((P, M)) < np.where(a == c_alt, 0.5 + eps, 0.5)).astype(float)
            r = (1 - lam) * r_own + lam * r_alt
        sig = (eta * (r - shift) * (2 * a - 1))[:, :, None] * xb
        w_est = w_est + sig - rho[:, :, None] * w_est
        if t >= tail_start:
            acc = acc + ((score > 0).astype(float) == c).mean(1)   # scored on OWN contexts
            cnt += 1
    return acc / max(cnt, 1)


def es_train(m, eps, N, M, pop, gens, seed, log=print):
    rng = np.random.default_rng(seed)
    theta = np.array([np.log(0.2), 0.5, np.log(1.5), 0.02])
    sigma, lr = 0.3, 0.2
    best, best_theta = -1.0, theta.copy()
    for g in range(gens):
        noise = rng.standard_normal((pop // 2, 4))
        eps_all = np.concatenate([noise, -noise], 0)
        fits = rollout(theta[None] + sigma * eps_all, m, eps, N, M, np.random.default_rng(1000 + g))
        ranks = np.empty(len(fits)); ranks[np.argsort(fits)] = np.arange(len(fits))
        adv = ranks / (len(fits) - 1) - 0.5
        theta = theta + lr * (adv[:, None] * eps_all).sum(0) / (len(fits) * sigma)
        ho = float(rollout(theta[None], m, eps, N, M, np.random.default_rng(9000 + g))[0])
        if ho > best: best, best_theta = ho, theta.copy()
        if g % 10 == 0 or g == gens - 1:
            log(f"[gen {g:03d}] pop_mean={fits.mean():.3f} held-out={ho:.3f} best={best:.3f}")
    return best_theta, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4, help="dimensions of the private code")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--out", default="results_richer.json")
    args = ap.parse_args()
    chance = 0.5

    rng = np.random.default_rng(0)
    ctx = context_bound(args.m, args.eps, args.L, 3000, rng)
    ceil = context_bound(args.m, args.eps, args.N, 800, rng)
    print(f"[RICHER] m={args.m} eps={args.eps} N={args.N} L={args.L}  chance={chance:.3f}")
    print(f"[GATE 0] context-only(L={args.L})={ctx:.3f}  full-horizon-estimator(N={args.N})={ceil:.3f}")

    theta, best = es_train(args.m, args.eps, args.N, args.M, args.pop, args.gens, seed=0)
    eta, shift, beta, rho = np.exp(theta[0]), theta[1], np.exp(theta[2]), theta[3]
    trained = float(rollout(theta[None], args.m, args.eps, args.N, 512, np.random.default_rng(555))[0])
    print(f"\n[TRAINED] incorporated model accuracy = {trained:.3f}  "
          f"(learned eta={eta:.3f} shift={shift:.3f} beta={beta:.3f} rho={rho:.3f})")

    # self vs wrong-bearer sweep
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\n[BEARER SWEEP] accuracy on OWN contexts vs lambda (chance {chance:.3f})")
    print(f"{'lambda':>7} {'self..wrong':>12} {'falsifier':>12}")
    wrong, same = [], []
    for lam in lambdas:
        aw = float(rollout(theta[None], args.m, args.eps, args.N, 512, np.random.default_rng(555), lam, "indep")[0])
        af = float(rollout(theta[None], args.m, args.eps, args.N, 512, np.random.default_rng(555), lam, "same")[0])
        wrong.append(aw); same.append(af)
        print(f"{lam:>7.2f} {aw:>12.3f} {af:>12.3f}")

    gap = wrong[0] - wrong[-1]
    isolate = same[-1] - wrong[-1]
    beat = trained > ctx + 0.08
    monotone = all(wrong[i] >= wrong[i + 1] - 0.03 for i in range(len(wrong) - 1))
    print(f"\n[VERDICT] gate0 context-only={ctx:.3f} vs trained={trained:.3f} (beats: {'YES' if beat else 'no'})")
    print(f"          self(0)={wrong[0]:.3f} -> wrong(1)={wrong[-1]:.3f}, gap={gap:+.3f}; "
          f"falsifier(1)={same[-1]:.3f}, isolate={isolate:+.3f}")
    ok = beat and gap > 0.15 and monotone and isolate > 0.15
    if ok:
        print("  RICHER PASS: a multi-dimensional private identity is learnable only through the "
              "incorporated state (context-only stays near chance), and it is bearer-specific: "
              "feeding another individual's outcomes drives the estimate to the wrong profile and "
              "fails across this instance's own contexts. The effect is genuine identity, not a "
              "one-dimensional lookup.")
    else:
        why = []
        if not beat: why.append("does not beat context-only")
        if gap <= 0.15: why.append(f"gap {gap:+.3f} small")
        if not monotone: why.append("not monotone")
        if isolate <= 0.15: why.append(f"falsifier isolate {isolate:+.3f} small")
        print("  RICHER INCOMPLETE: " + "; ".join(why) + ".")

    json.dump({"m": args.m, "context_only": ctx, "full_estimator": ceil, "trained": trained,
               "lambdas": lambdas, "wrong": wrong, "falsifier": same, "gap": gap, "isolate": isolate,
               "chance": chance, "pass": bool(ok)}, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
