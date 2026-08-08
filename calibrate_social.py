"""
Fixed-policy calibration probe for the PD-social environment.

The question this answers: is fitness behavior-sensitive, and is the optimum
interior or at a degenerate corner? ES can only learn if different policies
score differently AND the best policy is not a corner the agent reaches by
switching one behavior off.

It drives the env with FIXED actions (always cooperate / always defect /
always abstain / random) instead of a network, so the numbers isolate the
environment's homeostatic economy from any question about the agent.

Read the output like this:
  * spread ~0        -> flat landscape, ES cannot learn anything.
  * best is a corner -> degenerate; the agent just suppresses one action.
  * best is 'mixed'  -> interior optimum, which is what we want.
  * final E,I inside band_E/band_I for the best policy -> the agent can
    actually hold homeostasis, rather than eating a constant penalty.

Usage:  python calibrate_social.py
"""

from __future__ import annotations

import numpy as np

from pd_social_env import SocialConfig, make_pair_envs

POLICIES = [("always-coop", 0), ("always-defect", 4),
            ("always-abstain", 8), ("mixed", None)]


def probe(overrides: dict, steps: int = 1500, seeds: int = 3):
    """-> {policy: (mean_eps, final_E, final_I)}, spread"""
    out = {}
    for name, a10 in POLICIES:
        eps_v, E_v, I_v = [], [], []
        for seed in range(seeds):
            cfg = SocialConfig(**{**overrides, "seed": seed})
            env, _ = make_pair_envs(cfg, cfg.seed)
            rng = np.random.default_rng(seed)
            eps = 0.0
            for _ in range(steps):
                a = int(rng.integers(0, 10)) if a10 is None else a10
                _, _, _, eps, _ = env.step(a)
            eps_v.append(eps); E_v.append(env.E); I_v.append(env.I)
        out[name] = (float(np.mean(eps_v)), float(np.mean(E_v)),
                     float(np.mean(I_v)))
    spread = max(v[0] for v in out.values()) - min(v[0] for v in out.values())
    return out, spread


def main():
    base = dict(lambda_b=1.0, lambda_social=1.0, social_drift_every=0)
    cfg = SocialConfig(**base)
    print(f"bands: E {cfg.band_E}  I {cfg.band_I}")
    print(f"current: payoff_scale={cfg.payoff_scale} passive_decay={cfg.passive_decay} "
          f"sucker_hit={cfg.sucker_integrity_hit} regen={cfg.integrity_regen}\n")

    print("per-step energy drift by outcome:")
    for nm, pay in [("mutual coop R", cfg.R), ("defect-on-coop T", cfg.T),
                    ("mutual defect P", cfg.Pp), ("sucker S", cfg.S),
                    ("abstain", cfg.abstain_pay)]:
        print(f"  {nm:18s} dE = {cfg.payoff_scale * pay - cfg.passive_decay:+.4f}/step")

    res, spread = probe(base)
    print(f"\nfixed-policy eps (lower is better), spread = {spread:.3f}")
    for k, (e, E, I) in res.items():
        in_band = (cfg.band_E[0] <= E <= cfg.band_E[1] and
                   cfg.band_I[0] <= I <= cfg.band_I[1])
        print(f"  {k:15s} eps={e:.3f}   final E={E:.2f} I={I:.2f}"
              f"{'   <- both in band' if in_band else ''}")
    best = min(res.items(), key=lambda kv: kv[1][0])[0]
    print(f"\nbest policy: {best}")
    if spread < 0.05:
        print("VERDICT: landscape is flat. ES cannot learn. Recalibrate.")
    elif best != "mixed":
        print("VERDICT: optimum is a corner. Degenerate; prefer an interior optimum.")
    else:
        print("VERDICT: interior optimum with a real gradient. Good to train on.")


if __name__ == "__main__":
    main()
