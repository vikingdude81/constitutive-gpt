# Constitutive GPT M2: viability probe

*Run 2026-08-06. De-risking probe before the full M2 build. Question: can ES learn
to solve the obligate bandit-ID task (M1) through an incorporated integrating state,
beating the context-only baseline? No em-dashes.*

---

## The probe

A minimal incorporated agent, stripped so only the incorporated state can integrate
(no recurrence, no context window): an incorporated accumulator b in R^K, policy
a ~ softmax(beta b), online update b[a] += eta (r - shift - rho b[a]). ES optimizes
the five scalars {eta, rho, shift, beta, b0}. b is the sole carrier across steps, so
any above-baseline accuracy is the incorporated state doing the integration.
Implementation: constitutive_gpt_m2_probe.py.

The probe uses the locked M1 task (K=8, eps=0.06) at a shorter horizon N=2000 for
speed.

## Result: PROBE PASS

    chance                         0.125
    context-only baseline (L=32)   0.170
    full-horizon uniform ceiling   0.680
    incorporated agent (ES)        0.721

    learned: eta=0.017 rho=0.263 shift=0.475 beta=1.380 b0=-0.130

The incorporated agent reaches 0.721, far above the 0.170 context-only baseline, and
it exceeds the uniform-exploration ceiling (0.680). It beats the uniform ceiling
because it explores adaptively: the softmax on b concentrates pulls on promising
arms, gathering more information per step than undirected exploration. ES discovered
the bandit algorithm cleanly: center reward near 0.5 (shift 0.475), keep a decaying
running estimate (eta, rho), exploit via a moderate softmax (beta 1.38).

## What this retires and what remains

Retired: the task is solvable by an incorporated integrator, and ES can find the
solution. The obligate construction from M1 does not make the task unlearnable, it
makes it learnable only through the incorporated state, which is exactly the design
goal. This is the property the forager program never had at threshold.

Not yet tested (this is M2 proper): the same integration realized through the actual
mechanism, a low-rank fast-weight delta on a small transformer with the Lever B
learned update rule and attention over the L-window, rather than a hand-shaped
scalar accumulator. The probe gives M2 a concrete target (about 0.72 at N=2000, and
the M1 ceiling of 0.87 at N=4000) and confirms the effort is worth spending.

The Lever A/B caveat still applies to the fast-weight realization: the mechanism
must LEARN to write the running per-arm estimate into b and read it back through the
policy, which is harder than tuning five scalars. The probe says the target exists
and is reachable in principle; M2 tests whether the fast-weight-plus-attention
substrate reaches it.
