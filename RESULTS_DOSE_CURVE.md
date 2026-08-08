# The constitution-strength dose curve: the last lever, and a clean negative

Run 2026-08-02. The culminating test of Bearer Theory Prediction 2 in the CCA
forager. Three constitution strengths, each a full train + ablation gate +
bearer sweep, roughly 50 minutes per point. `dose_curve.py`,
`dose_curve_results.json`.

## The hypothesis

Every prior bearer sweep found the self-versus-wrong_bearer effect null, and the
diagnosis (RESULTS_BEARER_NOHOMEO) was that selectivity is never incentivized:
at const_sigma = 0.5, eating the wrong resources is not harmful enough, so the
agent regulates by timing and quantity rather than by choosing resources suited
to its own body. The dose-curve prediction was that raising const_sigma, so that
some resources become strong own-poison, would force bearer-specific selectivity,
lift self diet_quality above 1.0 (a diet better than random for the own body),
and make the self-versus-wrong gap grow positive. This is the const_sigma dose
curve the manuscript's Prediction 2 anticipates.

## Result: the prediction was falsified

```
sigma   gate   self_dq@1   dP_diet@1   max|dP|   trained_adv
0.50   +0.44     0.632      -0.150      0.246       +0.239
1.00   +0.30     0.747      -0.107      0.302       +0.184
2.00   +0.26     0.577      -0.258      0.598       +0.145
```

Three things, all against the prediction.

1. **self diet_quality never crossed 1.0.** It went 0.63, 0.75, 0.58:
   non-monotonic and, at the strongest constitution, lower than at the weakest.
   The agent did not become taste-selective at any strength. It continued to eat
   a diet at or below random value for its own body.
2. **dP_diet never turned positive.** At full bearer indexing (lambda = 1) it was
   -0.150, -0.107, -0.258. Flat to slightly negative, never the predicted
   positive growth. The dose relation dG/dlambda > 0 is absent across the curve.
3. **The agent trained worse and depended on the pathway less as sigma rose.**
   trained_adv fell 0.239, 0.184, 0.145 and the gate gap fell 0.44, 0.30, 0.26.
   Strong poison makes the task harder, but the agent still passed the gate and
   beat random at every strength, so the null is interpretable.

## The mechanism, sharpened

Even when the environment made bearer-specific selectivity survival-relevant, the
agent did not develop it. It kept a bearer-neutral survival strategy: eat enough
good resources to offset the poison, regulate intake by timing and quantity, and
never learn a stable resource-type avoidance keyed to its own private body. The
private constitution is not observable, so avoiding own-poison would require
learning a type-to-avoid map from outcomes within each drifting lifetime through
the plastic pathway. This class of agent does not learn that map strongly enough
to change its diet, because a coarser bearer-neutral policy remains viable.

The refinement to Prediction 2 is therefore stronger than the earlier one. It is
not enough that selectivity be load-bearing in principle, nor even that the
environment reward it. The task must be one in which no bearer-neutral strategy
suffices, so the agent is forced to act on its own private constitution. Where a
quantity-or-timing policy can keep the bearer viable, the self-versus-yoked gap
stays flat at every constitution strength, even though the outcome function V
differs sharply across bearers.

## How this bears on the theory

This is the strongest form of the criterion-internal reading. The continuant
probe's +0.909 appeared only when the agent was scored on the very quantity its
self-condition optimized. The forager, across a fourfold range of constitution
strength, never reproduces a bearer advantage on a task-grounded metric, because
the task always leaves a bearer-neutral path to viability. The bearer advantage
is not a task performance effect in this substrate at any strength. It is a
criterion-internal alignment that appears when the score is the agent's own
developed criterion and not otherwise. Two substrates and a full dose curve now
agree on this.

Bearer identity is genuinely causally present in the outcome (V differs by 36% of
signal, corr 0.79) and the update pathway is load-bearing (the gate passes
throughout). What does not follow, and what the dose curve was the last chance to
produce, is a bearer-neutral performance gain. It does not appear. The wall
between being changed by outcomes and having those outcomes matter for one's own
coupling stands where the paper put it.

## Honest bounds

- The diet_quality metric is noisy. The lambda = 0 values of dP, which should be
  zero by construction, ran to +/- 0.25 to 0.60, so its noise floor at 16 seeds
  is large. The decisive, low-variance signal is the self diet_quality absolute,
  which never crosses 1.0. That the agent is not selective is robust; the exact
  dP values are not.
- Three sigma points and one lineage. A denser curve, more seeds, or a task
  redesigned so that no bearer-neutral strategy survives would each sharpen this.
  The last of those is the natural next experiment: a foraging or metabolic task
  with obligate selectivity, where eating own-poison is fatal and cannot be
  offset by quantity.
