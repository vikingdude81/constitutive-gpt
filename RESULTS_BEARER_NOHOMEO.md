# Bearer sweep #4 (blind agent): the prediction falsified, and the clean negative

Run 2026-07-25, the culminating cycle. Removed the homeostatic channels
(E, I, eps) from the ACTION observation so the agent cannot regulate reactively
and must rely on the outcome-driven bearer/plastic pathway. Prediction (from
RESULTS_BEARER_CRITERION.md): this would uncap taste selectivity, push
diet_quality above 1.0 for self-coupled agents, and make self beat wrong_bearer
significantly - reproducing the CPU probe's +0.909.

**The prediction was falsified. This is a clean negative.**

## What held (the setup was ideal)

- Blind agent still TRAINED: +0.344 over random on held-out worlds, as well as
  the sighted agents. It regulates entirely through the learned pathway.
- Ablation gate PASSED bigger than ever: plastic ON -0.486 vs OFF -0.821,
  diff +0.335 (vs the sighted drift agent's +0.24). With no reactive fallback,
  the agent depends MORE on the plastic/bearer pathway. The bearer manipulation
  had maximal surface to act on.

## What failed (the prediction)

Bearer sweep, 32 worlds, all four metrics. diet_quality (the clean
bearer-INDEXED selectivity metric):

```
lambda   self              wrong_bearer      deltaP
0.00   +0.7528+/-0.1569  +0.8522+/-0.1253  -0.0994+/-0.1794
0.25   +0.8141+/-0.1225  +0.8261+/-0.1639  -0.0120+/-0.1664
0.50   +0.7668+/-0.1271  +0.9457+/-0.1350  -0.1789+/-0.1873
0.75   +0.8642+/-0.1495  +0.8343+/-0.1330  +0.0299+/-0.1591
1.00   +0.8084+/-0.1500  +0.8106+/-0.1527  -0.0023+/-0.1778
```

Two things, both against the prediction:

1. **diet_quality stayed ~0.8**, at or below 1.0 (random-diet value), even
   blind. The agent still does NOT preferentially eat resources good for its
   own body. Removing the reactive shortcut did not create taste selectivity.
2. **self vs wrong_bearer is null**, signs scattered/negative, all within CI.
   The faint positive lean seen in sweep #3 (drift, sighted) did NOT survive;
   it was noise. All four metrics null; falsifier OK on each.

## What the blind agent is actually doing

It reaches full performance and depends on the plastic pathway (gate +0.335),
but diet_quality ~ random means it is not regulating by WHICH resources it eats.
It regulates by WHEN and HOW MUCH: the learned pathway tells it to eat when its
(unobservable) energy is low, and it eats whatever is available. Homeostasis
only requires enough total energy, not the right resources, so taste
selectivity is never strongly incentivized - and the bearer-identity of the
outcome, which could only matter through selectivity, does not affect
performance.

## The robust conclusion of the CCA thread

Four sweeps, escalating the necessary conditions, each removing a confound:

1. Static world: null, uninterpretable (plastic pathway idle).
2. Drift world: interpretable null on the bearer-neutral yardstick.
3. Drift + bearer-indexed criteria: faint positive lean on diet_quality
   (later shown to be noise).
4. Drift + blind (no reactive shortcut): null everywhere, gate passing
   maximally. diet_quality still ~random.

Bearer-identity of consequence produces NO measurable bearer-neutral
performance effect in this forager, even when consequence-driven adaptation is
necessary, load-bearing, and the only channel available. This is a genuine
negative, reached by exhausting the obvious confounds rather than by a weak
test.

## How this relates to the paper (it supports the honest framing)

The paper (sec 8.4) already qualifies its bearer result as criterion-internal:
the +0.909 appeared only when scoring on the agent's own assigned criterion, and
was null on the bearer-neutral world-food yardstick (+0.002, 55%). This GPU
forager robustly reproduces the NULL side and does not reproduce the +0.909
side. The likely reason is structural: the CPU probe's +0.909 came from scoring
the agent on the very taste-filtered quantity its self-condition optimized (a
near-tautological criterion-internal alignment). The forager decouples
regulation (timing) from taste, so no such alignment exists, and the effect
vanishes. That is evidence the bearer advantage is criterion-internal and
fragile, not a robust performance gain - exactly what the paper claims, now with
an independent substrate confirming the careful version and refusing to inflate
it.

## The one remaining untested lever

Selectivity is never incentivized because eating own-poison is not harmful
enough at const_sigma=0.5. Raising const_sigma (more extreme private
constitution, strongly harmful poison resources) would force the agent to avoid
own-poison to survive, making taste selectivity load-bearing. Whether a bearer
effect appears there - and how it scales with const_sigma (the paper's dose
curve) - is the final open dimension. It needs retraining per const_sigma to
keep the agent in distribution. Deferred.
