# Bearer sweep #2 (drift world): the first interpretable measurement

Run 2026-07-25 on a champion trained in a DRIFTING world, which is what makes
this the first bearer sweep whose result means anything. Sweep #1 (static
world) measured an agent whose plastic pathway was idle; its null was an
artifact. This one clears the gate.

## The gate passes (this is the key difference)

Before running the sweep, the ablation gate checked whether the outcome-driven
plastic/bearer pathway is load-bearing for THIS champion:

```
plastic ON  : -0.5544 +/- 0.1235
plastic OFF : -0.7979 +/- 0.1315     (fast weights + bearer state frozen)
ON - OFF    : +0.2434 +/- 0.1382      GATE: PASS
```

Freezing the pathway drops the agent from -0.55 nearly to the random baseline
(-0.75). Contrast the static champion, where the same ablation moved
performance by 0.005. Drift created a genuine need to keep adapting from one's
own outcomes, and the agent now depends on that pathway. This confirms the
conceptual prediction: bearer-indexing can only have purchase where online
adaptation is necessary.

## The sweep result: interpretable null

Champion d=64, drift_every=200, lifetime 2000, 32 worlds, paired within-world.

```
lambda      self            wrong_bearer      deltaP
0.00   -0.5337+/-0.0716  -0.5032+/-0.0619  -0.0304+/-0.0883
0.25   -0.5049+/-0.0612  -0.5606+/-0.0818  +0.0557+/-0.0768
0.50   -0.5616+/-0.0692  -0.5675+/-0.0687  +0.0059+/-0.0734
0.75   -0.5323+/-0.0697  -0.5721+/-0.0801  +0.0398+/-0.1007
1.00   -0.5776+/-0.0685  -0.5947+/-0.0841  +0.0172+/-0.1055
```

Falsifier passes: Delta P(0) = -0.030 +/- 0.088, consistent with 0.

No significant bearer effect at any lambda. One honest detail: at all four
lambda>0 points Delta P leans positive (self >= wrong_bearer), which is the
PREDICTED direction, but every value is within its CI of 0 and the magnitudes
are tiny (~+0.03 mean). It is a faint lean, not an effect. Do not over-read it.

## The manipulation was real, not weak

The obvious worry: maybe valuing through the wrong body is nearly identical to
valuing through the own body, so the null is trivial. It is not. Over 8 worlds
x 2000 steps:

```
corr(V_self, V_wrong)        : +0.79   (1.0 would mean no manipulation)
mean |V_self - V_wrong|      : 0.00064
ratio (difference / signal)  : 0.36
```

The wrong-body valuation differs from the own-body one by 36% of the signal
magnitude and is far from perfectly correlated. The manipulation genuinely
perturbs the learning signal. Performance simply does not care.

## Interpretation: this reproduces the paper's criterion-internal finding

This is not a dead end; it is a small-scale reproduction of the sharpest result
in the paper (arxiv draft section 8.4). There, the bearer advantage was
criterion-internal: it appeared only when performance was scored on the agent's
own assigned criterion (+0.909, 100% of seeds), and was NULL on a bearer-neutral
yardstick (world food, +0.002, 55%).

Here, performance is homeostatic (did E and I stay in band). That is a
bearer-NEUTRAL yardstick: a world-ish fact about whether the agent survived,
not a fact constituted by the bearer. On exactly that kind of yardstick, the
paper predicts and the continuant probe found NO bearer advantage. This sweep
reproduces it: mis-valuing consequences through the wrong body does not hurt
bearer-neutral performance, even when the agent depends on consequence-driven
adaptation and the wrong-body signal genuinely differs.

The mechanism is consistent: V_self and V_wrong agree in direction most of the
time (both driven by the same events, differently weighted), so the plastic
pathway learns similar resource preferences either way, and homeostatic
survival - a directional, world-ish outcome - is preserved.

## The next question, precisely located

The paper's result says the effect lives on the bearer-INDEXED criterion, not
the bearer-neutral one. This sweep scored the bearer-neutral criterion (-eps,
homeostasis) and found the predicted null. The open follow-up is to score the
bearer-INDEXED criterion - performance as valued through the agent's own
constitution/taste - and test whether Delta P becomes positive there, matching
the +0.909 the CPU continuant probe measured. That is a scoring change, not a
new agent: same drift champion, same sweep, different performance metric.

## Status of the pipeline

Fully validated and gated end to end:
- GPU batched evaluator (parity-checked)
- held-out training with a functional test (drift champion +0.31 over random)
- ablation gate (mandatory; PASS required before a sweep is interpretable)
- paired within-world bearer sweep with a built-in falsifier
- a manipulation-strength check (V_self vs V_wrong correlation)

Two clean results now stand: static-world sweep (uninterpretable, pathway
idle) and drift-world sweep (interpretable null on the bearer-neutral
criterion, reproducing the paper). The bearer-indexed-criterion sweep is the
next step.
