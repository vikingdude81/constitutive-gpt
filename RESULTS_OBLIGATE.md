# Obligate selectivity: closing the last bearer-neutral escape (and a measurement wall)

Run 2026-08-02. The forager was modified so that eating a resource that is poison
for the agent's OWN body damages integrity, a channel that eating more good food
cannot restore (Config.obligate_poison). The intent: force bearer-specific
avoidance with no bearer-neutral substitute, the one design the dose curve had
not tried. const_sigma = 2.0, obligate_poison = 0.1, drift 200, sighted agent.

## Result: no bearer effect on the clean metric; the sharp metric is noise-limited

Training succeeded (advantage +0.186 over random) and the ablation gate passed
(+0.254), so the setup is valid. The 24-seed bearer sweep:

**homeostasis (rate-independent, least noisy):** the cleanest metric, and it is
null. dP (self - wrong) = +0.04, -0.00, +0.05, +0.06, -0.00 across lambda, every
value within CI, falsifier dP(0) = +0.04. A faint positive lean at mid-lambda,
nothing significant. This is the metric that would show the effect if obligate
poison worked, since eating own-poison lowers integrity, and it shows nothing.

**food metrics (raw, taste, diet_quality):** uninterpretable at this power. Their
lambda = 0 values of dP, which must be zero by construction, ran +0.20 to +0.43.
diet_quality for self came out 0.80 here versus 0.42 from the driver on the same
champion, a swing that is pure seed noise. The "significant" flags the script
prints are that noise. No conclusion can be drawn from the food metrics.

## Two hypotheses tested and rejected

- **"The agent eats less to avoid poison."** Not supported. raw_food for self
  (~0.4 to 0.8) is similar to the non-obligate runs. The agent did not
  dramatically cut intake.
- **"Obligate poison forces selectivity."** Not supported at this strength.
  diet_quality is ~0.8 to 1.0 (near random) for BOTH self and wrong_bearer, so
  neither condition developed bearer-selective foraging, and the two are
  therefore indistinguishable on homeostasis. The agent kept integrity adequate
  without learning to identify and avoid its own poison, so the manipulation had
  nothing to differentiate.

## The cumulative picture

Across six task designs the bearer effect never appeared as a task-grounded
performance gain, and each time a bearer-neutral escape was closed the agent kept
its coupling adequate some other way:

```
static world      -> pathway idle, no adaptation needed
drift world       -> regulate by timing
blind agent       -> still regulate by timing through the pathway
dose curve        -> offset poison with quantity
obligate poison   -> keep integrity adequate without full selectivity
```

Bearer identity remained causally real throughout (V differs by 36%, incorporation
is measurable, the gate passes every time), but it did not convert into a
bearer-neutral performance advantage under any of these designs.

**Epistemic caution (important).** This is not a falsification of obligate bearer
dependence. In each design, and in this obligate-poison run especially, the
intended mediator did not arise: neither self nor wrong-bearer agents developed
bearer-selective behavior (diet_quality ~ random for both). The intended causal
chain was private constitution -> bearer-selective behavior -> neutral advantage.
Because the middle term never appeared, the experiment cannot say whether
bearer-selective behavior would have produced the advantage. The correct
description is a FAILED INDUCTION of obligate dependence, not a decisive
disconfirmation of it. The philosophical reading "identity is what a system is,
not an edge it has" is retained as an interpretation, not presented as the direct
empirical conclusion. What the data establish is narrower and still important:
bearer identity can be causally real without producing a gain under the neutral
criteria tested.

## Two honest limits, and the real next levers

1. **Measurement.** The bearer-indexed food metric is too noisy to be decisive
   (its noise floor at 24 seeds rivals any plausible effect). A clean diet_quality
   read would need several times more seeds, or a lower-variance bearer-indexed
   score. The reliable homeostasis metric is null.
2. **Strength.** obligate_poison = 0.1 did not make poison fatal enough to force
   selectivity; the agent tolerated moderate integrity loss. A stronger setting,
   or a true balance task (the agent must consume a specific profile keyed to its
   body, where both under-eating and mis-eating are lethal and neither timing nor
   quantity substitutes), is the sharper obligate design.

The open question is whether a properly obligate task (one satisfying the
Generic-Policy Escape criterion in CONSOLIDATION_ESCAPE.md, where every
bearer-neutral policy is nonviable for some bearer while a bearer-correct policy
keeps every bearer viable) induces the mediator and then the advantage. That
experiment, the private-metabolism task, has not yet been run. Until it is, the
effect is neither confirmed nor ruled out; what is established is the escape
principle, which explains why the previous designs produced no advantage.
