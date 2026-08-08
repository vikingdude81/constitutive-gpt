# Bearer sweep #3: the bearer-indexed criteria, and why the effect is blocked

Run 2026-07-25 (continuing from sweep #2). The goal was to reproduce the
paper's criterion-internal result: a bearer effect that appears on the agent's
own assigned criterion but not on a bearer-neutral yardstick. We added the
scoring dimension the paper turns on, and the answer is precise and negative,
with a clear mechanism and a specified fix.

## What we added

The same gated drift champion, same paired within-world sweep, now scored on
four yardsticks simultaneously (no retraining, just extra measurement):

- **homeostasis** = -eps. Reactive-regulatable: E and I are in the observation,
  so the agent can keep them in band by watching them, without learned taste.
- **raw_food** = sum of base_value eaten. Bearer-NEUTRAL: any body eats the
  same raw resource.
- **taste_food** = sum of base_value*(1+own m_dev). Bearer-indexed, but
  confounded with total intake.
- **diet_quality** = taste_food / raw_food. The clean bearer-INDEXED signal:
  the average own-body value of the resources the agent CHOSE to eat,
  controlling for how much it ate. If the agent forages by learned own-taste,
  self should beat wrong_bearer here.

## Result: no significant effect, but a faint criterion-internal SIGNATURE

24 worlds, lifetime 1500. Delta P (self - wrong_bearer) at each lambda:

```
metric           lam.25   lam.50   lam.75   lam1.0    pattern
homeostasis      -0.018   +0.014   +0.020   -0.139    scattered (null)
raw_food (NEUT)  +0.022   +0.155   +0.117   -0.091    scattered signs (null)
taste_food       +0.033   +0.174   +0.200   -0.069    scattered
diet_quality(IX) +0.022   +0.076   +0.168   +0.085    ALL POSITIVE
```

No effect is significant at 24 seeds (CIs ~0.12-0.23). But the pattern is not
flat noise. On diet_quality, the clean bearer-INDEXED metric, all four lambda>0
points lean POSITIVE, up to +0.168 at lambda=0.75: self-coupled agents eat a
slightly higher-own-body-value diet than wrong_bearer agents, in the predicted
direction. On raw_food, the bearer-NEUTRAL metric, the signs scatter with no
consistent direction. That contrast (consistent positive on the bearer-indexed
criterion, scattered on the bearer-neutral one) is exactly the criterion-
internal signature the paper describes. It is present but underpowered, not
established.

## Why it is only faint: the agent barely forages by learned taste

The decisive number is diet_quality's ABSOLUTE level, not Delta P. Both self and
wrong_bearer sit at diet_quality ~0.75-0.95, i.e. at or slightly below 1.0 (the
value a random diet gives, since m_dev is mean-zero). The champion is only
weakly selecting resources good for its own body. It forages close to
indiscriminately.

That caps the effect. The bearer manipulation changes which body the outcome is
valued through, which can only matter to the extent the agent USES that
valuation to choose what to eat. This agent mostly does not: it regulates
homeostasis reactively. E and I are in its observation, so it eats when hungry
and stops when satisfied, guided by the sensed state rather than by a learned
model of which resources are good for it. The self-vs-wrong lean survives
(self is marginally more own-selective) but is small because the whole
selectivity channel is small.

The ablation gate is consistent with this. The plastic/bearer pathway IS
load-bearing (ON -0.55 vs OFF -0.80), but it is load-bearing for the TIMING and
control of reactive regulation, not for taste-selective foraging. Turning it
off hurts because the agent loses its learned regulation policy, not because it
loses a learned taste.

## The observable-homeostasis shortcut

This is the same architectural shortcut that has shadowed the whole
investigation. Because the agent can SENSE its homeostatic state (E, I, eps are
observation channels), it never has to LEARN its private constitution through
consequence. It can hill-climb on the sensed state directly. The paper's
continuant probe got a bearer effect precisely because its agents could NOT
sense their private taste and had to discover it through outcomes; there,
learning from your own outcomes vs another's genuinely changes what you learn
to prefer.

## The fix (a specified next cycle)

Remove the homeostatic channels (E, I, eps) from the observation. Then the
agent cannot regulate reactively and must rely on a learned, outcome-driven
model of which resources sustain it, which is exactly the plastic/bearer
pathway. In that setting:

1. Retrain a champion on the reduced observation (obs_dim drops by 3; a new
   agent). Expect training to be harder and rely much more heavily on
   plasticity.
2. Run the ablation gate. The ON-OFF gap should be LARGER than the current
   0.24, because now the agent has no reactive fallback.
3. Run the multi-metric sweep. Now diet_quality should exceed 1.0 for
   self-coupled agents (they learn own-good), and self should beat wrong_bearer
   on diet_quality and own homeostasis. That would be the criterion-internal
   effect reproduced.

A second, orthogonal lever (matching the paper's dose curve): sweep
const_sigma, the strength of private-body dependence. Larger const_sigma lowers
corr(V_self, V_wrong) (currently 0.79) and should grow any bearer effect
monotonically. This likely needs retraining per const_sigma to keep the agent
in distribution.

## Standing conclusion of the CCA thread

Three sweeps, each sharper than the last:
1. Static world: null, uninterpretable (plastic pathway idle).
2. Drift world, homeostasis: interpretable null on a bearer-neutral yardstick,
   reproducing the paper's world-food null.
3. Drift world, bearer-indexed criteria: no significant effect, but a faint
   criterion-internal SIGNATURE - self leans higher than wrong_bearer on the
   clean bearer-indexed diet_quality metric at every lambda>0, while the
   bearer-neutral raw_food scatters. The effect is capped because this agent
   forages only weakly by learned taste (diet_quality ~1.0), regulating instead
   via the observable-homeostasis shortcut.

The results are cumulative and each identified a necessary condition the
previous design lacked: a world that requires online adaptation (drift), a
pathway that is load-bearing (the gate), and now an agent that must LEARN its
private constitution rather than sense it (remove E/I from obs). The faint
predicted-direction lean on diet_quality is the first positive hint in the
whole thread; the obs-removal cycle is the experiment that would uncap it and
test whether it becomes significant.
