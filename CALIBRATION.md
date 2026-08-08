# PD-social calibration: a saturation bug and its fix

Measured 2026-07-24. Same family as the earlier calibration failures in this
program (v0.4 yardstick inflation, v0.5 beta cancellation, v0.6 novelty_gain
30x too strong). Recording it so the next person does not rediscover it.

## Symptom

ES meta-training on the social environment did not learn. Across 20
generations the population mean sat flat at -0.89 and never moved. The
forager environment, same ES code and same hyperparameters, climbed from
-0.65 to -0.36 over the same budget.

```
[gen 000] mean=-0.8948 max=-0.8895 best=-0.8895
[gen 010] mean=-0.8935 max=-0.8890 best=-0.4338
[gen 019] mean=-0.8921 max=-0.8853 best=-0.4338
```

The `best=-0.4338` was a single generation-1 outlier that was never
reproduced. It was noise, not progress.

## Diagnosis

Both homeostatic variables were pinned at their extremes, so fitness barely
depended on behavior.

**Energy saturated at the ceiling.** With `payoff_scale=0.02` and
`passive_decay=0.004`, every outcome except being a sucker added energy:
mutual cooperation +0.056/step, defecting on a cooperator +0.096, even mutual
defection +0.016. Energy was a free lunch, so E ratcheted to the 1.0 clip and
stayed there. Since `band_E=(0.4,0.8)`, sitting at 1.0 is ABOVE the band and
costs a fixed 0.200 on eps that no action could reduce.

**Integrity collapsed to the floor.** A betrayal cost 0.03 while regeneration
returned 0.001 per step, a 30:1 ratio against recovery. Any nonzero
cooperation rate against a population containing defectors drove I to 0,
costing a further 0.695.

0.200 + 0.695 = 0.895, which is exactly the observed plateau. Every genome
scored the same because both terms were saturated.

The optimum was also degenerate. A fixed-policy probe showed the best
available behavior was simply to never cooperate:

```
always-coop     eps=0.895      always-defect   eps=0.200
always-abstain  eps=0.200      mixed           eps=0.894
```

That is a cliff, not a gradient. Any cooperation floored integrity, so the
whole population sat in the bad basin with nothing pointing out of it.

## Fix

Make energy something the agent must earn, and make integrity damage
survivable.

| parameter | was | now |
|---|---|---|
| `payoff_scale` | 0.02 | 0.006 |
| `passive_decay` | 0.004 | 0.012 |
| `sucker_integrity_hit` | 0.03 | 0.010 |
| `integrity_regen` | 0.001 | 0.004 |

The energy economy now has real tension:

```
mutual coop        dE = +0.0060/step
defect-on-coop     dE = +0.0180/step     <- defection pays
mutual defect      dE = -0.0060/step     <- but mutual defection bleeds
sucker             dE = -0.0120/step
abstain            dE = -0.0072/step     <- opting out is not free
```

## Result

The optimum moved inside the bands, which is what the environment was always
supposed to model: defect for energy, cooperate to protect integrity, and
trade the two off.

```
always-coop     eps=0.200   final E=1.00 I=1.00
always-defect   eps=0.182   final E=0.98 I=1.00
always-abstain  eps=0.400   final E=0.00 I=1.00
mixed           eps=0.026   final E=0.80 I=0.99   <- interior optimum
spread = 0.374
```

ES on the recalibrated environment is behavior-sensitive again. Population
mean moved from a dead -0.89 to a live -0.37 to -0.69 range, and absolute
fitness improved about fourfold.

Honest bound: this was verified over 12 generations. It establishes that the
saturation pathology is gone, not that ES converges well. Best fitness found
-0.1985 by generation 1 and held it for ten more generations, and the mean
trend is noisy. Convergence behavior needs a longer run to judge.

## Re-measuring

Run `python calibrate_social.py` after touching any of these parameters. It
drives the environment with fixed policies, so it isolates the homeostatic
economy from the agent. Three failure verdicts to watch for:

- spread near 0 means a flat landscape and ES cannot learn.
- a corner optimum means the agent just switches one behavior off.
- the best policy leaving E or I outside its band means a constant penalty
  the agent cannot avoid.

A note on the forager: it did not show this pathology and trains fine. The
calibration change here is scoped to `SocialConfig` and does not touch it.
