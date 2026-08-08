# Lever A: viability-threshold calibration

*Run 2026-08-05. Tests whether softening the obligate viability threshold lets the
curriculum's partial pathway-selectivity register as viable, producing a trained
agent on which the ablation gate is informative. Framework: BEARER_OBSERVABILITY.md
section 6, lever A. No em-dashes.*

---

## The knob

A `--gain-scale` (module GAIN) multiplier on the net-energy function:

    net(k) = CAL * GAIN * (r_own[k] * K - 1)

The zero-crossing stays pinned at uniform (r*K = 1), so uniform eating still nets
zero per meal and starves against the 0.004/step maintenance cost. The escape is NOT
reopened. A larger GAIN only amplifies the reward for above-uniform selection, so a
modest own-demand alignment can accrue energy faster and hold the homeostatic band
E in [0.4, 0.8].

## Setup

Curriculum K=2 (blindness ramped 0 to 1, evaluated fully blind), pop 768, 100
generations, const_sigma 1.5. Identical to the baseline curriculum run except for
GAIN. Baseline (GAIN=1, prior run): mediator 0.656 at full blind, advantage +0.003,
gate FAIL (floor artifact).

## Result: GAIN = 2.0

    advantage       +0.006   (NOT TRAINED; random floor)
    mediator align   0.567   (uniform 0.500, ceiling 0.761)  PRESENT but weak
    gate ON-OFF     +0.002   (FAIL)
    bearer sweep    no effect above noise at any lambda

The held-out blind score sat at the random floor (about -0.39) for the entire run,
gen 0 through 99, including the early phase when training was still sighted. The
sighted-trained policy does not transfer to blind evaluation at all, and once
training goes blind the fixed-Hebbian incorporated pathway never climbs above the
random floor.

## Reading

Gain-scaling at 2x did not reach the window. The binding constraint is not the
energy threshold, it is learnability under blindness. Two facts localize this:

1. The blind held-out score is at the random floor throughout, not just at the end.
   An energy-threshold problem would show a competent-but-sub-band agent (foraging
   well, energy just short of the band). Instead the agent is at random on the blind
   eval from the start, so there is no competent trajectory for a gentler threshold
   to rescue.
2. The achieved alignment (0.567) is only marginally above uniform, so the net
   energy per meal is tiny (about 0.016 at 2x). Amplifying a near-zero net stays
   near zero. The agent has not learned strong enough selectivity for energy scaling
   to matter.

This matches the framework's ordering: the cleaner remaining lever is B (a
meta-learned inner-loop update rule), because the failure is that the fixed Hebbian
update cannot acquire the blind inference, not that the viable band is too steep.

## Result: GAIN = 5.0

    advantage       +0.027   (off the floor, still below the +0.05 trained bar)
    mediator align   0.542   (uniform 0.500, ceiling 0.761)  ABSENT
    gate ON-OFF     +0.032   (PASS)
    bearer sweep    no effect above noise, large CIs at every lambda

The higher gain lifted the agent partly off the random floor (held-out -0.356 vs
rand -0.382) and the gate now passes. But the mediator collapsed to near-uniform:
the agent reaches partial viability by eating a barely-above-uniform diet in high
volume, amplified by the large gain, not by concentrating on its own high-demand
types. The gate passes on a route that is not bearer-selective, and the sweep
confirms no self-versus-wrong effect.

## Verdict: Lever A closed, the squeeze moves to the calibration axis

The two gain points bracket the same observability tension, now expressed through
the calibration knob itself.

    GAIN = 2  (strict)   selective (0.567) but sub-viable, gate dark
    GAIN = 5  (soft)     viable enough to pass the gate, but selectivity collapses
                         to near-uniform (0.542), gate passes on a quantity route

Softening the threshold enough to reach viability re-admits a quasi-uniform,
high-volume quantity route, which washes selectivity out (mediator absent).
Keeping it strict enough to force selectivity leaves the agent sub-viable (gate
dark). The clean window, viable AND selective AND on the incorporated pathway, is
not reached at either gain. Lever A does not open the window; it reproduces the
tension on a new axis.

This is consistent with, and sharpens, the run's diagnosis. The binding constraint
is that the fixed-Hebbian inner loop cannot acquire strong blind selectivity. When
the threshold is soft enough for weak selectivity to be viable, weak selectivity is
indistinguishable from a quantity escape. Only a learner that produces genuinely
strong selectivity would make viability and selectivity coincide. That is Lever B
(a meta-learned inner-loop update rule), the indicated next step.

