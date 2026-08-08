# Constitutive GPT: seed replicates of the self versus wrong-bearer effect

*Run 2026-08-08. Puts error bars on the M3 headline. The single-policy result and a
noisy late-training curve invite the question of robustness to training randomness;
this trains six independent policies and reports the effect across them. No em-dashes.*

---

## Setup

Six policies trained from the same init with independent evolutionary-search noise
streams (seeds 1000 to 1005), then each evaluated with the M3 self versus wrong-bearer
sweep on a common held-out instance set. K=8, eps=0.06, d=32, horizon 1500,
population 96, 130 generations. Implementation in constitutive_gpt_replicates.py.

## Result: robust across seeds

    seed    self(0)   wrong(1)    gap     falsifier(1)  isolate
    1000     0.654     0.128    +0.527      0.645       +0.517
    1001     0.701     0.129    +0.572      0.693       +0.563
    1002     0.630     0.122    +0.507      0.613       +0.491
    1003     0.626     0.132    +0.494      0.594       +0.461
    1004     0.546     0.122    +0.424      0.491       +0.369
    1005     0.637     0.107    +0.530      0.611       +0.504

    across 6 seeds (mean +/- sd):
      self (lambda=0)          0.632 +/- 0.051
      wrong-bearer (lambda=1)  0.123 +/- 0.009    (chance 0.125)
      gap (self - wrong)       0.509 +/- 0.049
      falsifier isolation      0.484 +/- 0.066
      gap 95% CI               [0.469, 0.548]
      every seed shows self >> wrong (gap > 0.2): YES

## Reading

The effect is robust to training randomness. All six seeds show self beating
wrong-bearer by a wide margin, the gap is 0.509 with a standard deviation of 0.049
and a tight 95% confidence interval, and wrong-bearer lands on exact chance every
time, with a standard deviation of 0.009. The falsifier isolation is likewise stable
near 0.48. The original single-run gap of 0.515 sits in the middle of the
distribution, so it was representative rather than a lucky seed.

This is the error bar a reviewer asks for first. The self versus wrong-bearer
non-transferability of the incorporated state is a stable property of the trained
model, not an artifact of one run. Combined with the richer multi-dimensional code
(RESULTS_RICHER.md), where the same non-transferability holds for a continuous
m-dimensional identity, the bearer effect is now shown to be both robust and not a
one-bit artifact.
