# Forager sweep #1: null result, and why it is uninterpretable

Run 2026-07-24. 75 cells (5 lambdas x 3 modes x 5 seeds), 25,000 steps each,
12 workers, 28 minutes. Champion from 20 ES generations at pop 24.

**Headline: no bearer effect was detected, but the result does not bear on the
bearer question, because the agent was never trained.**

## What worked

The infrastructure is sound and this run validates it.

- The falsifier passes exactly. At lambda=0 the bearer-indexed component has
  zero weight, so `self` and `wrong_bearer` are identical by construction.
  Measured: both -0.9586 +/- 0.0509, agreeing to four decimals. The
  same-event/wrong-bearer plumbing is correct.
- The sweep completes in 28 minutes with flat memory, after the autograd leak
  fix (see README). Cost is linear and predictable at ~2.2 min/cell.

## The numbers

perf_B, mean +/- 95% CI:

```
lambda=0.00  self=-0.9586+/-0.0509  wrong_bearer=-0.9586+/-0.0509  stream=-0.9436+/-0.0572
lambda=0.25  self=-0.8875+/-0.1625  wrong_bearer=-0.9263+/-0.1229  stream=-0.9487+/-0.0497
lambda=0.50  self=-0.9426+/-0.0443  wrong_bearer=-0.9723+/-0.0456  stream=-0.9588+/-0.0623
lambda=0.75  self=-0.9676+/-0.0455  wrong_bearer=-0.9276+/-0.1049  stream=-0.8576+/-0.2070
lambda=1.00  self=-0.8969+/-0.1527  wrong_bearer=-0.9002+/-0.1555  stream=-0.9090+/-0.1272
```

Delta P = self - wrong_bearer: 0.0000, +0.0388, +0.0297, -0.0400, +0.0033.
Non-monotonic, sign flips at lambda=0.75, every value swamped by its CI.

## Why it is uninterpretable

**The champion is no better than random weights.** Evaluated against three
random genomes on four identical worlds:

```
world     champion    rand1     rand2     rand3
10004      -1.0369   -0.7975   -0.9999   -0.8799
10010      -0.4043   -1.0393   -1.0590   -1.0504
  777      -1.0496   -0.9940   -1.0637   -1.0416
 4242      -0.9718   -0.4090   -0.7648   -0.4000
mean       -0.8657   ---------- -0.8749 ----------
```

Training advantage: +0.0093. That is noise.

Note world 10010: the champion scores -0.40 there while every random genome
scores about -1.05. ES uses `env_seed = 10000 + generation`, so 10010 is the
generation-10 training world. The champion is overfit to the specific worlds
it was selected on and does not generalize.

Two further checks that rule out competing explanations:

- **Not long-horizon collapse.** Within a 25,000-step run, performance is flat:
  first 10% -0.8799, last 25% -0.8789, deciles bouncing -0.74 to -1.02 with no
  trend. The agent does not degrade, it simply never was good.
- **Not the ES "improvement" it appeared to be.** The log shows best going
  -0.65 -> -0.36, but `best` is a running maximum while the world changes every
  generation. Population mean stayed near -0.9 throughout and `max` swung
  -0.36 to -0.95 between generations. That is lucky world draws, not learning.

## Root cause

ES with population 24, 20 generations, in a **1,157,265-dimensional** space.
A gradient estimate from 24 samples in a million dimensions. Published ES at
this parameter count uses populations in the high hundreds to thousands and
far more iterations. The compute budget was never sufficient to train an
agent, so the sweep measured an untrained one.

## What this changes

1. **The GPU batched evaluator is now the critical path**, not an optimization.
   Batching the population is what makes pop 512-1024 affordable, which is the
   difference between ES that cannot learn and ES that can.
2. **Shrink the agent.** `d=256` drives most of the 1.16M parameters. `d=64`
   brings it near 100k, a far more reasonable ES target for a toy forager.
3. **Hold out evaluation worlds.** Champion selection is contaminated by
   training seeds, as world 10010 shows. Score champions on worlds never used
   in training.

Until an agent demonstrably beats random on held-out worlds, no sweep result
should be read as evidence about bearer-indexing in either direction.
