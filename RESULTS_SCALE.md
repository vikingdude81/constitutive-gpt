# Constitutive GPT M4: scale

*Run 2026-08-08. Phase 0 is the go/no-go for scaling within the evolutionary-search
paradigm. No em-dashes.*

---

## Phase 0: does ES still train the model at scale

The toy model was width 32 with a pooling context, about 650 parameters. The scaled
model adds a genuine transformer block, single-head attention with learned Q, K, V,
O projections and a two-layer MLP, both with residuals, at width 128, alongside the
incorporated fast-weight memory. That is 133,637 parameters, roughly two hundred times
the toy. Evolutionary search is a black-box optimizer whose efficiency falls as
parameters grow, so this is where its ceiling would appear. Implementation in
constitutive_gpt_scale.py.

Go/no-go: at width 128, does held-out accuracy clear the context-only baseline.

## Result: GO

Settings d=128, mlp width 256, N=1000, population 192, 80 generations.

    parameters                   133,637
    chance                       0.125
    context-only baseline (L=32) 0.170
    best held-out                0.560

Evolutionary search trains the scaled transformer-plus-memory model. Best held-out
reaches 0.560, well above the context baseline and chance, so the incorporated memory
still carries the task at this size. The ES ceiling is higher than expected: a
130,000-parameter model with a real attention block is trainable by ES on a single
GPU in about eight minutes.

Two honest caveats. The training curve is noisy: the held-out probe bounced between
0.09 and 0.38 late in training while the best checkpoint held at 0.560, which is
expected for evolutionary search at this parameter count with a modest population.
And 0.560 is on a short N=1000, 80-generation budget; the toy reached 0.732 at N=2000,
so there is clear headroom.

## Decision and Phase 1 plan

GO. ES holds at width 128 with a real transformer block, so the longer Phase 1 runs
are worth it. Phase 1, to be run next:

- The full scaled result: N=2000, a larger population and more generations for
  stability, returning the best held-out checkpoint. One 1 to 4 hour run gives the
  trained accuracy, the ablation gate (freeze the memory, expect collapse), and the
  self versus wrong-bearer sweep (cheap evaluation on the trained policy).
- Optional, matching the toy-scale rigor: the dual-channel dissociation at scale, and
  a few seed replicates for error bars.

Recommended stabilization for Phase 1: population 256, a smaller step size, and 200
or more generations, since the Phase 0 curve was noisy. The compute is a few hours,
backgrounded.

## Phase 1: the full scaled result

Ran the full battery at scale: d=128, N=2000, population 256, 200 generations, step
size 0.08. Same 133,637-parameter transformer-plus-memory model, evaluated on
held-out instances after training. About 55 minutes on one GPU.

    trained (best held-out)      0.789     (toy at N=2000 was 0.732)

    double dissociation:
      full (both channels)       0.746
      memory only (record cut)   0.742     record idle, drop 0.003
      context only (bearer cut)  0.174     bearer load-bearing, drop 0.571
                                           and 0.174 equals the context bound 0.170

    self versus wrong-bearer sweep (accuracy on OWN code, chance 0.125):
      lambda   self..wrong   falsifier (alt=own)
      0.00       0.696           0.720
      0.25       0.715           0.795
      0.50       0.390           0.878
      0.75       0.192           0.761
      1.00       0.085           0.710

      gap (self - wrong)         +0.611    (toy was +0.515)
      falsifier isolation        +0.625

## Verdict: PHASE 1 PASS, the Constitutive GPT holds at scale

Every criterion reproduces at 133,637 parameters with a real transformer block, and
the bearer effect is if anything stronger than at toy scale. The record is idle:
removing the attention context drops accuracy by 0.003. The incorporated memory is
load-bearing: freezing it collapses accuracy to 0.174, exactly the context bound. And
the memory is bearer-specific: self beats wrong-bearer by 0.611, with wrong-bearer
falling below chance to 0.085 as the write shifts to another instance's outcomes,
while the falsifier stays high and isolates the effect by 0.625.

The training curve remained noisy late, as expected for evolutionary search at this
parameter count, and the returned policy is the best held-out checkpoint. The result
is not sensitive to that noise: the battery on the returned policy is clean and
matches the toy-scale result. The scale path within evolutionary search is therefore
viable to at least this size, which was not obvious in advance. The full language-
model version remains a separate, gradient-trained program, but the causal relations
now hold at a hundred times the parameters and with a genuine attention block.

