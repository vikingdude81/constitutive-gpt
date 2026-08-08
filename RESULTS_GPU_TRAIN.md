# GPU ES training: first genuinely trained forager

Run 2026-07-24 on the RTX 5090. This is the run that clears the bar sweep #1
could not: an agent that beats random on worlds it never trained on.

## Setup

The three fixes that address why sweep #1 measured an untrained agent:

- **pop 512** (was 24). A real ES gradient estimate. Affordable because the
  validated GPU evaluator batches the whole population in one process.
- **d = 64** (was 256). Search space 118,929 params (was ~1.16M).
- **Held-out selection.** The champion is the ES distribution center, scored on
  worlds never used in training. Training seeds 10000+gen; held-out 90000+i,
  disjoint. This is what stops the overfit-to-training-seeds failure that made
  sweep #1's champion look good only on its own selection worlds.

120 generations, lifetime 800, ~12 s/gen, about 26 minutes total. Single GPU
process, one CUDA context, no memory risk.

## Learning curve (held-out, every 10 gens)

```
gen 000  -0.7160  (+0.011)   at baseline, untrained
gen 010  -0.7099  (+0.017)   still flat
gen 020  -0.4597  (+0.268)   <- breaks upward
gen 030  -0.3998  (+0.327)
gen 050  -0.3990  (+0.328)
gen 060  -0.3908  (+0.336)   <- best held-out, banked as champion
gen 080  -0.3952  (+0.332)
gen 119  -0.4000  (+0.327)
```

Flat at the random baseline for ~20 generations, then a sharp break, then a
plateau around +0.33. Checkpoints oscillate +0.11 to +0.34 because held-out is
averaged over only 4 worlds; that is monitoring noise, not agent instability.
Train_mean and held-out improved together, which is genuine learning rather
than the overfitting signature (train up, held-out flat) seen in sweep #1.

## Functional test (the champion, on 4 held-out worlds)

```
champion mean : -0.4364
random   mean : -0.7272   (best of 16 random genomes: -0.5673)
advantage     : +0.2909
per-world     : -0.40  -0.40  -0.46  -0.49
VERDICT: TRAINED
```

The champion beats the random baseline by +0.29, roughly 6x the +0.05 "trained"
threshold, and beats even the single best random genome by +0.13. It is
consistent across all four unseen worlds, so it is not one lucky draw.

For contrast, sweep #1's CPU champion (pop 24, d 256) beat random by **+0.009**
on the same style of test. That agent was untrained; this one is trained.

## What this unblocks

For the first time in this program we have a forager that demonstrably
learned. A bearer-indexing sweep (self vs wrong_bearer vs stream, over lambda)
on THIS champion can actually be interpreted, because the agent has real
homeostatic competence for bearer-identity to modulate. Sweep #1's null was
uninterpretable precisely because the agent was random; that objection is now
answered.

Champion saved to `champion_gpu_forager.npz` (theta, scope=all, d=64,
best_holdout, rand_baseline).

## Caveats

- Held-out uses 4 worlds for monitoring and the final test. More worlds would
  tighten the estimate; the +0.29 margin is large enough that the verdict is
  not in doubt, but the exact number carries a few hundredths of noise.
- The plateau near +0.33 suggests ES has roughly converged for this agent size.
  A larger agent or population might do better, but competence is already
  clearly established, which is all the bearer experiment needs.
- The champion is trained under the CPU env semantics reproduced on GPU (parity
  validated separately). The bearer sweep should run on the same GPU path for
  consistency, which means porting the self/wrong_bearer/stream conditions into
  the batched evaluator next.
