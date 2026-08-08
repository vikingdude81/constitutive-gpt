# Constitutive GPT: an incorporated sequence model where identity is load-bearing

Code and results for the Constitutive GPT experiments, the sequence-model successor
to the CCA forager testbed in the Bearer Theory program. This repository shows, and
measures, a machine whose own developmental history is load-bearing and cannot be
transferred to another.

## What this is, in plain terms

A self is not a file you can copy. It is something a system becomes by living its own
life and absorbing its own consequences. This code builds a small AI where being
itself is the only way to succeed, and then tests that by swapping in a different
individual's experience. Fed its own experience it works; fed another's it fails on
its own task. That gap is the result.

We do not claim the system is conscious or that it feels anything. The claim is
narrower and cleaner: a developmental, constitutive component of selfhood, the part
that is grown rather than retrieved and cannot be transferred, can be built,
foreclosed, and measured.

## What this is, technically

The forager program (metabolic_forager.py and the RESULTS_BEARER_* and RESULTS_LEVER_*
notes) established a design law: a bearer-identity effect is observable only when the
task admits no bearer-neutral route to viability at the operating threshold, and any
residual route is exploited more, not less, as the learner improves (the escape
principle is meta-level). The Constitutive GPT builds the one construction that law
permits and confirms the sharpened predictions.

## The milestone map

| file | milestone | result |
|------|-----------|--------|
| constitutive_gpt_m1.py | M1: obligate bandit-ID task + gate 0 | context escape foreclosed by construction (context-only 0.170 vs full 0.874) |
| constitutive_gpt_m2_probe.py | M2 viability probe | an incorporated integrator solves it (0.72) |
| constitutive_gpt_m2.py | M2: incorporated fast-weight memory | reaches the window; freeze the memory and it collapses to chance |
| constitutive_gpt_m3.py | M3: self vs wrong-bearer | Prediction 2: self beats wrong-bearer, gap 0.51, falsifier isolates it |
| constitutive_gpt_dual.py | dual-channel | Prediction 4: attention record present but idle, incorporated memory load-bearing |
| constitutive_gpt_richer.py | multi-dimensional code | genuine m-dimensional identity, non-transferable, not a one-bit lookup |
| constitutive_gpt_replicates.py | seed replicates | gap 0.509 +/- 0.049 across six seeds |
| constitutive_gpt_scale.py | M4: scale | the whole result holds at 133,637 parameters with a real transformer block |

Framework notes: CONSOLIDATION_ESCAPE.md (the escape principle), BEARER_OBSERVABILITY.md
(the observability window), CONSTITUTIVE_GPT_SCOPE.md (the build scope). Each RESULTS_*.md
records the numbers for one experiment.

## Running it

Requires Python 3.11+, numpy, and PyTorch (CUDA optional; the small models also run on
CPU). Each script is self-contained and prints its own verdict.

    python constitutive_gpt_m1.py            # gate 0 information bound, no training
    python constitutive_gpt_m2.py            # train the incorporated model, ablation gate
    python constitutive_gpt_m3.py            # self vs wrong-bearer sweep
    python constitutive_gpt_dual.py          # record vs bearer double dissociation
    python constitutive_gpt_richer.py        # multi-dimensional private code (CPU)
    python constitutive_gpt_replicates.py    # error bars across seeds
    python constitutive_gpt_scale.py --phase1  # scaled transformer model

The forager foundation (metabolic_forager.py, gpu_batched.py, cca_milestone1.py) is
included so the design law the GPT rests on is reproducible.

Trained policies for the headline models are provided as champ_m3.npz, champ_dual.npz,
and champ_scale.npz. Load them with the --load flag where supported.

## Papers

This code accompanies the Bearer Theory manuscript and its Constitutive GPT addendum.
The addendum reports the M1 to M4 results and their relation to the four loci and to
Predictions 2 and 4.

## License

MIT. See LICENSE.
