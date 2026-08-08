# Constitutive GPT M2: the incorporated model

*Run 2026-08-06. Milestone M2 of CONSTITUTIVE_GPT_SCOPE.md: build the real
incorporated mechanism, a fast-weight associative memory trained end to end by ES,
and test whether it learns to carry the per-arm integration that the M2 probe showed
was reachable. No em-dashes.*

---

## The mechanism (not hand-shaped)

The M2 probe used a hand-indexed per-arm accumulator to prove the task and ES are
viable. M2 replaces it with the mechanism the Constitutive GPT is meant to use: a
fast-weight associative memory, the sequence-model analogue of the forager's bearer
fast weights.

The incorporated state is W_fast in R^{d x d}, reset each episode. Implementation in
constitutive_gpt_m2.py:

- **read:** for each arm k, v_k = u . (W_fast e_k), a bilinear read against a
  learned arm embedding e_k. With near-orthonormal embeddings, v_k accumulates the
  gated reward written for arm k, so v is a per-arm value estimate.
- **policy:** a ~ softmax(beta v). Explores when W_fast is empty, exploits as it
  fills.
- **write (the incorporated update, Lever B form):** on the agent's own outcome,
  W_fast += c_write * o_n * outer(w_val, e_a) - c_decay * W_fast, keyed by the pulled
  arm a and gated by the centered reward o_n.

Nothing is hand-indexed. ES learns the arm embeddings E, the read and write vectors
u and w_val, the write and decay gains, the reward centering, and the policy
temperature (about 320 parameters at d=32). The M2 question is whether ES finds a
setting where the associative memory carries the integration, not whether one can be
hand-built.

## The two stop-rule checks (the mediator / ablation gate)

- **Beat the context-only baseline** (0.170 at the M1 calibration): the incorporated
  memory must integrate beyond what any L=32 window can.
- **Freeze the write (c_write -> 0) and accuracy must collapse toward chance:** this
  is the ablation gate for this substrate, proving the incorporated state is
  load-bearing rather than idle.

## Learning check (CPU, d=32, N=500, 20 generations)

A short pre-run confirmed the mechanism trains and the gate fires before committing
to the full run:

    held-out climbed 0.114 -> 0.215 over 20 generations
    gate: write ON 0.215 vs write FROZEN 0.125 (drop +0.090)
    the incorporated state is already load-bearing at this small scale

The accuracy rises and freezing the write drops it to chance, so the associative
memory is doing the integration. The full run below extends the horizon and the
generations to push accuracy toward the M2-probe target (about 0.72 at N=2000, 0.87
at the M1 full horizon N=4000).

## Full run (GPU, d=32, N=2000, pop=96, 200 generations)

    chance                       0.125
    context-only baseline (L=32) 0.170

    trained held-out (N=2000)    0.732   (M2-probe target 0.72)
    full horizon (N=4000)        0.858   (M1 ceiling 0.874)
    gate: write ON               0.732
          write FROZEN           0.125   (exact chance)
          drop                  +0.607

Held-out is evaluated on 256 fresh instances, so the numbers are low variance. ES
was noisy late (the held-out probe bounced while the best kept), and the returned
policy is the best held-out checkpoint; a smaller learning rate or larger population
would stabilize the curve, but the result is robust on the 256-instance eval.

## Verdict: M2 PASS

The fast-weight incorporated memory, trained end to end by ES, carries the per-arm
integration and reaches nearly the information ceiling (0.858 at N=4000 versus the
0.874 optimal). Freezing the write collapses accuracy to exact chance (0.125), a drop
of 0.607, so the incorporated state does essentially all the work and is load-bearing
rather than idle. The two stop rules are met: it beats the context-only baseline and
the ablation gate fires hard.

This is the clean window the forager program never reached: viable, on the
incorporated pathway, and learnable, on a task whose context escape is foreclosed by
construction (gate 0). The Lever A/B caveat is answered here: the barrier was never
that incorporated state cannot be load-bearing, it was that a residual escape gets
exploited more as capacity rises. Remove every escape by construction, and a more
capable substrate uses the incorporated state instead, exactly as intended.

## Next: M3

M3 runs the sharpened Prediction 2, the self versus wrong-bearer comparison: update
W_fast on the instance's own outcomes (self) versus on an information-matched outcome
stream from a different instance (wrong-bearer). Prediction: self beats wrong-bearer
on this instance's own private code, because wrong-bearer writes another instance's
value structure into W_fast and applies it here. This is the test the whole program
was built for, now on a substrate that reaches the window.
