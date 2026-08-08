# Constitutive GPT dual-channel: the record present and idle (Prediction 4)

*Run 2026-08-08. Adds a real context/attention channel alongside the incorporated
memory and lets them compete, to show inside one model that the record is idle and
the incorporated state carries the task. The sharp form of Prediction 4. No
em-dashes.*

---

## Why this model

M2 and M3 used a pure incorporated-memory cell, which has no context at all. The
claim that incorporation beats a retrievable record was therefore carried only by the
gate-0 information bound, not shown inside a single model that has both options. This
model gives the model both channels and lets evolutionary search lean on either.

- **Context channel, the record.** A single-head attention pools the last L action
  and reward tokens into a summary, a bounded transformer context. By gate 0 an
  L-window cannot solve the task, so this channel is a present but insufficient
  record.
- **Incorporated channel, the bearer.** The fast-weight associative memory of M2,
  integrating over the full horizon.

Per-arm logits are logit_k equals beta times the memory read plus gamma times the
context read. Both weights are learned. Implementation in constitutive_gpt_dual.py.

## The Prediction 4 test: a double dissociation

- **Ablate the context (gamma to zero).** If the record is idle, accuracy barely
  moves.
- **Freeze the memory (write to zero).** If incorporation is load-bearing, accuracy
  collapses.

If both hold, the record is present in the model yet idle, and the incorporated state
carries the task and the identity. That is the record-versus-bearer distinction shown
inside one sequence model, not argued from a bound.

## Result: DUAL PASS

Trained policy (K=8, eps=0.06, d=32, N=2000, L=32), scored on 256 held-out instances.

    full (both channels)          0.732
    ablate context (memory only)  0.703   record removed, drop only 0.029
    freeze memory (context only)  0.160   bearer removed, drop 0.572
    learned weights               beta(memory) 1.72   gamma(context) 0.44

The double dissociation is clean. Removing the retrievable record barely moves
accuracy, from 0.732 to 0.703, so the context is idle. Freezing the incorporated
memory collapses accuracy to 0.160, a drop of 0.572. And 0.160 is almost exactly the
gate-0 context-only bound of 0.170: the context channel on its own reaches precisely
its information-theoretic ceiling and no more, which confirms it is a genuine but
insufficient record. The learned context weight gamma, 0.44, is small next to the
memory weight beta, 1.72, corroborating that the model leans on the incorporated
state.

## Verdict

DUAL PASS. The model has a real attention channel over a bounded context, it could
lean on that record, and it does not. The incorporated fast-weight memory carries the
task. Removing the record is nearly free; freezing the incorporated state is
catastrophic and drops the model to exactly what a bounded record can do alone.

This is Prediction 4 shown inside one sequence model rather than argued from a bound:
the record is present and idle, the incorporated bearer is load-bearing. It also
settles the naming. The Constitutive GPT is a real sequence model with attention in
which the incorporated state, not the context, carries the task and the identity.
Combined with M3, where that incorporated state is bearer-specific and
non-transferable, the model now demonstrates both halves of the record-versus-bearer
distinction: a record cannot substitute for the bearer, and the bearer cannot be
swapped for another.

