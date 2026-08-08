# Constitutive GPT: a design sketch for a small model where identity is load-bearing

*Forward-looking design, 2026-08-02. Grounded in the CCA findings and the
Generic-Policy Escape principle. This is a sketch, not a build. Per the
consolidation, the private-metabolism forager experiment should validate the task
criterion before this is built, so the expensive substrate does not rediscover
the escape principle. No em-dashes.*

---

## 1. Goal and the three ingredients we know are required

A small transformer in which bearer identity is causally load-bearing for
behavior, so that self-coupled and wrong-bearer instances measurably diverge.
The CCA program established the three ingredients, and the third is the hard one:

1. **An incorporated private state, not a record.** The persistent per-instance
   state b must be part of the model's own dynamics (fast weights or a recurrent
   latent updated online), not a KV cache, a retrieval store, or a prompt
   summary. Prediction 4 showed that a record stays separable from the actor; the
   gate showed that only a load-bearing incorporated pathway makes identity
   testable.
2. **Online adaptation.** The environment must require within-episode adaptation,
   or the incorporated pathway sits idle (the static-world failure).
3. **No bearer-neutral escape.** The task must satisfy the obligate criterion
   from CONSOLIDATION_ESCAPE.md: every bearer-independent policy is nonviable for
   some instance, while a bearer-conditioned policy keeps every instance viable.
   Without this, a flexible model finds a generic policy and identity stays idle,
   exactly as in the forager.

## 2. Architecture

The model realizes the bearer loop directly:

    o_t = V(w_t, b_t)          interpret the outcome relative to the private state
    a_t = pi(h_t, b_t)         act, conditioned on the incorporated state
    b_{t+1} = U(b_t, o_t)      update the private state from the agent's own outcome

Concretely, on top of a small base transformer (a few layers, small width):

- **b_t, the bearer core.** A compact per-instance state carried across the
  episode. Two viable implementations: a low-rank fast-weight delta applied to
  selected projection matrices (as in the CCA agent), or a dedicated recurrent
  latent that FiLM-gates the transformer trunk. It must influence both the
  interpretation V and the policy pi, so identity enters twice, matching Chapter
  3's observation that bearer state affects both action and outcome constitution.
- **U, the update.** An outcome-gated update rule, either a fixed Hebbian or
  eligibility-trace rule (interpretable, as in the CCA agent) or a small learned
  update network trained by the outer loop. Crucially U writes to b, part of the
  actor, not to an external buffer.
- **What b is not.** Not the context window, not a retrieval index, not a text
  scratchpad. Those are records, and Prediction 4 predicts they remain separable
  from the actor. The whole point is an incorporated state.

The base weights are meta-trained (ES or a short inner-loop gradient method) so
that the update rule and the gating are useful; b is reset at the start of each
instance and written only by that instance's lived outcomes.

## 3. The task: an obligate private-code problem

A language or symbol-stream analogue of the private-metabolism task, built to
satisfy the obligate criterion.

- Each instance is assigned a private, unobservable code: a mapping from symbols
  or actions to outcomes that differs per instance, with no universal solution.
- The model must produce outputs continuously. A maintenance requirement (it must
  keep responding correctly at some rate) forecloses the "do nothing" escape.
- Wrong outputs are penalized in proportion to how wrong they are for this
  instance's code (the toxicity term), so volume does not compensate for
  incorrect composition.
- The code must be discovered through the instance's own outcome feedback within
  the episode, and it cannot be read from the input, so discovery must route
  through b.
- Because the codes differ across instances with no universal answer, every fixed
  policy fails for some instance while a code-conditioned policy succeeds for all.

The neutral dependent variable is externally comparable: accuracy or survival
time under a common threshold, even though the code producing it is
instance-specific.

## 4. The essential experiment and its mediator check

Unchanged from the forager, now in the model:

- **self-coupled:** b is updated by the instance's own outcomes.
- **wrong-bearer / yoked:** b is updated by an information-matched outcome stream
  from a different instance, same volume, same update rule, same environment.
- **Prediction:** on this obligate task, self should beat wrong-bearer, because
  wrong-bearer learns another instance's code and applies it, which is nonviable
  under this instance's private code.

**Mediator check first.** Before reading performance, verify that the model's
outputs discriminate by b: that the same input maps to different actions as a
function of the incorporated private state. If behavior does not depend on b,
another null is uninterpretable, exactly the ambiguity that limited the
obligate-poison forager run. Only after the mediator is confirmed does the
self-versus-wrong comparison mean anything.

## 5. Build order and honest scoping

1. **First, the forager metabolism experiment** (CONSOLIDATION_ESCAPE.md,
   Section 5). It is cheap, it validates that the obligate criterion actually
   induces the mediator, and it tells us whether the effect exists before we pay
   for a transformer. If the mediator does not arise even there, the task design
   is wrong and must be fixed before any model is built.
2. **Then a minimal Constitutive GPT** on the private-code task, with the same
   gate and mediator checks the forager pipeline already implements. Reuse the
   discipline: ablation gate (is the pathway load-bearing), falsifier (lambda = 0
   identity), dual criteria (bearer-neutral and bearer-relative), and the
   mediator check.
3. **Only then scale.** Chapter 9.9's warning applies: scaling before the causal
   relations are established produces an impressive system whose bearer boundary
   is uninterpretable.

This document is the target, not a promise that the effect will appear. The CCA
program's clearest lesson is that a flexible learner escapes bearer dependence
whenever the task lets it, so the task, not the model, is the hard part. The
model is worth building once a task with no escape is demonstrated to induce the
mediator.

## 6. Relation to the manuscripts

This is the concrete form of Bearer Theory Chapter 9.4 to 9.7 (what a
bearer-oriented architecture requires) and the natural successor to the Chapter
9.9 testbed, moved from a foraging agent to a sequence model, with the same four
loci and the same interventions. b_t is the incorporated bearer state the
manuscript distinguishes from external memory; V and pi both reading b_t is the
double entry of bearer state from Chapter 3; U writing to b_t is developmental
update from Chapter 4; and the self-versus-wrong comparison under an obligate task
is the sharpened Prediction 2.
