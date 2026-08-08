# Constitutive GPT: build scope

*Scoping document, 2026-08-05. Turns CONSTITUTIVE_GPT_DESIGN.md into an actionable
plan, updated with the Lever A and B result. The forager program is closed by
tuning; this is the substrate where the obligate construction is achievable. This
document commits to concrete decisions, milestones, controls, and go/no-go gates.
No em-dashes.*

---

## 0. What changed since the design sketch

The sketch predates Levers A and B. Their result raises the bar and sharpens the
design.

- **The Generic-Policy Escape Principle is meta-level.** A more capable optimizer
  does not discover its bearer identity; it exploits the bearer-neutral route more
  efficiently. Softening viability re-created a quantity escape (Lever A);
  strengthening the learner exploited it better (Lever B).
- **A transformer is a much more capable optimizer than the fast-weight forager.**
  So it will find any residual escape harder, not softer. The design cannot rely on
  the model failing to notice an escape.
- **The context window is the most powerful escape of all.** In-context attention
  over past (input, outcome) pairs is a retrievable record. If the code the model
  needs can be inferred by attending over the context, attention does the inference
  and the incorporated state b is bypassed. This is Prediction 4 (record versus
  incorporated bearer) with the sharpest possible edge.

The single most important decision in this build is therefore not the architecture
of b. It is the foreclosure of the in-context escape. Get that wrong and the model
will read its identity off the context and b will sit idle, exactly as the forager
read it off observable state.

## 1. The central design problem: foreclose the in-context escape

The task must require integrating outcome evidence over a horizon that EXCEEDS what
the context window can hold, so the accumulated evidence has nowhere to live except
b. Two mechanisms, used together:

1. **Horizon greater than context.** The private code requires aggregating evidence
   over N outcomes to infer, where N is larger than the context length L. No length-L
   window contains enough to solve it; only a running incorporated summary does.
2. **Context clearing (or a short rolling window) while b persists.** Periodically
   clear the KV cache and the visible history, but do not reset b. The only carrier
   across a clear is the incorporated state. This makes b load-bearing by
   construction and gives a clean ablation: freeze b's update and the model cannot
   survive a clear.

The go/no-go for the whole build is a context-only baseline (a model with the same
capacity but no persistent b, or b frozen) that MUST FAIL the task. If it passes,
the code leaked into context and the task is not obligate. Redesign before spending
on the incorporated variant.

## 2. Architecture (decisions, not options)

On a small base transformer (target: 4 layers, width 128 to 256, small symbol
vocabulary):

- **b, the bearer core: low-rank fast-weight deltas on selected projection
  matrices** (the value and output projections of one or two layers), exactly the
  CCA mechanism lifted into the transformer. Chosen over a recurrent FiLM latent
  because it reuses the entire CCA measurement pipeline (the ablation gate is
  literally freeze the delta), and because the fast-weight delta is unambiguously
  incorporated, not a token in context.
- **Double entry.** b must modify both the interpretation V (how an outcome is
  scored into the update) and the policy pi (the forward pass). Identity enters
  twice, matching Chapter 3.
- **U, the update: start with the 4-term learned rule from Lever B**, ES-optimized.
  It is already built and it is the most expressive incorporated update we have. The
  Lever B lesson is not that a learned rule is useless; it is that a learned rule
  with a residual escape learns the escape. With the escape foreclosed by the
  context design, the learned rule is the right tool.
- **What b is not:** not the KV cache, not a retrieval index, not a scratchpad in
  tokens. Those are records.

## 3. The task: obligate private-code, horizon greater than context

A symbol-stream task satisfying the obligate criterion AND the horizon condition.

- Each instance draws a private, unobservable code: a mapping from context-symbols
  to correct-actions that differs per instance, drawn so no universal policy is
  viable for all instances.
- The model emits an action every step (maintenance requirement forecloses the
  do-nothing escape).
- Reward is proportional to correctness under this instance's code, and wrong
  actions are penalized in proportion to how wrong they are (forecloses the volume
  and quantity escape that beat us in Lever A).
- The code can be inferred only by aggregating outcome feedback over a horizon N
  greater than the context length L, and the raw (symbol, outcome) history is not
  retained in context beyond L (rolling window or periodic clear). This forecloses
  the in-context retrieval escape, the one a transformer would otherwise take.
- Neutral dependent variable: accuracy or time-in-band under a common threshold,
  externally comparable across instances though the code is private.

## 4. Controls and go/no-go gates (reuse the CCA discipline)

Every check the forager pipeline already implements, plus one new gate specific to
the transformer.

- **Gate 0 (new, decisive): the context-only baseline must fail.** A model with b
  frozen (or absent) must not solve the task. This proves the escape is foreclosed.
  This gate runs FIRST, before any incorporated training is interpreted.
- **Mediator check:** the same input maps to different actions as a function of b.
  If behavior does not depend on b, a null is uninterpretable. Runs before the
  self-versus-wrong comparison.
- **Ablation gate:** freeze the b update and viability must drop (the pathway is
  load-bearing). On the transformer this is freeze-the-delta, and a clear-survival
  version: does frozen-b fail after a context clear.
- **Falsifier:** at lambda = 0 the self and wrong-bearer conditions are identical by
  construction; dP(0) must be within noise.
- **The essential comparison:** self-coupled (b updated by own outcomes) versus
  wrong-bearer / yoked (b updated by an information-matched outcome stream from a
  different instance). Prediction: self beats wrong on this instance's private code.

The window conditions carry over: E = 0 (no escape, enforced by gate 0), gate passes
(incorporated pathway load-bearing), L = 1 (learnable). The transformer is the bet
that attention through fast-weight-modified projections can learn the blind
inference that the pure fast-weight forager could not.

## 5. Build milestones (phased, each with a stop rule)

1. **M1, task harness + gate 0.** Build the private-code environment and the
   context-only baseline. Stop rule: gate 0 must fail (baseline cannot solve it). If
   it passes, the task leaks; fix before proceeding. Cheapest and most important
   milestone.
2. **M2, minimal incorporated model, mediator.** Add b (fast-weight delta) and the
   Lever B update rule; ES-train. Stop rule: mediator present (output depends on b)
   AND the model beats the context-only baseline. If the mediator is absent, the
   incorporated pathway is idle; diagnose before the comparison.
3. **M3, the essential comparison.** Self versus wrong-bearer with the falsifier and
   the ablation gate. This is the actual test of sharpened Prediction 2.
4. **M4, scale only if M3 is positive.** Chapter 9.9's warning: scaling before the
   causal relations are established yields an impressive system with an
   uninterpretable bearer boundary.

## 6. Compute and effort estimate

- M1: about a day of build, cheap compute (baseline is small).
- M2: a few days including ES tuning; per-run compute comparable to the forager
  scaled by sequence length and model size. On the 5090, plan on runs of a few hours
  each, backgrounded.
- M3: reuses M2 infrastructure; the comparison and controls add seeds, not new code.
- The honest variable is M2. Meta-training a transformer with an online fast-weight
  update by ES is heavier than the forager, and ES may not scale; a short inner-loop
  gradient method is the fallback if ES stalls.

## 7. Risks and the open decisions to settle before M2

- **Risk: the horizon-versus-context foreclosure is hard to calibrate.** Too short a
  horizon and attention solves it (gate 0 passes, no good); too long and the model
  cannot learn at all (the difficulty valley returns). This is the same valley, and
  M1 is where we find the calibration.
- **Risk: ES does not scale to the transformer.** Fallback to truncated inner-loop
  gradients through the fast-weight update.
- **Open decision, meta-training:** ES (reuses the whole pipeline, proven, may not
  scale) versus inner-loop gradient (scales better, new code, backprop through the
  update). Recommendation: ES for M1 and M2, switch only if it stalls.
- **Open decision, scope of the first build:** minimal (single private code per
  instance, one fast-weight layer) versus richer (structured codes, multiple
  layers). Recommendation: minimal, because the point of M1 to M3 is the causal
  test, not capability.

## 8. Relation to the manuscript

This is the concrete form of Bearer Theory Chapters 9.4 to 9.7 and the successor to
the 9.9 testbed, moved from a forager to a sequence model with the same four loci.
The new contribution the Lever results add: the design law that the task, not the
learner, must foreclose every bearer-neutral route, and that for a sequence model
the primary route to foreclose is the context window itself. That is a
manuscript-level claim about why incorporated state is necessary and not reducible
to context, which is the empirical core of the record-versus-bearer distinction.
