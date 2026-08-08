# Lever B: a learned inner-loop update rule

*Run 2026-08-05. Tests whether replacing the fixed gated-Hebbian plasticity rule
with a learned multi-term update lets the incorporated pathway acquire strong blind
selectivity, so that viability and bearer-selectivity coincide (the window).
Framework: BEARER_OBSERVABILITY.md section 6, lever 3. No em-dashes.*

---

## Why this lever

Lever A localized the binding constraint to learnability: at the gain that reaches
viability (GAIN=5), the fixed-Hebbian agent stayed near-uniform (mediator 0.542,
absent), reaching the band by a quantity route rather than by selecting its own
demand types. The update rule cannot acquire strong blind selectivity. Lever B
changes the rule itself.

## The change

The plasticity controller previously output two scalars per fast-weight layer, a
learning rate eta and a decay delta, over a fixed update:

    dP = eta * o_n * elig - delta * P            (gated Hebbian)

Lever B expands the controller to four coefficients per layer that mix a basis of
plasticity terms, all from tensors already on hand:

    dP = c_elig * o_n * elig                      eligibility-Hebbian (slow credit)
       + c_inst * o_n * outer(y, x)               instantaneous-Hebbian (fast credit)
       - c_decay * P                              homeostatic decay
       - c_omod * o_n * P                         outcome-modulated forgetting (Oja-like)

The coefficients are produced by the same controller network (now with a wider
output head) conditioned on the bearer state b, the normalized outcome, eps,
degradation, and the eligibility and fast-weight norms. ES co-optimizes them with
the rest of the agent. The outcome o_n carries sign, so anti-Hebbian responses to
bad outcomes are representable even though the coefficients are non-negative. Flag:
`Config.learned_update`, CLI `--learned-update`. Implementation in
cca_milestone1.PlasticityController and gpu_batched.BatchedAgent.

## The A/B

The comparison is clean because the fixed-rule control at the same operating point
is already run (RESULTS_LEVER_A.md, GAIN=5):

    control (fixed Hebbian, GAIN=5):   mediator 0.542 ABSENT, gate PASS, advantage +0.027
    lever B (learned rule,  GAIN=5):   [pending]

Question: does the learned rule raise the mediator clearly above uniform (real
own-demand selectivity) while staying viable, and does self then beat wrong in the
bearer sweep? That would be the first time the window (viable AND selective AND on
the incorporated pathway) is occupied in this substrate.

Setup: curriculum K=2 (blindness ramp 0 to 1, eval fully blind), pop 768, 100
generations, const_sigma 1.5, gain-scale 5.0.

## Result

    advantage       +0.039   (best of any run, still below the +0.05 trained bar)
    mediator align   0.507   (uniform 0.500, ceiling 0.761)  ABSENT
    gate ON-OFF     +0.028   (FAIL, borderline)
    bearer sweep    no effect above noise, large CIs

Against the fixed-rule control at the same operating point:

    control (fixed Hebbian, GAIN=5):   mediator 0.542 ABSENT, advantage +0.027
    lever B (learned rule,  GAIN=5):   mediator 0.507 ABSENT, advantage +0.039

The learned rule raised viability to the highest of any run and lowered selectivity
to essentially uniform. More expressive plasticity did not buy bearer-selectivity.
It bought a better bearer-neutral escape.

## Verdict: the escape principle operates at the meta level

This is the informative result. Lever A softened the viability threshold and
re-created a near-uniform high-volume quantity route. Lever B gave the learner a
richer update rule, and the learner used the extra capacity to exploit that route
more efficiently (higher advantage) while abandoning selectivity (lower mediator).
The two levers fail for complementary reasons and together close the forager
program:

    soften viability (A)   -> re-admits a near-uniform quantity escape
    strengthen learner (B) -> exploits the escape better, not the bearer

So the Generic-Policy Escape Principle is not an artifact of a weak learner. A more
capable optimizer does not discover its bearer identity; it finds the bearer-neutral
route more effectively. The barrier to observing a bearer effect is structural:
whenever any bearer-neutral path to viability survives at the operating threshold,
increasing learner capacity increases escape, not incorporation.

## Consequence for the design

The window cannot be reached in the ES-plus-fast-weight forager by tuning the
threshold or the update rule, because both moves are absorbed by the escape. The
requirement is sharper than "make it learnable": the task must admit NO
bearer-neutral route to viability at the threshold at all, and any residual escape
will be exploited more as the learner improves. The homeostatic-band plus gain
structure of this forager cannot provide that without tipping into collapse (too
hard, no viable policy) or escape (too soft, a quantity route appears).

This is the technical reason to move to the Constitutive GPT with the escape
foreclosed by construction: an incorporated private state that gates value and
policy, on a task whose required information cannot be held in context or read from
observation, so that no in-context or quantity route substitutes for the bearer.
The lesson from Levers A and B travels with it: raw capacity alone will find any
escape that exists, so the task, not the learner, must close the door.

