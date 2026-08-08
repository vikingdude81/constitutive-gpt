# Constitutive GPT M1: obligate private-code task + gate 0

*Run 2026-08-06. Milestone M1 of CONSTITUTIVE_GPT_SCOPE.md: build the obligate
task and prove it is genuinely obligate before any incorporated model is built. No
em-dashes.*

---

## The task

A stochastic bandit-identification problem, the sequence-model analogue of the
private-metabolism forager (constitutive_gpt_m1.py, BanditIDEnv):

- Each instance draws a hidden correct arm theta_i in {0..K-1}, its private code,
  uniform and unobservable.
- Each step the agent pulls an arm and receives a noisy reward. The correct arm
  pays Bernoulli(0.5 + eps); every other arm pays Bernoulli(0.5). The gap eps is
  small, so one pull is weakly informative and identifying theta_i requires
  integrating outcomes over many pulls.
- No universal policy beats chance (each instance has a different theta_i), so the
  task is obligate: the only way above chance is to infer theta_i from the agent's
  own outcome stream.

This is purely outcome-driven. The bearer state b is updated by the agent's own
action-outcomes, and the self-versus-wrong-bearer comparison in M3 will feed b
another instance's reward stream, which identifies the wrong arm.

## Gate 0: the task must be obligate, not leaky

Gate 0 is the decisive M1 test. A context-only learner, one that can use only the
last L pulls, must FAIL (stay near chance), while a full-horizon integrator (all N
pulls) succeeds. If a short window suffices, the code leaks into context, attention
would do the inference, and the incorporated state b would sit idle. The bound is
computed against the Bayes-optimal decoder given n uniformly-explored pulls, which
upper-bounds any context-only model including a transformer attending over its
context. No model is trained: the gate is an information bound, the cheapest and
strongest way to establish it.

## Result: gate 0 PASS

Locked calibration: K=8, eps=0.06, L=32, N=4000 (chance = 0.125).

    pulls   accuracy
        8     0.144
       16     0.158
       32     0.170   <- context window L: near chance
      128     0.220
      256     0.280
      512     0.378
     1000     0.500
     4000     0.874   <- episode horizon N: solvable

    context-only (L=32) accuracy = 0.170
    full-horizon (N=4000) accuracy = 0.874

Context-only sits at 0.170, just above the 0.125 chance floor, while the full
horizon reaches 0.874, a five-fold separation. The inference horizon exceeds the
context by construction, so the incorporated state is load-bearing: any model that
solves this task must carry an integrated summary of its own outcomes across the
episode, which the L=32 context cannot hold.

Calibration notes. K=4 has a floor: even 16 pulls over 4 arms leave a small edge
(context-only ~0.30 against chance 0.25), too leaky. Raising to K=8 lowers the
chance floor and widens the transition, so the L=32 window is genuinely near chance.
The N over L ratio must straddle the statistical transition, which spans about two
orders of magnitude in pulls; K=8, L=32, N=4000 places L below the transition and N
above it.

## What M1 establishes and what is next

The task is obligate and the escape most dangerous to a sequence model, the context
window, is foreclosed by construction. This is the precondition the whole build
needed and the one the forager program lacked at threshold.

M2 builds the minimal incorporated model on this task: b as a low-rank fast-weight
delta, the Lever B learned update rule, ES-trained. Its two stop-rule checks are
that it beats the context-only baseline (0.170) and that the mediator is present
(output depends on b). Only then does M3 run the self-versus-wrong-bearer
comparison, the sharpened Prediction 2.

A caveat carried forward from Levers A and B: gate 0 forecloses the context escape,
but a trained model must still LEARN the blind integration. That is the difficulty
valley in this substrate, and M2 is where we find whether attention through
fast-weight-modified projections can climb it where the pure fast-weight forager
could not.
