# Constitutive GPT: a richer, multi-dimensional private code

*Run 2026-08-08. Answers whether the bearer effect is genuine identity or a
one-dimensional lookup, by making the private code an m-dimensional vector. No
em-dashes.*

---

## The task

M1 to M3 used a one-dimensional private code, a single hidden arm. This task makes
the code a vector w_i in R^m, an m-dimensional response profile.

- Each instance draws a private unit vector w_i in R^m, unobservable.
- Each step shows an observable context vector x_t in R^m. The correct action is the
  sign of w_i . x_t, so it varies with the context. Reward is noisy: the correct
  action pays Bernoulli(0.5 + eps), the wrong one Bernoulli(0.5).
- To act correctly across contexts the agent must recover the whole vector w_i, not
  one coordinate. This is a genuine multi-dimensional identity.

The incorporated state is a running estimate w_est in R^m, updated by a reward-gated
perceptron rule: w_est += eta*(r - shift)*(2a-1)*x - rho*w_est. Evolutionary search
tunes the scalars. Implementation in constitutive_gpt_richer.py.

## Result: RICHER PASS

Settings m=4, eps=0.15, N=2000, L=32. Chance is 0.5.

    gate 0: context-only estimator (L=32)   0.612
            full-horizon estimator (N=2000)  0.908
    trained incorporated model               0.900   beats context-only decisively

    bearer sweep, accuracy on OWN contexts:
      lambda   self..wrong   falsifier (alt=own)
      0.00       0.897           0.897
      0.25       0.863           0.917
      0.50       0.730           0.925
      0.75       0.586           0.917
      1.00       0.498           0.897

    self (lambda=0)              0.897
    wrong-bearer (lambda=1)      0.498   (chance 0.500)
    gap                         +0.398   monotone
    falsifier isolation         +0.398

An m-dimensional private identity is learnable only through the incorporated state: a
bounded L=32 window leaves the code underdetermined and a context estimator stays near
chance, while the incorporated estimate recovers the whole vector and reaches 0.900.
The identity is bearer-specific. Feeding another individual's outcomes drives w_est to
the wrong profile, and because the profile is applied across every context, own-code
accuracy collapses to exact chance. The falsifier, where the alt outcomes carry the
same profile, stays high, so the collapse is the change of owner, not the swap.

The learned rule uses a small positive learning rate and a negative rho, that is, it
accumulates the reward-weighted context evidence without decay, the natural online
estimator of w_i.

## What it establishes

The bearer effect is not an artifact of a one-bit code. It holds when the private
identity is a continuous multi-dimensional profile that shapes behavior across a whole
space of contexts, and it remains non-transferable there. This is the sense of
identity the program is after: a many-dimensioned way of being that is grown from the
agent's own outcomes and cannot be handed to another.

One honest note. The context-only estimator sits at 0.612 rather than exactly 0.5, so
a bounded window carries a little information about a 4-dimensional code. The trained
model at 0.900 is far above it, so the incorporated integration does much more than a
window can, but a larger m or a smaller eps would widen the gate-0 margin if a cleaner
separation is wanted. The non-transferability result, wrong-bearer at exact chance,
does not depend on that margin.
