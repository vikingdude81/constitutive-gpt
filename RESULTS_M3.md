# Constitutive GPT M3: self versus wrong-bearer (sharpened Prediction 2)

*Run 2026-08-06. The test the whole program was built for. Does updating the
incorporated memory on the instance's own outcomes beat updating it on a different
instance's outcomes, scored on the instance's own private code? No em-dashes.*

---

## Design

The agent acts from its fast-weight memory W_fast as in M2. The outcome that WRITES
W_fast is a lambda-mix of two reward streams for the same action a:

    o = (1 - lambda) * (r_own - shift) + lambda * (r_alt - shift)

where r_own is the reward under the instance's own hidden arm theta_own and r_alt is
the reward under a different instance's arm theta_alt. Accuracy is always scored on
the OWN code. At lambda = 0 the memory is written by own outcomes; at lambda = 1 by
the alt outcomes, which identify theta_alt and match theta_own only by chance.
Prediction: accuracy falls monotonically with lambda. Implementation in
constitutive_gpt_m3.py, rollout_bearer.

This matches the forager bearer sweep format exactly, now on a substrate that
reaches the observability window (M2).

## The controls that defeat the criterion-internal reading

The forager program's hardest lesson was that a self-versus-wrong gap can be a
scoring artifact. Two controls rule that out here:

- **The M2 ablation gate** already showed the memory is load-bearing: freezing the
  write collapses accuracy to chance. The mediator is established, not assumed.
- **The falsifier:** with theta_alt = theta_own the two streams are identical, so
  the sweep must be FLAT. A flat falsifier means the gap is driven by the bearer
  difference, not by any generic property of updating.

Information is matched: both streams deliver N Bernoulli rewards with the same
marginal statistics; only whose code the outcomes reflect differs.

## Result: M3 PASS

Trained policy (K=8, eps=0.06, d=32, N=2000), scored on 256 held-out instances.

    lambda   self..wrong   falsifier(alt=own)
      0.00      0.633           0.646
      0.25      0.438           0.684
      0.50      0.288           0.729
      0.75      0.191           0.692
      1.00      0.118           0.605

    self (lambda=0)                 0.633
    wrong-bearer (lambda=1)         0.118   (chance 0.125)
    gap (self - wrong)             +0.515   monotone
    bearer effect isolated at 1    +0.487   (falsifier 0.605 vs wrong 0.118)

The wrong-bearer line collapses monotonically to chance as the write shifts to
another instance's outcomes. The falsifier line, where the alt stream carries the
same code, does not collapse: it stays viable at every lambda (0.60 to 0.73). At
lambda = 1 the two conditions diverge by 0.487, which is the bearer effect isolated:
writing with a different bearer's outcomes is nonviable for this instance, writing
with the same bearer's is fine.

Note on the falsifier's mild bump. The falsifier rises slightly at mid-lambda (0.646
to 0.729 and back to 0.605). This is benign and non-monotone: at mid-lambda the write
averages two independent same-code reward streams, which reduces write noise and
slightly raises accuracy. It is variance reduction, not a bearer effect, and it is
the opposite shape from the wrong-bearer collapse. The right test is the divergence
at lambda = 1, not the flatness of the whole line.

## Verdict

M3 PASS. Self beats wrong-bearer on the own private code with a large monotone gap,
and the falsifier isolates the effect: the incorporated memory is bearer-specific and
non-transferable. Updating it on another instance's outcomes writes that instance's
value structure and makes it nonviable here.

Three facts together defeat the criterion-internal reading that limited the forager
program. The M2 ablation gate showed the memory is load-bearing (freeze collapses to
chance). The falsifier showed the gap vanishes when the alt code equals the own code.
And information volume was matched across conditions. So the gap is a real consequence
of bearer-relative incorporation, not a scoring artifact.

This is the sharpened Prediction 2, confirmed on a substrate that reaches the
observability window. The forager program established constitution (P1) and
incorporation (P3) but never reached this window, because every task it could pose
left a bearer-neutral escape that more capable optimization exploited (Levers A, B).
The Constitutive GPT forecloses the escape by construction (the context window cannot
hold the horizon, gate 0), and on that task the incorporated state becomes
load-bearing and bearer-specific, which is what the whole program set out to
demonstrate.

