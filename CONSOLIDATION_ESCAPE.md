# Generic-Policy Escape in Bearer-Coupled Agents

*Consolidation of the CCA bearer-identity experiments, 2026-08-02. Frames six
task designs under a single principle, states the epistemics carefully, and
specifies the decisive next experiment. Refined with external review. No
em-dashes.*

---

## 1. The result in one sentence

Across six progressively constrained task designs, bearer-relative constitution
and historical incorporation stayed measurable, but self-coupling produced no
reliable advantage on bearer-neutral performance, because in every design the
task retained a policy that preserved viability without conditioning behavior on
the agent's private bearer state.

## 2. What is and is not established

**Established:**
- Bearer-relative constitution is real: the same event is a different effective
  outcome for different bodies (corr(V_self, V_wrong) = 0.79, difference 36% of
  signal).
- Historical incorporation is real: a specific outcome leaves a residue in the
  continuant that survives after the cue (delayed ||db|| = 5.76).
- The outcome-driven update pathway is load-bearing under drift (the ablation
  gate passes in every trained design).
- Yet self-coupling gives no bearer-neutral performance gain in any of the six
  designs.

**Not established, and a correction to an earlier overreach:**
- These are not falsifications of obligate bearer dependence. The intended causal
  chain was: private constitution -> bearer-selective behavior -> neutral
  advantage. In each design the middle term did not arise. Neither self nor
  wrong-bearer agents developed bearer-selective behavior (diet quality stayed
  near random for both). Because the mediator never appeared, the experiments
  cannot say whether bearer-selective behavior would have produced the advantage.
  The honest description is a failed induction of obligate dependence.
- The philosophical reading, that identity is what a system is rather than an
  edge it has, is retained as interpretation, not presented as the empirical
  conclusion. The data establish the narrower and still important claim that
  bearer identity can be causally real without producing a gain under the neutral
  criteria tested.

## 3. The Generic-Policy Escape Principle

The six designs are unified by one principle, which converts a sequence of nulls
into a structural result.

**Principle.** When a task contains a bearer-independent policy that preserves
viability, a sufficiently flexible agent can satisfy the task without making its
private constitution behaviorally load-bearing.

Let J_i(pi) be the neutral viability score of policy pi for bearer i, and tau the
minimum viable score. A generic-policy escape exists when

    there is a bearer-independent policy pi_g with J_i(pi_g) >= tau for all i.

When such a pi_g exists, the agent does not need a policy conditioned on its own
bearer state. Timing, quantity regulation, reduced exposure, or another generic
strategy solves the task. Every design we ran contained such a pi_g:

```
design            the generic escape pi_g
static world      no adaptation needed (fixed policy)
drift world       regulate by timing
blind agent       regulate by timing through the pathway
dose curve        offset poison with quantity
obligate poison   keep integrity adequate without full selectivity
```

## 4. The design criterion for a genuinely obligate task

A task makes bearer identity instrumentally necessary when no bearer-independent
policy is viable for all bearers, while a bearer-conditioned policy is:

    for every bearer-independent pi_g, there exists a bearer i with J_i(pi_g) < tau,
    and there exists a bearer-conditioned pi_b(a | s, b_i) with J_i(pi_b) >= tau for all i.

In words: every bearer-neutral policy must fail for at least one bearer, while a
bearer-correct policy must let each bearer remain viable. Only under this
condition can the self-versus-yoked comparison test whether bearer identity
becomes load-bearing, because only then is there no escape.

## 5. The decisive next experiment: the private-metabolism task

Replace poisoned-food preference with a private metabolism the agent must match.
Each agent sees the same externally visible resources but transforms a consumed
nutrient mixture x_t through its own metabolism:

    delta_e_t = M_i x_t - c_t - d_i(x_t)

where M_i is bearer i's private metabolism, c_t is an unavoidable maintenance
cost, d_i(x_t) is toxicity from an incorrect mixture for bearer i, and e_t is the
neutral viability state (energy, or time within a common homeostatic band).

Four constraints enforce the obligate criterion of Section 4:

1. Under-eating starves. Continuous maintenance cost c_t makes "eat less"
   nonviable.
2. Over-eating poisons. Toxicity d_i(x_t) of the wrong composition means quantity
   cannot compensate.
3. Timing cannot solve it. Maintenance demand is continuous or randomized, so
   there is no safe window.
4. No universal mixture exists. Different M_i require genuinely different
   balances, so no single bearer-independent mixture keeps every bearer viable.

The neutral dependent variable (survival time, accumulated energy deficit, or
time in a common band) is externally comparable across agents even though the
metabolism producing it is bearer-specific.

The comparison is unchanged: self-coupled versus wrong-bearer or yoked, matched
in information volume, update rule, and environment, differing only in whose
consequences perform the update.

**Mediator manipulation-check first.** Before reading performance, verify that
the learned policy actually discriminates among nutrient mixtures as a function
of the agent's own metabolism. Without this check, another null would again be
ambiguous between "identity gives no advantage" and "the agent never used its
identity." The check is the fix for exactly the ambiguity that limited the
obligate-poison run.

## 6. What this contributes to Bearer Theory

This clarifies the manuscript's Prediction 2 and its criterion-internal
qualification with a mechanism. The clarification is:

    Causal bearer identity does not become adaptive leverage merely because it
    exists. The environment must make the identity necessary.

That is a structural claim with a testable criterion, not an intuition. It also
gives Section 8.3 a precise necessary condition (the obligate criterion of
Section 4) and a concrete protocol (the metabolism task with a mediator check).
It reframes the six nulls as a positive finding about task structure rather than
a failure to obtain a positive effect.

## 7. The private-metabolism task, run: escape seven and the difficulty valley

The private-metabolism task was built and run in two forms, and both taught
something.

**Form A (composition match).** Viability required the diet composition to match
a private demand vector r_i over K=6 types. It trained well (advantage +0.27),
and a first mediator check reported the agent as bearer-selective. That check was
a false positive. A uniform diet scores 0.523 against these demand vectors, and
the champion scored 0.500, statistically identical. The agent had found a seventh
bearer-neutral escape: eat everything in balance, which covers any demand
partially and beats a random agent, without conditioning on r_i at all. The
sharpened mediator check is the fix: measure diet_norm . r_own against the uniform
value 1/K, not against a random agent. A uniform eater sits at 1/K exactly; only a
selective eater exceeds it.

**Form B (obligate energy).** Energy was made to come only from above-average-demand
types, net CAL*(r_i[k]*K - 1) per meal, so uniform eating nets zero against a
continuous maintenance cost and starves. This forecloses the escape. It also
made the task unlearnable for this agent: with r_i unobservable and the only
viable policy requiring its blind discovery over a 6-way simplex, ES with
fast-weight plasticity could not train it (advantage +0.003, mediator alignment
0.150 below the uniform 0.167, gate FAIL). Not trained, so the sweep is again
uninterpretable, this time for the opposite reason.

**The difficulty valley.** These two forms are the two sides of a valley.

    escape available   -> agent trains via the escape        -> no bearer effect
    escape foreclosed  -> only policy is hard to learn blind  -> agent does not train

Bearer-selective behavior is either unnecessary or, once made necessary, too hard
to acquire at this scale. This is not a defect of the theory. It is a finding
about learnability: an agent escapes bearer-specificity partly because a generic
policy is easier, and foreclosing the generic policy raises the learning problem
to one the substrate cannot solve. The escape principle and the difficulty valley
are the same phenomenon seen from the two sides.

## 8. Recommendation

The clean next experiment threads the valley: a learnable-and-obligate task. Shrink
the private parameter so it is tractable to learn blind while still admitting no
generic escape. The metabolic task with K=2 or 3 is exactly this: uniform eating
still nets zero and starves, but the private demand is about one bit, which
scalar energy feedback can plausibly teach through the pathway. Run that with the
sharpened mediator check. Only after a design that both trains and shows a real
mediator does the self-versus-yoked comparison finally test obligate dependence,
and only then should implementation effort move into a small model.
