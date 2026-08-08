# The Bearer Observability Window and the Observability Tension

*Formal note, revised 2026-08-03 after the K=2 result. States the conditions
under which a bearer-identity performance effect can be observed, and the tension
that squeezes them. Grounded in nine CCA task designs. Its central positive cell
is predicted but not yet observed in this substrate. No em-dashes.*

---

## 1. What the designs measured

Nine task designs produced no bearer-neutral performance advantage, and the
reasons sort into a small structure. Reading them together gives three necessary
conditions and one tension, which is more informative than any single positive
would have been.

## 2. Three conditions for an observable bearer effect

Let J_i(pi) be the neutral viability of policy pi for bearer i, tau the viable
threshold, Pi_free the bearer-independent policies, and pi_b* the best
bearer-conditioned policy.

**C1, no escape.**

    E = 0  iff  no bearer-independent policy is viable for all bearers:
                for every pi in Pi_free there is a bearer i with J_i(pi) < tau.

If a viable generic policy exists (E = 1), a flexible optimizer finds it and
identity stays idle. Seven designs sat here.

**C2, on-pathway (the gate).** The bearer-conditioned behavior must be produced
by the incorporated outcome-driven pathway, the one the self-versus-yoked
manipulation acts on, and not by an observable or retrievable shortcut. Measured
by the ablation gate: freezing the incorporated pathway must cost viability.

**C3, learnable.**

    L = 1  iff  the agent, trained on its own outcomes over horizon H, converges
                near pi_b* through the incorporated pathway.

Read from two gates: trains above random, and the mediator is present.

The self-versus-yoked dose relation dG/dlambda > 0 is observable only if all
three hold:

    observable  iff  (E = 0) and (gate passes) and (L = 1).

## 3. The observability tension (the central result)

C2 and C3 are pulled apart by a single design variable: whether the agent can
observe the state it needs.

- If the agent can observe its own relevant state (energy, diet, recent history),
  learning becomes tractable, which helps C3. But it then adapts through the
  observable and recurrent route, which bypasses the incorporated pathway and
  fails C2. The manipulation acts on an idle channel and the effect is null even
  when bearer-selective behavior is present.
- If the agent is blind to that state, the only channel left is the incorporated
  pathway, which satisfies C2. But blind acquisition of the bearer-conditioned
  policy from scalar outcomes is hard, which fails C3, and training collapses.

Observability is therefore both the enabler of learning and the provider of the
bypass. The window (E = 0, gate passes, L = 1) is squeezed from both sides, and
in the ES plus fast-weight substrate it was not reached.

## 4. Every design placed

    design                         E   gate   L    outcome
    static forager                 1    -     -     escape null (idle)
    drift forager                  1    P     1     escape null (timing)
    blind forager                  1    P     1     escape null (timing)
    dose curve (const_sigma)       1    P     1     escape null (quantity)
    obligate poison                1    P     ~     escape null (tolerate)
    metabolism K=6, composition    1    -     -     escape null (uniform)
    metabolism K=6, obligate       0   FAIL   0     collapse null (unlearnable)
    metabolism K=2, obligate       0   FAIL  ~0.5   partial-learn null (off-pathway)

The K trend is real and directional. At K=6 obligate the mediator alignment was
0.150, below the uniform value 0.167, so no selectivity at all. At K=2 obligate it
was 0.552, above the uniform 0.500, so weak selectivity arose. Fewer bits in the
private parameter did improve learnability, as the two-axis form predicted. But at
K=2 the selectivity came through the observable-energy recurrent route, not the
incorporated pathway (gate FAIL), and it was too weak to cross the steep obligate
viability threshold (not trained). So K=2 sits on the boundary of C3 while failing
C2. The positive cell (E = 0, gate passes, L = 1) remains empty in this substrate.

## 5. The unifying principle

The three conditions and the tension reduce to one statement:

    Incorporation becomes load-bearing exactly when the information the agent needs
    is not available from an observable or retrievable source. The bearer state
    matters precisely when the agent cannot read its history off a record or sense
    it from the environment.

This subsumes the manuscript's Prediction 4 (record versus incorporated bearer): a
record is a retrievable source, and whenever the needed information sits in the
record or in observable state, the incorporated pathway is bypassed and identity
is idle. It also explains every escape: each escape is a bearer-neutral route that
supplies viability from generic, non-incorporated information.

## 6. What it predicts and where to look

The window is not shown empty in general, only unreached by this substrate. Three
levers were identified, and the cheap one has now been tested.

1. **A curriculum (tested, partial progress).** Begin with observability, so the
   agent can learn to forage, then ramp observation-blindness from 0 to 1 during
   training, pushing the agent off the observable shortcut onto the incorporated
   pathway after it is already competent. Run on the K=2 obligate task (metabolic,
   100 generations, blindness ramp 0 to 1, evaluated fully blind). Result, read
   carefully:
   - The mediator was present at FULL BLINDNESS, alignment 0.656 against a uniform
     value of 0.500 and a ceiling of 0.812. With no observable state, this
     selectivity can only route through the incorporated pathway. It is the
     highest alignment of any run (K=6: 0.150; cold-start K=2: 0.552; curriculum
     K=2: 0.656). The curriculum genuinely moved bearer-selective behavior onto
     the pathway.
   - But the agent did not reach viability (advantage +0.003, held-out at the
     random floor). The 0.656 alignment was not strong enough to lift energy into
     band under the steep obligate threshold.
   - The gate read FAIL (0.000), but this is a floor artifact, not proof of an idle
     pathway. The gate compares viability with the pathway on versus off, and when
     the agent is below the viability floor in both conditions the comparison
     cannot resolve the pathway's contribution. On a sub-viable agent the gate is
     uninformative.

   The honest reading is progress, not closure. The curriculum increased
   pathway-based selectivity toward the ceiling, but fixed Hebbian could not push
   it across the viability threshold, so the downstream gate and sweep went dark.
   The window is hard and approachable in this substrate, not shown empty.

2. **Calibrate the viability threshold (tested, negative: the squeeze moves to the
   gain axis).** Ran a `--gain-scale` multiplier on net = CAL*GAIN*(2r - 1), which
   amplifies the reward for above-uniform selection while keeping the zero-crossing
   pinned at uniform so no bearer-neutral escape reopens by construction. Two points
   bracket the outcome (RESULTS_LEVER_A.md):
   - GAIN=2 (strict): mediator present but weak (0.567), advantage +0.006 at the
     random floor, gate dark (+0.002). Sub-viable.
   - GAIN=5 (soft): advantage +0.027 off the floor, gate PASSES (+0.032), but the
     mediator collapses to near-uniform (0.542) and the sweep is null. Viability is
     reached via a quasi-uniform high-volume quantity route, not selectivity.

   So softening the threshold enough to be viable re-admits a near-uniform quantity
   route (selectivity washes out); keeping it strict enough to force selectivity
   leaves the agent sub-viable. The calibration does not open the clean window; it
   reproduces the observability tension along the gain axis. The blind held-out
   score also sat at the random floor across the whole GAIN=2 run, localizing the
   binding constraint to learnability: the fixed-Hebbian inner loop cannot acquire
   strong blind selectivity. This is the cheap test done, and it points at lever 3.

3. **A stronger inner-loop learner (tested, negative: the escape is meta-level).**
   Replaced the fixed gated-Hebbian update with a 4-term learned rule (eligibility
   and instantaneous Hebbian, decay, outcome-modulated forgetting), ES-optimized
   (RESULTS_LEVER_B.md). At GAIN=5 against the matched fixed-rule control:
   - control (fixed Hebbian): mediator 0.542 absent, advantage +0.027.
   - learned rule: mediator 0.507 absent, advantage +0.039 (best of any run).

   The richer rule raised viability to the highest of any run and lowered
   selectivity to essentially uniform. More expressive plasticity did not buy
   bearer-selectivity; it bought a better bearer-neutral escape. So the residual
   failure is NOT that the update rule was too weak. The Generic-Policy Escape
   Principle operates at the meta level: a more capable learner does not discover
   its bearer identity, it exploits the bearer-neutral route more efficiently.
   Levers A and B fail for complementary reasons (softening viability re-admits a
   quantity escape; strengthening the learner exploits it better), which together
   close the forager program by tuning.

4. **A more powerful bearer, the Constitutive GPT (the destination).** In-context
   attention can do the blind inference fast-weights struggle with. The same
   tension applies with a sharper edge: the KV cache and context window are the
   record, the observable bypass. The GPT makes its incorporated state load-bearing
   only on tasks whose required information cannot be held in context, the concrete
   design constraint and the technical reason the incorporated state must be
   genuinely persistent rather than reconstructed each call.

The forager is at diminishing returns but not exhausted. The curriculum showed the
window is approachable; a threshold calibration or a stronger update rule may reach
it, and the GPT is the destination. What is closed is the search by task variation
alone; the productive moves now are the viability calibration, the update rule, and
the model.

## 7. Honest status

This is a framework refined across nine designs plus a curriculum test, not a
proven law. Its three conditions are each demonstrated: E = 1 by seven escape
nulls, C2 by the K=2 gate behavior, C3 = 0 by the K=6 collapse. The tension is
demonstrated by the boundary cases. The curriculum then showed the window is
approachable: bearer-selective behavior was moved onto the incorporated pathway at
full blindness (alignment 0.656, the best of any run), but the agent stayed below
the steep viability threshold, so the viability-based gate could not yet resolve
the pathway on a sub-viable agent. The positive cell is predicted, motivated, and
approached but not occupied.

A measurement lesson also lands here: the ablation gate is informative only on a
trained agent. On a sub-viable agent it compares two floored performances and
reads FAIL regardless of what the pathway is doing. Mediator alignment, which is
defined on behavior rather than viability, remains informative below the floor and
is the more sensitive instrument in that regime.

The value of the framework is that it converts every null into a labeled
measurement of E, gate, and L, names the obstacle (the observability tension), and
now locates the productive moves: calibrate the viability threshold so partial
pathway selectivity registers, strengthen the update rule, and ultimately build a
persistent incorporated model. What is closed is the search by task variation
alone.
