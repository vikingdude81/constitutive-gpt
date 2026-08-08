# Testbed Extensions: the two Chapter 9.9 items, and the predictions they unlock

Built 2026-08-02 to complete the Chapter 9.9 testbed spec. Adds independent
external logs (item 7) and constitution/record exchange and corruption ablations
(item 8), which unlock Bearer Theory Predictions 3, 4, and the exchange half of 4.
`testbed_extensions.py`. Demo run on champion_gpu_drift.npz (d=64, drift 200).

## What was added

- **EventLog (item 7).** An auditor's record of every step (action, resource
  eaten, outcome components, |dP|, |db|), held in a separate object. The agent's
  act() and plastic_update() never take it as input, so by construction it is a
  record, not bearer state. Exportable to npz. Demonstrates the manuscript's
  discipline that persistence of information is not persistence of the bearer.

- **Exchange and ablation access (item 8).** We designate the fast-weight store P
  as the external record m and the recurrent bearer state b as the incorporated
  bearer state, following Chapter 5's external-record vs changed-actor split.
  Both can be exchanged, corrupted, or removed independently.

## Demo results (one champion, illustrative not conclusive)

**Prediction 4 (record vs bearer-state exchange).** Four conditions in one shared
world, plasticity frozen so the injected state is the only cause of behavior:
```
(P_A,b_A)=-0.389  (P_B,b_A)=-0.389  (P_A,b_B)=-0.381  (P_B,b_B)=-0.390
spread with P fixed: 0.009 | with b fixed: 0.009
```
Marginal, a faint incorporation lean. This champion re-finds its homeostatic
setpoint from almost any injected state, so it shows little storage-vs-
incorporation asymmetry. The instrument is correct; the agent is simply robust.
An agent whose competence lives in a specific developed configuration would
separate the conditions more sharply.

**Item 8 signature (corruption / removal).** More informative:
```
intact=-0.499  record-corrupted(donor P)=-0.551  record-removed(P=0)=-0.943  bearer-removed(b=0)=-0.883
record sensitivity (intact - removed): +0.444
bearer sensitivity (intact - removed): +0.383
```
Removing either store collapses coupling toward random. Corrupting the record
with a donor's P costs only 0.05. Presence of a developed store matters; its
specific content is tolerated. This is the presence-tax/content-indifference
pattern from the holographic-retrieval probe, now seen at the fast-weight-store
level of the actor itself.

**Prediction 3 (delayed counterfactual incorporation).** Omitting a 25-step
window of updates and measuring bearer-state divergence 300 steps AFTER the cue
is gone:
```
||b_full - b_omit|| = 5.76
```
Large and delayed. The specific omitted outcomes were historically incorporated
and remained causally present in the bearer long after the event, not merely held
in immediate context. The action stream was held identical across the two runs,
so the divergence is incorporation, not different sampled actions. A single-step
omission gave zero divergence (the chosen step was a null outcome), which is why
the window version is the correct manipulation.

## Status against Chapter 9.9

With these additions the CCA build satisfies all ten items of the 9.9 spec:
items 7 and 8 were the two gaps, now closed. The testbed can run Predictions 3
(delayed incorporation), 4 (record vs bearer-state exchange), and the corruption
and removal controls, in addition to the bearer sweep, the ablation gate, and the
construction-order variants already present.

## Honest bounds

These are single-champion demonstrations of working instruments, not a study. The
Prediction 4 exchange in particular is underpowered on a champion with a robust
setpoint. The natural next use is to run these instruments across the
constitution-strength dose curve, where an agent forced into taste selectivity
should show sharper exchange asymmetry and larger developmental contributions.
