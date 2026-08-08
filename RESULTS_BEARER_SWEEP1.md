# Bearer sweep #1: a null that is an artifact, and the design fix it revealed

Run 2026-07-25 on the trained GPU champion (`champion_gpu_big.npz`, d=128,
+0.32 over random on held-out worlds). 24 worlds, lifetime 2000, paired
within-world self-vs-wrong_bearer design.

## The measured null

Delta P(lambda) = perf(self) - perf(wrong_bearer):

```
lambda      self            wrong_bearer      deltaP
0.00   -0.4639+/-0.0588  -0.4316+/-0.0329  -0.0323+/-0.0571
0.25   -0.5134+/-0.0798  -0.4469+/-0.0426  -0.0665+/-0.0873
0.50   -0.4357+/-0.0433  -0.4835+/-0.0671  +0.0478+/-0.0864
0.75   -0.5193+/-0.0728  -0.4770+/-0.0675  -0.0423+/-0.1048
1.00   -0.4837+/-0.0672  -0.5239+/-0.0873  +0.0402+/-0.1123
```

Delta P is within its CI of 0 at every lambda, non-monotonic, signs flipping.
The falsifier passes: Delta P(0) = -0.032 +/- 0.057, consistent with 0, so the
plumbing is correct.

## Why the null is uninterpretable

The bearer manipulation (self vs wrong_bearer) acts ONLY through the
outcome-driven pathway: the outcome value feeds the Hebbian fast-weight update
and the bearer-state GRU, which in turn shape behavior. If a trained agent does
not rely on that pathway, the manipulation has no surface to act on and Delta P
is structurally ~0 regardless of whether bearer-identity matters in principle.

An ablation confirms this is exactly what happened. On 8 held-out worlds,
lifetime 2000:

```
plastic pathway ON  : -0.4265 +/- 0.0395
plastic pathway OFF : -0.4218 +/- 0.0492   (fast weights and bearer state frozen)
difference (ON-OFF) : -0.0047              (within noise)
mean ||P_fast|| end : 3.75                 (the fast weights DO move)
```

Turning the entire outcome-driven pathway off changes performance by 0.005.
The fast weights move (norm 3.75, not zero), but moving them does not affect
what the agent achieves. ES solved foraging with the base recurrent policy
(GRU + trained weights + the bearer FiLM at init) and left online plasticity
doing nothing load-bearing.

So this is sweep #1's failure in a subtler form. Sweep #1's agent was
untrained. This agent is genuinely trained (beats random by +0.32), but trained
in a way that bypasses the very channel the bearer experiment probes. Being
competent is not sufficient; the agent must DEPEND on consequence-driven
adaptation for the bearer question to have any purchase.

## The root cause is conceptual, not just a bug

The world is static (`drift_every=0`). A static forager is solvable by a fixed
policy, so there is no role for continually adapting from one's own outcomes,
so the bearer-identity of those outcomes cannot matter. Whose body a
consequence belongs to is only load-bearing where the agent MUST keep adapting
to consequences to survive.

This is actually the sharper version of the paper's own claim: bearer-indexing
has purchase exactly where online adaptation is necessary. A world you can
solve once and coast through is the wrong test bed by construction.

## The fix

Re-run the whole chain in a DRIFTING world (`drift_every` > 0, e.g. every few
thousand steps), where resource values and hazards move, so a fixed policy
decays and the agent must keep updating from its own outcomes. Then:

1. Retrain the champion under drift (the base policy alone can no longer coast,
   forcing reliance on the plastic/bearer pathway).
2. Re-run the ablation check FIRST: confirm plastic ON beats plastic OFF by a
   real margin. Only if the pathway is load-bearing is the bearer sweep
   interpretable.
3. Then run the bearer sweep. Now a null would mean something, and a positive
   Delta P(lambda) would be the real result.

Until the ablation shows the pathway is load-bearing, no bearer sweep on this
agent (or any agent trained in a static world) can be interpreted.

## What still stands

The infrastructure is validated and reusable: the GPU batched evaluator, the
held-out training with a functional test, the paired within-world bearer sweep,
and the falsifier all work. The ablation check is now a required gate before
any bearer result. The negative finding here is real and useful: it identifies
the necessary condition (a world requiring online adaptation) that the original
milestone-1 design omitted.
