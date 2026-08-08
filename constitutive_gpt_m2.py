"""
Constitutive GPT, milestone M2: the incorporated model.

Per CONSTITUTIVE_GPT_SCOPE.md and RESULTS_M2_PROBE.md. The probe showed an ES-tuned
incorporated integrator solves the obligate bandit-ID task (M1). M2 replaces the
hand-shaped scalar accumulator with the real mechanism the Constitutive GPT is meant
to use, and asks whether ES can learn to use it.

The incorporated state is a fast-weight associative memory W_fast in R^{d x d},
reset each episode, the sequence-model analogue of the forager's bearer fast weights:

  - read: for each arm k, v_k = u . (W_fast e_k), a bilinear read against a learned
    arm embedding e_k. With near-orthonormal embeddings, v_k accumulates the
    gated reward written for arm k, so v is a per-arm value estimate.
  - policy: a ~ softmax(beta v). Explores when W_fast is empty (v flat), exploits as
    it fills.
  - write (the incorporated update, Lever B form): on the agent's own outcome,
    W_fast += c_write * o_n * outer(w_val, e_a) - c_decay * W_fast, where a is the
    pulled arm and o_n is the centered reward. This is an outcome-gated Hebbian
    write into the associative memory, keyed by the chosen arm.

Nothing here is hand-indexed: ES learns the embeddings, the read and write vectors,
the write gains, and the policy temperature. The question M2 answers is whether ES
finds a setting where the fast-weight memory carries the per-arm integration, not
whether we can hand-build one.

Two stop-rule checks (the mediator / ablation gate for this substrate):
  - the trained agent beats the context-only baseline (~0.170 at the M1 calibration);
  - freezing the write (c_write -> 0) collapses accuracy toward chance, proving the
    incorporated state is load-bearing rather than idle.

Usage:  python constitutive_gpt_m2.py --K 8 --eps 0.06 --N 2000 --gens 120
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse, json, time
import numpy as np
import torch

from constitutive_gpt_m1 import bayes_map_accuracy


def make_spec(K, d):
    """Parameter layout for one policy. Returns list of (name, shape, numel)."""
    spec = [
        ("E_arm", (K, d)), ("u_read", (d,)), ("w_val", (d,)),
        ("c_write", (1,)), ("c_decay", (1,)), ("o_shift", (1,)), ("log_beta", (1,)),
    ]
    return [(n, s, int(np.prod(s))) for n, s in spec]


def init_theta(spec, K, d, device):
    """Sensible init: near-orthonormal arm embeddings, unit read/write, small gains."""
    parts = {}
    g = torch.Generator().manual_seed(0)
    E = torch.linalg.qr(torch.randn(max(K, d), d, generator=g))[0][:K]   # near-orthonormal rows
    parts["E_arm"] = E
    parts["u_read"] = torch.randn(d, generator=g) / d ** 0.5
    parts["w_val"] = torch.randn(d, generator=g) / d ** 0.5
    parts["c_write"] = torch.tensor([1.0])
    parts["c_decay"] = torch.tensor([0.01])
    parts["o_shift"] = torch.tensor([0.5])
    parts["log_beta"] = torch.tensor([np.log(3.0)])
    flat = torch.cat([parts[n].reshape(-1).float() for n, _, _ in spec])
    return flat.to(device=device, dtype=torch.float32)


def unflat(theta, spec, K, d):
    """theta [P, D] -> dict of [P, *shape]."""
    out, i = {}, 0
    P = theta.shape[0]
    for n, s, ln in spec:
        out[n] = theta[:, i:i + ln].reshape(P, *s)
        i += ln
    return out


@torch.no_grad()
def rollout(theta, spec, K, d, eps, N, M, seed, device, tail=0.25, write=True):
    """Evaluate policies theta [P, D] on M instances each. Returns tail accuracy [P].

    write=False freezes the incorporated update (c_write -> 0): the ablation gate.
    """
    P = theta.shape[0]
    p = unflat(theta, spec, K, d)
    E = p["E_arm"]                                   # [P,K,d]
    u = p["u_read"]                                  # [P,d]
    wv = p["w_val"]                                  # [P,d]
    cw = p["c_write"].squeeze(-1) * (1.0 if write else 0.0)   # [P]
    cd = p["c_decay"].squeeze(-1).clamp(0.0, 1.0)    # [P]
    osh = p["o_shift"].squeeze(-1)                   # [P]
    beta = p["log_beta"].squeeze(-1).exp().clamp(max=50.0)    # [P]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    theta_true = torch.randint(0, K, (P, M), generator=gen).to(device)     # hidden arm
    W = torch.zeros(P, M, d, d, device=device)
    tail_start = int(N * (1 - tail))
    acc = torch.zeros(P, device=device); cnt = 0
    Eexp = E.unsqueeze(1)                            # [P,1,K,d]
    for t in range(N):
        # read: v_k = u . (W e_k)
        We = torch.einsum("pmde,pmke->pmkd", W, Eexp.expand(P, M, K, d))
        v = torch.einsum("pmkd,pd->pmk", We, u)      # [P,M,K]
        logits = beta.view(P, 1, 1) * v
        gum = -torch.log(-torch.log(torch.rand(P, M, K, device=device) + 1e-12) + 1e-12)
        a = torch.argmax(logits + gum, dim=-1)       # [P,M] sampled action
        correct = (a == theta_true)
        pr = torch.where(correct, 0.5 + eps, torch.full_like(v[..., 0], 0.5))
        r = (torch.rand(P, M, device=device) < pr).float()          # [P,M]
        o = r - osh.view(P, 1)                        # centered reward
        # gather arm embedding e_a: [P,M,d]
        e_a = torch.gather(E.unsqueeze(1).expand(P, M, K, d), 2,
                           a.view(P, M, 1, 1).expand(P, M, 1, d)).squeeze(2)
        outer = torch.einsum("pd,pme->pmde", wv, e_a)               # outer(w_val, e_a) [P,M,d,d]
        W = W + cw.view(P, 1, 1, 1) * o.view(P, M, 1, 1) * outer - cd.view(P, 1, 1, 1) * W
        if t >= tail_start:
            acc = acc + (torch.argmax(v, dim=-1) == theta_true).float().mean(dim=1)
            cnt += 1
    return acc / max(cnt, 1)


def es_train(spec, K, d, eps, N, M, pop, gens, device, log=print, seed=7):
    D = sum(ln for _, _, ln in spec)
    theta = init_theta(spec, K, d, device)
    sigma, lr = 0.1, 0.15
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_theta = -1.0, theta.clone()
    for gi in range(gens):
        t0 = time.time()
        noise = torch.randn(pop // 2, D, device=device, generator=gen)
        eps_all = torch.cat([noise, -noise], 0)
        fits = rollout(theta.unsqueeze(0) + sigma * eps_all, spec, K, d, eps, N, M, 100 + gi, device)
        ranks = torch.empty_like(fits); ranks[fits.argsort()] = torch.arange(len(fits), device=device, dtype=fits.dtype)
        adv = ranks / (len(fits) - 1) - 0.5
        theta = theta + lr * (adv.unsqueeze(1) * eps_all).sum(0) / (pop * sigma)
        if gi % 5 == 0 or gi == gens - 1:
            ho = float(rollout(theta.unsqueeze(0), spec, K, d, eps, N, M, 9000 + gi, device)[0])
            if ho > best: best, best_theta = ho, theta.clone()
            log(f"[gen {gi:03d}] pop_mean={fits.mean():.3f} held-out={ho:.3f} best={best:.3f} ({time.time()-t0:.1f}s)")
    return best_theta, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--d", type=int, default=48, help="model width / fast-weight dim")
    ap.add_argument("--N", type=int, default=2000, help="training horizon")
    ap.add_argument("--Neval", type=int, default=4000, help="eval horizon (M1 full)")
    ap.add_argument("--M", type=int, default=48, help="instances per policy")
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=120)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--out", default="results_m2.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = make_spec(args.K, args.d)
    chance = 1.0 / args.K
    ctx = bayes_map_accuracy(args.K, args.eps, args.L, 8000, 1)
    print(f"[M2] K={args.K} eps={args.eps} d={args.d} N={args.N} pop={args.pop} device={device}")
    print(f"     chance={chance:.3f}  context-only(L={args.L})={ctx:.3f}")

    theta, best = es_train(spec, args.K, args.d, args.eps, args.N, args.M, args.pop, args.gens, device)

    # eval at full horizon, and the ablation gate (freeze the write)
    acc_full = float(rollout(theta.unsqueeze(0), spec, args.K, args.d, args.eps, args.Neval, 256, 555, device)[0])
    acc_train = float(rollout(theta.unsqueeze(0), spec, args.K, args.d, args.eps, args.N, 256, 555, device)[0])
    acc_frozen = float(rollout(theta.unsqueeze(0), spec, args.K, args.d, args.eps, args.N, 256, 555, device, write=False)[0])

    print(f"\n[RESULT] trained held-out (N={args.N}) = {acc_train:.3f} | full horizon (N={args.Neval}) = {acc_full:.3f}")
    print(f"[GATE]   write ON = {acc_train:.3f}  vs  write FROZEN = {acc_frozen:.3f}  "
          f"(drop {acc_train - acc_frozen:+.3f})")
    beat = acc_train > ctx + 0.05
    mediator = (acc_train - acc_frozen) > 0.05 and acc_frozen < ctx + 0.08
    print(f"[VERDICT] beats context-only ({ctx:.3f}): {'YES' if beat else 'no'} | "
          f"incorporated state load-bearing: {'YES' if mediator else 'no'}")
    if beat and mediator:
        print("  M2 PASS: the fast-weight incorporated memory carries the per-arm integration, "
              "trained end to end by ES, and freezing it collapses performance. Proceed to M3 "
              "(self versus wrong-bearer, sharpened Prediction 2).")
    else:
        why = []
        if not beat: why.append("does not beat the context-only baseline (integration not learned)")
        if not mediator: why.append("frozen-write does not collapse (state idle or task leaking)")
        print("  M2 INCOMPLETE: " + "; ".join(why) + ".")

    json.dump({"acc_train": acc_train, "acc_full": acc_full, "acc_frozen": acc_frozen,
               "context_only": ctx, "chance": chance, "beat": bool(beat), "mediator": bool(mediator)},
              open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
