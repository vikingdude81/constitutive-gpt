"""
Constitutive GPT, dual-channel model: the record present and idle, the incorporated
state load-bearing. The sharp form of Prediction 4.

M2 showed a pure incorporated-memory cell solves the obligate bandit-ID task. But a
pure cell has no context at all, so the claim that incorporation beats a retrievable
record was carried only by the gate-0 information bound, not shown inside one model.
This model has BOTH channels and lets them compete:

  - Context channel (the record). A single-head attention pools the last L action and
    reward tokens into a summary, exactly a bounded transformer context. By gate 0 an
    L-window cannot solve the task, so this channel is a present but insufficient
    record.
  - Incorporated channel (the bearer). The fast-weight associative memory W_fast of
    M2, integrating over the full horizon.

Per-arm logits are logit_k = beta * (memory read for k) + gamma * (context read for
k). Evolutionary search trains everything and is free to lean on either channel.

The Prediction 4 test is a double dissociation:
  - ablate the context (gamma -> 0): accuracy should barely move if the record is
    idle;
  - freeze the memory (c_write -> 0): accuracy should collapse if incorporation is
    load-bearing.

If both hold, the record is present in the model yet idle, and the incorporated state
carries the task and the identity. That is the record-versus-bearer distinction shown
inside one sequence model, not just argued from a bound.

Usage:  python constitutive_gpt_dual.py --gens 200
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse, json, time, math
import numpy as np
import torch

from constitutive_gpt_m1 import bayes_map_accuracy


def make_spec(K, d):
    spec = [
        ("E_arm", (K, d)), ("u_read", (d,)), ("w_val", (d,)),
        ("W_tok", (d, K + 1)), ("q_attn", (d,)),
        ("c_write", (1,)), ("c_decay", (1,)), ("o_shift", (1,)),
        ("log_beta", (1,)), ("log_gamma", (1,)),
    ]
    return [(n, s, int(np.prod(s))) for n, s in spec]


def init_theta(spec, K, d, device):
    g = torch.Generator().manual_seed(0)
    parts = {}
    parts["E_arm"] = torch.linalg.qr(torch.randn(max(K, d), d, generator=g))[0][:K]
    parts["u_read"] = torch.randn(d, generator=g) / d ** 0.5
    parts["w_val"] = torch.randn(d, generator=g) / d ** 0.5
    parts["W_tok"] = torch.randn(d, K + 1, generator=g) / (K + 1) ** 0.5
    parts["q_attn"] = torch.randn(d, generator=g) / d ** 0.5
    parts["c_write"] = torch.tensor([1.0])
    parts["c_decay"] = torch.tensor([0.01])
    parts["o_shift"] = torch.tensor([0.5])
    parts["log_beta"] = torch.tensor([math.log(3.0)])
    parts["log_gamma"] = torch.tensor([math.log(0.5)])
    flat = torch.cat([parts[n].reshape(-1).float() for n, _, _ in spec])
    return flat.to(device=device, dtype=torch.float32)


def unflat(theta, spec, K, d):
    out, i = {}, 0
    P = theta.shape[0]
    for n, s, ln in spec:
        out[n] = theta[:, i:i + ln].reshape(P, *s); i += ln
    return out


@torch.no_grad()
def rollout(theta, spec, K, d, eps, N, M, L, seed, device, tail=0.25,
            write=True, use_ctx=True):
    """theta [P,D] -> tail accuracy [P].  write=False freezes the incorporated
    memory (ablate the bearer); use_ctx=False zeros the context channel (ablate the
    record)."""
    P = theta.shape[0]
    p = unflat(theta, spec, K, d)
    E = p["E_arm"]; u = p["u_read"]; wv = p["w_val"]
    W_tok = p["W_tok"]; q = p["q_attn"]
    cw = p["c_write"].squeeze(-1) * (1.0 if write else 0.0)
    cd = p["c_decay"].squeeze(-1).clamp(0.0, 1.0)
    osh = p["o_shift"].squeeze(-1)
    beta = p["log_beta"].squeeze(-1).exp().clamp(max=50.0)
    gamma = p["log_gamma"].squeeze(-1).exp().clamp(max=50.0) * (1.0 if use_ctx else 0.0)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    theta_true = torch.randint(0, K, (P, M), generator=gen).to(device)
    W = torch.zeros(P, M, d, d, device=device)                 # incorporated memory
    buf = torch.zeros(P, M, L, d, device=device)               # context token buffer
    a_prev = torch.zeros(P, M, dtype=torch.long, device=device)
    r_prev = torch.zeros(P, M, device=device)
    have_prev = False
    Eexp = E.unsqueeze(1)
    scale = 1.0 / math.sqrt(d)
    tail_start = int(N * (1 - tail)); acc = torch.zeros(P, device=device); cnt = 0
    for t in range(N):
        # push previous (action, reward) as a token into the rolling context buffer
        if have_prev:
            tok_in = torch.cat([torch.nn.functional.one_hot(a_prev, K).float(),
                                r_prev.unsqueeze(-1)], -1)             # [P,M,K+1]
            emb = torch.einsum("pdc,pmc->pmd", W_tok, tok_in)         # [P,M,d]
            buf = torch.cat([buf[:, :, 1:, :], emb.unsqueeze(2)], dim=2)
        # context read: attention-pool the buffer, then per-arm dot with embeddings
        att = torch.softmax((torch.einsum("pmld,pd->pml", buf, q)) * scale, dim=2)
        z_ctx = torch.einsum("pml,pmld->pmd", att, buf)               # [P,M,d]
        c_k = torch.einsum("pmd,pkd->pmk", z_ctx, E)                  # [P,M,K]
        # memory read
        We = torch.einsum("pmde,pmke->pmkd", W, Eexp.expand(P, M, K, d))
        v = torch.einsum("pmkd,pd->pmk", We, u)                       # [P,M,K]
        logits = beta.view(P, 1, 1) * v + gamma.view(P, 1, 1) * c_k
        gum = -torch.log(-torch.log(torch.rand(P, M, K, device=device) + 1e-12) + 1e-12)
        a = torch.argmax(logits + gum, dim=-1)
        r = (torch.rand(P, M, device=device) < torch.where(a == theta_true, 0.5 + eps, 0.5)).float()
        o = r - osh.view(P, 1)
        e_a = torch.gather(E.unsqueeze(1).expand(P, M, K, d), 2,
                           a.view(P, M, 1, 1).expand(P, M, 1, d)).squeeze(2)
        outer = torch.einsum("pd,pme->pmde", wv, e_a)
        W = W + cw.view(P, 1, 1, 1) * o.view(P, M, 1, 1) * outer - cd.view(P, 1, 1, 1) * W
        a_prev, r_prev, have_prev = a, r, True
        if t >= tail_start:
            # score the model's actual combined decision, so ES is free to use either
            # channel and the ablations reveal which one carries the task
            acc = acc + (torch.argmax(logits, dim=-1) == theta_true).float().mean(dim=1)
            cnt += 1
    return acc / max(cnt, 1)


def es_train(spec, K, d, eps, N, M, L, pop, gens, device, log=print):
    D = sum(ln for _, _, ln in spec)
    theta = init_theta(spec, K, d, device)
    sigma, lr = 0.1, 0.15
    gen = torch.Generator(device=device).manual_seed(7)
    best, best_theta = -1.0, theta.clone()
    for gi in range(gens):
        t0 = time.time()
        noise = torch.randn(pop // 2, D, device=device, generator=gen)
        eps_all = torch.cat([noise, -noise], 0)
        fits = rollout(theta.unsqueeze(0) + sigma * eps_all, spec, K, d, eps, N, M, L, 100 + gi, device)
        ranks = torch.empty_like(fits); ranks[fits.argsort()] = torch.arange(len(fits), device=device, dtype=fits.dtype)
        adv = ranks / (len(fits) - 1) - 0.5
        theta = theta + lr * (adv.unsqueeze(1) * eps_all).sum(0) / (pop * sigma)
        if gi % 5 == 0 or gi == gens - 1:
            ho = float(rollout(theta.unsqueeze(0), spec, K, d, eps, N, M, L, 9000 + gi, device)[0])
            if ho > best: best, best_theta = ho, theta.clone()
            log(f"[gen {gi:03d}] pop_mean={fits.mean():.3f} held-out={ho:.3f} best={best:.3f} ({time.time()-t0:.1f}s)")
    return best_theta, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--M", type=int, default=48)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--load", default="")
    ap.add_argument("--save", default="champ_dual.npz")
    ap.add_argument("--out", default="results_dual.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = make_spec(args.K, args.d)
    chance = 1.0 / args.K
    ctx = bayes_map_accuracy(args.K, args.eps, args.L, 8000, 1)
    print(f"[DUAL] K={args.K} eps={args.eps} d={args.d} N={args.N} L={args.L} pop={args.pop} device={device}")
    print(f"       chance={chance:.3f}  context-only bound(L={args.L})={ctx:.3f}")

    if args.load:
        theta = torch.tensor(np.load(args.load)["theta"], device=device, dtype=torch.float32)
        print(f"[DUAL] loaded {args.load}")
    else:
        theta, best = es_train(spec, args.K, args.d, args.eps, args.N, args.M, args.L, args.pop, args.gens, device)
        np.savez(args.save, theta=theta.cpu().numpy())
        print(f"[DUAL] trained (best {best:.3f}), saved {args.save}")

    th = theta.unsqueeze(0)
    ev = lambda **kw: float(rollout(th, spec, args.K, args.d, args.eps, args.N, 256, args.L, 555, device, **kw)[0])
    full = ev(write=True, use_ctx=True)
    mem_only = ev(write=True, use_ctx=False)     # ablate the record (context)
    ctx_only = ev(write=False, use_ctx=True)     # freeze the bearer (memory)
    learned_gamma = float(unflat(th, spec, args.K, args.d)["log_gamma"].exp().squeeze())
    learned_beta = float(unflat(th, spec, args.K, args.d)["log_beta"].exp().squeeze())

    print(f"\n[DOUBLE DISSOCIATION]")
    print(f"  full (both channels)          {full:.3f}")
    print(f"  ablate context (memory only)  {mem_only:.3f}   (record removed)")
    print(f"  freeze memory (context only)  {ctx_only:.3f}   (bearer removed)")
    print(f"  learned weights: beta(memory)={learned_beta:.2f}  gamma(context)={learned_gamma:.2f}")
    record_idle = abs(full - mem_only) < 0.06
    bearer_loadbearing = (full - ctx_only) > 0.3
    print(f"\n[VERDICT] record idle (full ~= memory-only): {'YES' if record_idle else 'no'} "
          f"(delta {full - mem_only:+.3f}) | bearer load-bearing (full >> context-only): "
          f"{'YES' if bearer_loadbearing else 'no'} (delta {full - ctx_only:+.3f})")
    if record_idle and bearer_loadbearing:
        print("  DUAL PASS: the model has a real context channel yet the incorporated memory carries "
              "the task. Removing the record barely moves accuracy; freezing the incorporated state "
              "collapses it. The record-versus-bearer distinction shown inside one sequence model, the "
              "sharp form of Prediction 4. The name Constitutive GPT is earned.")
    else:
        why = []
        if not record_idle: why.append(f"record not idle (context contributes {full - mem_only:+.3f})")
        if not bearer_loadbearing: why.append(f"bearer not load-bearing (memory contributes {full - ctx_only:+.3f})")
        print("  DUAL INCOMPLETE: " + "; ".join(why) + ".")

    json.dump({"full": full, "memory_only": mem_only, "context_only": ctx_only,
               "context_bound": ctx, "chance": chance, "beta": learned_beta, "gamma": learned_gamma,
               "record_idle": bool(record_idle), "bearer_loadbearing": bool(bearer_loadbearing)},
              open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
