"""
Constitutive GPT, M4 scale, Phase 0 calibration.

The dual-channel model used a pooling context and width 32, about 650 parameters.
Phase 0 asks the go/no-go question for scaling within the evolutionary-search
paradigm: does ES still train the model once we add a genuine transformer block, a
single-head attention with learned Q, K, V, O projections and an MLP, at a larger
width. That raises the parameter count by roughly two orders of magnitude, and ES is
a black-box optimizer whose efficiency falls as parameters grow, so this is where the
ceiling of the ES approach shows up.

Architecture per step:
  - token embed of the previous (action, reward), pushed into a rolling L-window;
  - one transformer block over the window: single-head attention with learned
    W_q, W_k, W_v, W_o and a two-layer MLP, both with residual connections, giving a
    context summary z_ctx (the record);
  - the incorporated fast-weight associative memory W_fast (the bearer), as in the
    dual-channel model;
  - per-arm logits = beta * (memory read) + gamma * (z_ctx dot arm embedding).

Go/no-go: at the target width, does held-out accuracy climb clearly above the
context-only baseline. If yes, ES holds and Phase 1 (the full scaled result) is
worth running. If it stalls near baseline, ES has hit its ceiling and the scale path
needs gradient-based training instead.

Usage:  python constitutive_gpt_scale.py --d 128 --gens 80 --N 1000
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse, json, time, math
import numpy as np
import torch

from constitutive_gpt_m1 import bayes_map_accuracy


def make_spec(K, d, h):
    spec = [
        ("E_arm", (K, d)), ("u_read", (d,)), ("w_val", (d,)),
        ("W_tok", (d, K + 1)), ("q0", (d,)),
        ("W_q", (d, d)), ("W_k", (d, d)), ("W_v", (d, d)), ("W_o", (d, d)),
        ("W_mlp1", (h, d)), ("W_mlp2", (d, h)),
        ("c_write", (1,)), ("c_decay", (1,)), ("o_shift", (1,)),
        ("log_beta", (1,)), ("log_gamma", (1,)),
    ]
    return [(n, s, int(np.prod(s))) for n, s in spec]


def init_theta(spec, K, d, h, device):
    g = torch.Generator().manual_seed(0)
    P = {}
    P["E_arm"] = torch.linalg.qr(torch.randn(max(K, d), d, generator=g))[0][:K]
    for n in ("u_read", "w_val", "q0"):
        P[n] = torch.randn(d, generator=g) / d ** 0.5
    P["W_tok"] = torch.randn(d, K + 1, generator=g) / (K + 1) ** 0.5
    for n in ("W_q", "W_k", "W_v", "W_o"):
        P[n] = torch.randn(d, d, generator=g) / d ** 0.5
    P["W_mlp1"] = torch.randn(h, d, generator=g) / d ** 0.5
    P["W_mlp2"] = torch.randn(d, h, generator=g) / h ** 0.5
    P["c_write"] = torch.tensor([1.0]); P["c_decay"] = torch.tensor([0.01])
    P["o_shift"] = torch.tensor([0.5]); P["log_beta"] = torch.tensor([math.log(3.0)])
    P["log_gamma"] = torch.tensor([math.log(0.5)])
    flat = torch.cat([P[n].reshape(-1).float() for n, _, _ in spec])
    return flat.to(device=device, dtype=torch.float32)


def unflat(theta, spec, shapes):
    out, i = {}, 0
    Pn = theta.shape[0]
    for n, s, ln in spec:
        out[n] = theta[:, i:i + ln].reshape(Pn, *s); i += ln
    return out


@torch.no_grad()
def rollout(theta, spec, K, d, h, eps, N, M, L, seed, device, tail=0.25, write=True,
            use_ctx=True, lam=0.0, alt=None):
    """alt=None trains/evaluates on own outcomes. alt='indep' mixes in a different
    instance's outcomes (wrong-bearer), alt='same' mixes an independent stream from
    the SAME code (falsifier); lam is the mix weight. Accuracy always scored on own."""
    Pn = theta.shape[0]
    p = unflat(theta, spec, None)
    E = p["E_arm"]; u = p["u_read"]; wv = p["w_val"]; q0 = p["q0"]; W_tok = p["W_tok"]
    Wq, Wk, Wv, Wo = p["W_q"], p["W_k"], p["W_v"], p["W_o"]
    M1, M2 = p["W_mlp1"], p["W_mlp2"]
    cw = p["c_write"].squeeze(-1) * (1.0 if write else 0.0)
    cd = p["c_decay"].squeeze(-1).clamp(0.0, 1.0)
    osh = p["o_shift"].squeeze(-1)
    beta = p["log_beta"].squeeze(-1).exp().clamp(max=50.0)
    gamma = p["log_gamma"].squeeze(-1).exp().clamp(max=50.0) * (1.0 if use_ctx else 0.0)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    theta_true = torch.randint(0, K, (Pn, M), generator=gen).to(device)
    if alt == "same":
        theta_alt = theta_true.clone()
    elif alt == "indep":
        theta_alt = torch.randint(0, K, (Pn, M), generator=gen).to(device)
    else:
        theta_alt = None
    W = torch.zeros(Pn, M, d, d, device=device)
    buf = torch.zeros(Pn, M, L, d, device=device)
    a_prev = torch.zeros(Pn, M, dtype=torch.long, device=device)
    r_prev = torch.zeros(Pn, M, device=device)
    have = False
    Eexp = E.unsqueeze(1)
    scale = 1.0 / math.sqrt(d)
    tstart = int(N * (1 - tail)); acc = torch.zeros(Pn, device=device); cnt = 0
    for t in range(N):
        if have:
            tok = torch.cat([torch.nn.functional.one_hot(a_prev, K).float(), r_prev.unsqueeze(-1)], -1)
            emb = torch.einsum("pdc,pmc->pmd", W_tok, tok)
            buf = torch.cat([buf[:, :, 1:, :], emb.unsqueeze(2)], dim=2)
        # transformer block: q0 attends over the window (learned Q/K/V/O), then MLP
        q = torch.einsum("pd,pde->pe", q0, Wq)                     # [P,d]
        Kk = torch.einsum("pmld,pde->pmle", buf, Wk)              # [P,M,L,d]
        Vv = torch.einsum("pmld,pde->pmle", buf, Wv)
        att = torch.softmax(torch.einsum("pe,pmle->pml", q, Kk) * scale, dim=2)
        z = torch.einsum("pml,pmle->pme", att, Vv)               # [P,M,d]
        z = torch.einsum("pme,ped->pmd", z, Wo)
        z = q0.unsqueeze(1) + z                                    # residual
        mlp = torch.einsum("phd,pmd->pmh", M1, z).clamp(min=0)   # ReLU
        z = z + torch.einsum("pdh,pmh->pmd", M2, mlp)            # residual
        c_k = torch.einsum("pmd,pkd->pmk", z, E)                  # context read per arm
        # memory read
        We = torch.einsum("pmde,pmke->pmkd", W, Eexp.expand(Pn, M, K, d))
        v = torch.einsum("pmkd,pd->pmk", We, u)
        logits = beta.view(Pn, 1, 1) * v + gamma.view(Pn, 1, 1) * c_k
        gum = -torch.log(-torch.log(torch.rand(Pn, M, K, device=device) + 1e-12) + 1e-12)
        a = torch.argmax(logits + gum, dim=-1)
        r = (torch.rand(Pn, M, device=device) < torch.where(a == theta_true, 0.5 + eps, 0.5)).float()
        if theta_alt is None:
            o = r - osh.view(Pn, 1)
        else:
            r_alt = (torch.rand(Pn, M, device=device) < torch.where(a == theta_alt, 0.5 + eps, 0.5)).float()
            o = (1 - lam) * (r - osh.view(Pn, 1)) + lam * (r_alt - osh.view(Pn, 1))
        e_a = torch.gather(E.unsqueeze(1).expand(Pn, M, K, d), 2,
                           a.view(Pn, M, 1, 1).expand(Pn, M, 1, d)).squeeze(2)
        outer = torch.einsum("pd,pme->pmde", wv, e_a)
        W = W + cw.view(Pn, 1, 1, 1) * o.view(Pn, M, 1, 1) * outer - cd.view(Pn, 1, 1, 1) * W
        a_prev, r_prev, have = a, r, True
        if t >= tstart:
            acc = acc + (torch.argmax(logits, dim=-1) == theta_true).float().mean(dim=1)
            cnt += 1
    return acc / max(cnt, 1)


def es_train(spec, K, d, h, eps, N, M, L, pop, gens, device, seed=7, log=print):
    D = sum(ln for _, _, ln in spec)
    theta = init_theta(spec, K, d, h, device)
    sigma, lr = 0.08, 0.08          # smaller step for the larger model (stability)
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_theta = -1.0, theta.clone()
    for gi in range(gens):
        t0 = time.time()
        noise = torch.randn(pop // 2, D, device=device, generator=gen)
        eps_all = torch.cat([noise, -noise], 0)
        fits = rollout(theta.unsqueeze(0) + sigma * eps_all, spec, K, d, h, eps, N, M, L, 100 + gi, device)
        ranks = torch.empty_like(fits); ranks[fits.argsort()] = torch.arange(len(fits), device=device, dtype=fits.dtype)
        adv = ranks / (len(fits) - 1) - 0.5
        theta = theta + lr * (adv.unsqueeze(1) * eps_all).sum(0) / (pop * sigma)
        if gi % 5 == 0 or gi == gens - 1:
            ho = float(rollout(theta.unsqueeze(0), spec, K, d, h, eps, N, M, L, 9000 + gi, device)[0])
            if ho > best: best, best_theta = ho, theta.clone()
            log(f"[gen {gi:03d}] pop_mean={fits.mean():.3f} held-out={ho:.3f} best={best:.3f} ({time.time()-t0:.1f}s)")
    return best_theta, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--mlp-mult", type=int, default=2)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--phase1", action="store_true", help="run the full battery (gate + self-vs-wrong) after training")
    ap.add_argument("--save", default="champ_scale.npz")
    ap.add_argument("--out", default="results_scale_phase0.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h = args.d * args.mlp_mult
    spec = make_spec(args.K, args.d, h)
    D = sum(ln for _, _, ln in spec)
    chance = 1.0 / args.K
    ctx = bayes_map_accuracy(args.K, args.eps, args.L, 8000, 1)
    print(f"[SCALE p0] d={args.d} h={h} params={D} pop={args.pop} gens={args.gens} N={args.N} device={device}")
    print(f"           chance={chance:.3f}  context-only bound={ctx:.3f}")

    theta, best = es_train(spec, args.K, args.d, h, args.eps, args.N, args.M, args.L, args.pop, args.gens, device)
    np.savez(args.save, theta=theta.cpu().numpy())
    print(f"\n[TRAIN] best held-out {best:.3f} vs context-only {ctx:.3f} (chance {chance:.3f})  saved {args.save}")

    if not args.phase1:
        json.dump({"d": args.d, "params": D, "best": best, "context_only": ctx, "chance": chance,
                   "pop": args.pop, "gens": args.gens, "go": bool(best > ctx + 0.08)}, open(args.out, "w"), indent=2)
        print(f"saved {args.out}"); return

    # ---- Phase 1 battery on the trained policy ----
    th = theta.unsqueeze(0)
    ev = lambda **kw: float(rollout(th, spec, args.K, args.d, h, args.eps, args.N, 128, args.L, 555, device, **kw)[0])
    full = ev(write=True, use_ctx=True)
    mem_only = ev(write=True, use_ctx=False)
    ctx_only = ev(write=False, use_ctx=True)
    print(f"\n[DISSOCIATION] full {full:.3f} | memory-only {mem_only:.3f} (record removed) | "
          f"context-only {ctx_only:.3f} (bearer removed)")

    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    wrong, same = [], []
    print(f"\n[BEARER SWEEP] accuracy on OWN code vs lambda (chance {chance:.3f})")
    print(f"{'lambda':>7} {'self..wrong':>12} {'falsifier':>12}")
    for lam in lambdas:
        aw = ev(write=True, use_ctx=True, lam=lam, alt="indep")
        af = ev(write=True, use_ctx=True, lam=lam, alt="same")
        wrong.append(aw); same.append(af)
        print(f"{lam:>7.2f} {aw:>12.3f} {af:>12.3f}")

    gap = wrong[0] - wrong[-1]; isolate = same[-1] - wrong[-1]
    record_idle = abs(full - mem_only) < 0.07
    bearer_gate = (full - ctx_only) > 0.25
    monotone = all(wrong[i] >= wrong[i + 1] - 0.03 for i in range(len(wrong) - 1))
    bearer_effect = gap > 0.25 and isolate > 0.25
    print(f"\n[VERDICT] record idle {'YES' if record_idle else 'no'} ({full-mem_only:+.3f}) | "
          f"bearer gate {'YES' if bearer_gate else 'no'} ({full-ctx_only:+.3f}) | "
          f"self>>wrong {'YES' if bearer_effect else 'no'} (gap {gap:+.3f}, isolate {isolate:+.3f})")
    if record_idle and bearer_gate and bearer_effect and monotone:
        print(f"  PHASE 1 PASS: at {D} params the scaled transformer+memory model reproduces the whole "
              f"result. The record is idle, the incorporated memory is load-bearing, and it is "
              f"bearer-specific (self beats wrong-bearer, falsifier isolates). The Constitutive GPT "
              f"holds at scale.")
    else:
        print(f"  PHASE 1 PARTIAL: not all criteria met at this budget; see numbers above.")

    json.dump({"d": args.d, "params": D, "best": best, "full": full, "memory_only": mem_only,
               "context_only": ctx_only, "context_bound": ctx, "chance": chance,
               "lambdas": lambdas, "wrong": wrong, "falsifier": same, "gap": gap, "isolate": isolate,
               "record_idle": bool(record_idle), "bearer_gate": bool(bearer_gate),
               "bearer_effect": bool(bearer_effect)}, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
