"""
CCA -- Batched-population GPU evaluator
=======================================

The port target named in es_outer_loop.py: replace the ProcessPool (one OS
process per genome) with ONE process that evaluates the whole population as a
batched tensor on the GPU.

Why this is the right shape for the hardware:
  * The CPU path runs P separate processes, each stepping a 256-dim net one
    observation at a time. Tiny kernels, no parallel work to amortize launch
    overhead, and P CUDA contexts if the GPU is visible (which is what OOMs a
    Windows box: each context reserves system RAM scaled to card VRAM).
  * Here every genome is a slot in a leading batch dimension. One process, one
    CUDA context, and each step is a handful of big batched matmuls instead of
    P tiny ones. That is work the GPU is actually built for.

Per-agent weights mean every layer is a bmm/einsum rather than a shared matmul:
    y[p,o] = sum_i (W[p,o,i] + Pfast[p,o,i]) * x[p,i]

Parameter layout is taken from a CPU PlasticAgent's named_parameters(), so a
theta vector is byte-compatible with the CPU champion.npz in both directions.

Faithfulness note: this is a reimplementation, not a bit-exact clone. Random
streams differ (per-agent GPU RNG vs per-process numpy), so individual
trajectories will not match the CPU path step for step. The claim under test is
distributional: same fitness scale and same ES learning curve. Check it with
    python gpu_batched.py --parity

Usage:
    python gpu_batched.py --benchmark          # GPU vs CPU throughput
    python gpu_batched.py --parity             # fitness-scale agreement
    python gpu_batched.py --train --generations 100 --pop 64 --lifetime 2000
"""

from __future__ import annotations

# Claim the GPU BEFORE importing cca_milestone1, whose CPU-only guard would
# otherwise hide the card from us (it uses setdefault, so winning the race is
# enough). Override by exporting CUDA_VISIBLE_DEVICES yourself.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from cca_milestone1 import (A_CONSUME, A_WAIT, Config, MOVES, N_ACTIONS,
                            PlasticAgent, band_dist)


# ----------------------------------------------------------------------------
# Parameter spec: the exact order nn.Module.parameters() yields on the CPU agent
# ----------------------------------------------------------------------------

def param_spec(cfg: Config, obs_dim: int):
    """(name, shape, numel) in CPU registration order -> champion-compatible."""
    torch.manual_seed(0)
    tmpl = PlasticAgent(cfg, obs_dim)
    return [(n, tuple(p.shape), p.numel()) for n, p in tmpl.named_parameters()]


def spec_dim(spec, scope: str) -> int:
    return sum(n for name, _, n in spec if _in_scope(name, scope))


def _in_scope(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    return name.startswith(("controller.", "film.", "bearer_cell."))


def unflatten_batch(theta: torch.Tensor, spec, scope: str, base: dict | None):
    """theta [P, D] -> {name: [P, *shape]}.  Out-of-scope params come from
    `base` (a shared template), broadcast across the population."""
    P = theta.shape[0]
    out, i = {}, 0
    for name, shape, n in spec:
        if _in_scope(name, scope):
            out[name] = theta[:, i:i + n].reshape(P, *shape)
            i += n
        else:
            out[name] = base[name].unsqueeze(0).expand(P, *shape).contiguous()
    assert i == theta.shape[1], f"theta size mismatch: consumed {i} of {theta.shape[1]}"
    return out


# ----------------------------------------------------------------------------
# Batched Private Constitution Forager
# ----------------------------------------------------------------------------

class BatchedForagerEnv:
    """P parallel foragers on GPU.

    Within one ES generation the CPU path gives every genome the SAME env seed
    (common random numbers), so world init and the constitution are shared and
    broadcast here; only the per-agent trajectories diverge.
    """

    def __init__(self, cfg: Config, P: int, seed: int, device):
        self.cfg, self.P, self.dev = cfg, P, device
        self.g = torch.Generator(device=device); self.g.manual_seed(seed)
        G, K, H = cfg.grid, cfg.K, cfg.H
        # Match the CPU path's two independent streams exactly: the constitution
        # is drawn from seed+555 (CPU's Constitution.draw rng), while base_value
        # and world placement come from seed (CPU's PCFEnv self.rng), in the same
        # draw order. This makes t=0 bit-identical to the CPU env, so any
        # GPU/CPU fitness gap is per-step stream divergence only, not a
        # different world.
        rng_c = np.random.default_rng(seed + 555)
        rng_w = np.random.default_rng(seed)

        # --- shared constitution (matches CPU Constitution.draw order) ---
        self.m_dev = self._t(rng_c.normal(0, cfg.const_sigma, K))
        self.v_dev = self._t(rng_c.normal(0, cfg.const_sigma, H))
        self.kappa = self._t(np.clip(rng_c.normal(0.002, 0.001, N_ACTIONS), 2e-4, 6e-3))
        self.gnoise = self._t(np.abs(rng_c.normal(0.1, 0.08, K)))
        self.alt_m = self.alt_v = self.alt_kappa = None   # wrong-bearer body

        # --- world from the seed stream (matches CPU PCFEnv order) ---
        self.base_value = self._t(rng_w.uniform(0.05, 0.15, K))
        self.base_damage = self._t(np.full(H, cfg.hazard_base_damage))
        res0 = np.zeros((G, G), dtype=np.int64); haz0 = np.zeros((G, G), dtype=np.int64)
        for _ in range(int(cfg.hazard_frac * G * G)):
            x, y = rng_w.integers(0, G, 2); haz0[x, y] = rng_w.integers(1, H + 1)
        for _ in range(int(0.10 * G * G)):
            x, y = rng_w.integers(0, G, 2)
            if haz0[x, y] == 0:
                res0[x, y] = rng_w.integers(1, K + 1)
        self.res = torch.as_tensor(res0, device=device).unsqueeze(0).repeat(P, 1, 1)
        self.haz = torch.as_tensor(haz0, device=device).unsqueeze(0).repeat(P, 1, 1)

        self.pos = torch.full((P, 2), G // 2, dtype=torch.long, device=device)
        self.E = torch.full((P,), 0.6, device=device)
        self.I = torch.ones(P, device=device)
        self.t = 0
        self.moves = torch.tensor(MOVES, dtype=torch.long, device=device)  # [8,2]
        # egocentric window offsets
        r = cfg.view // 2
        dy, dx = torch.meshgrid(torch.arange(-r, r + 1, device=device),
                                torch.arange(-r, r + 1, device=device), indexing="ij")
        self.off = torch.stack([dy.reshape(-1), dx.reshape(-1)], 0)   # [2, v*v]
        self.eps_prev = self._eps()

    def _t(self, a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.dev)

    # -- homeostasis ---------------------------------------------------------
    def _band(self, x, band):
        lo, hi = band
        return (lo - x).clamp(min=0) + (x - hi).clamp(min=0)

    def _eps(self):
        return self._band(self.E, self.cfg.band_E) + self._band(self.I, self.cfg.band_I)

    def _degradation(self):
        return (2.0 * self._eps()).clamp(max=1.0)

    @property
    def obs_dim(self) -> int:
        base = (self.cfg.K + self.cfg.H + 1) * self.cfg.view ** 2
        return base + (3 if self.cfg.observe_homeo else 0)

    def _flat(self, pos):
        return pos[:, 0] * self.cfg.grid + pos[:, 1]

    # -- observation ---------------------------------------------------------
    def obs(self):
        cfg, G, P = self.cfg, self.cfg.grid, self.P
        v2 = cfg.view ** 2
        C = cfg.K + cfg.H + 1
        wx = (self.pos[:, 0:1] + self.off[0].unsqueeze(0)) % G     # [P, v2]
        wy = (self.pos[:, 1:2] + self.off[1].unsqueeze(0)) % G
        idx = wx * G + wy
        rk = self.res.view(P, -1).gather(1, idx)                    # [P,v2] in 0..K
        hk = self.haz.view(P, -1).gather(1, idx)
        deg = self._degradation().unsqueeze(1)

        win = torch.zeros(P, C, v2, device=self.dev)
        # resources: plane k-1 holds clip(1 + N(0, g_k*(1+deg)), 0, 2)
        rmask = rk > 0
        scale = self.gnoise[(rk - 1).clamp(min=0)] * (1.0 + deg)
        noise = torch.randn(P, v2, device=self.dev, generator=self.g) * scale
        val = (1.0 + noise).clamp(0.0, 2.0) * rmask
        win.scatter_(1, (rk - 1).clamp(min=0).unsqueeze(1), val.unsqueeze(1))
        # hazards: plane K+h-1 = 1.0
        hmask = hk > 0
        win.scatter_(1, (cfg.K + (hk - 1).clamp(min=0)).unsqueeze(1),
                     hmask.float().unsqueeze(1))
        win[:, -1, v2 // 2] = 1.0        # self marker at window center
        flat = win.reshape(P, -1)
        if not cfg.observe_homeo:        # blind to own state; must learn it
            return flat
        homeo = torch.stack([self.E, self.I, self._eps()], 1)
        return torch.cat([flat, homeo], 1)

    # -- dynamics ------------------------------------------------------------
    def step(self, a: torch.Tensor):
        """a [P] long. Returns (obs, F_generic[P], V_self[P], eps[P], V_wrong[P]|None)."""
        cfg, G, P = self.cfg, self.cfg.grid, self.P
        self.t += 1
        E_pre, I_pre, eps_before = self.E.clone(), self.I.clone(), self.eps_prev
        deg = self._degradation()

        # out-of-band action slip is part of the EVENT (shared by counterfactual)
        slip = torch.rand(P, device=self.dev, generator=self.g) < 0.3 * deg
        a = torch.where(slip, torch.full_like(a, A_WAIT), a)

        F_gen = -self.kappa.mean().expand(P).clone()
        self.E = self.E - self.kappa[a]

        # move
        is_move = a < 8
        delta = torch.where(is_move.unsqueeze(1), self.moves[a.clamp(max=7)],
                            torch.zeros_like(self.pos))
        self.pos = (self.pos + delta) % G

        # consume
        flat = self._flat(self.pos)
        res_flat = self.res.view(P, -1)
        k_here = res_flat.gather(1, flat.unsqueeze(1)).squeeze(1)
        eating = (a == A_CONSUME) & (k_here > 0)
        ki = (k_here - 1).clamp(min=0)
        gain = self.base_value[ki] * (1.0 + self.m_dev[ki]) * eating
        self.E = (self.E + gain).clamp(0.0, 1.0)
        F_gen = F_gen + self.base_value[ki] * eating
        # per-step food, exposed for scoring on the two yardsticks:
        #   last_raw   = base_value           -> bearer-NEUTRAL (any body eats the
        #                                        same raw resource)
        #   last_taste = base_value*(1+m_dev) -> bearer-INDEXED (valued through
        #                                        THIS agent's own constitution)
        self.last_raw = self.base_value[ki] * eating.float()
        self.last_taste = gain.float()
        # obligate poison: eating own-poison damages integrity (uncompensable by
        # energy quantity), forcing bearer-specific avoidance. Own body m_dev.
        if cfg.obligate_poison > 0:
            poison = cfg.obligate_poison * (-self.m_dev[ki]).clamp(min=0) * eating
            self.I = (self.I - poison).clamp(0.0, 1.0)
        res_flat.scatter_(1, flat.unsqueeze(1),
                          torch.where(eating, torch.zeros_like(k_here),
                                      k_here).unsqueeze(1))

        # hazard
        h_here = self.haz.view(P, -1).gather(1, flat.unsqueeze(1)).squeeze(1)
        hit = h_here > 0
        hi = (h_here - 1).clamp(min=0)
        dmg = self.base_damage[hi] * (1.0 + self.v_dev[hi]) * hit
        self.I = (self.I - dmg).clamp(0.0, 1.0)
        F_gen = F_gen - self.base_damage[hi] * hit

        # passive
        self.E = (self.E - cfg.passive_decay).clamp(0.0, 1.0)
        self.I = torch.where(hit, self.I,
                             (self.I + cfg.integrity_regen).clamp(0.0, 1.0))

        self._respawn()
        if cfg.drift_every and self.t % cfg.drift_every == 0:
            self._drift()

        eps = self._eps()
        V_self = eps_before - eps
        self.eps_prev = eps

        # --- same-event / wrong-bearer counterfactual ---
        V_wrong = None
        if self.alt_m is not None:
            E_cf = E_pre - self.alt_kappa[a]
            E_cf = E_cf + self.base_value[ki] * (1.0 + self.alt_m[ki]) * eating
            E_cf = (E_cf - cfg.passive_decay).clamp(0.0, 1.0)
            I_cf = torch.where(
                hit, I_pre - self.base_damage[hi] * (1.0 + self.alt_v[hi]),
                I_pre + cfg.integrity_regen).clamp(0.0, 1.0)
            # obligate poison valued through the ALT body: a wrong-bearer learner
            # avoids ALT-poison, not own-poison, so it keeps eating own-poison.
            if cfg.obligate_poison > 0:
                poison_cf = cfg.obligate_poison * (-self.alt_m[ki]).clamp(min=0) * eating
                I_cf = (I_cf - poison_cf).clamp(0.0, 1.0)
            eps_cf = self._band(E_cf, cfg.band_E) + self._band(I_cf, cfg.band_I)
            V_wrong = eps_before - eps_cf
        return self.obs(), F_gen, V_self, eps, V_wrong

    def _respawn(self):
        cfg, G, P = self.cfg, self.cfg.grid, self.P
        res_flat = self.res.view(P, -1)
        for k in range(cfg.K):
            fire = torch.rand(P, device=self.dev, generator=self.g) < cfg.respawn_p
            if not bool(fire.any()):
                continue
            dest = torch.randint(0, G * G, (P,), device=self.dev, generator=self.g)
            cur = res_flat.gather(1, dest.unsqueeze(1)).squeeze(1)
            hz = self.haz.view(P, -1).gather(1, dest.unsqueeze(1)).squeeze(1)
            ok = fire & (cur == 0) & (hz == 0)
            newv = torch.where(ok, torch.full_like(cur, k + 1), cur)
            res_flat.scatter_(1, dest.unsqueeze(1), newv.unsqueeze(1))

    def _drift(self):
        cfg, G, P = self.cfg, self.cfg.grid, self.P
        c = cfg.drift_corr
        self.base_value = (c * self.base_value + (1 - c) * (
            torch.rand(cfg.K, device=self.dev, generator=self.g) * 0.10 + 0.05)
        ).clamp(0.01, 0.3)
        self.base_damage = (c * self.base_damage + (1 - c) * (
            torch.rand(cfg.H, device=self.dev, generator=self.g) * 0.12 + 0.02)
        ).clamp(0.01, 0.3)
        hz = self.haz.view(P, -1)
        moved = (hz > 0) & (torch.rand_like(hz, dtype=torch.float) < 0.2)
        types = hz * moved
        hz = hz * (~moved)
        dest = torch.randint(0, G * G, hz.shape, device=self.dev, generator=self.g)
        newh = torch.zeros_like(hz)
        newh.scatter_reduce_(1, dest, types, reduce="amax")   # 0s never clobber
        self.haz = torch.where(hz > 0, hz, newh).view(P, G, G)


# ----------------------------------------------------------------------------
# Batched agent: per-agent weights, per-agent fast weights
# ----------------------------------------------------------------------------

def _bmv(W, x):
    """W [P,o,i] @ x [P,i] -> [P,o]"""
    return torch.einsum("poi,pi->po", W, x)


class BatchedRunningNorm:
    """Per-agent windowed z-score, clipped to +/-3 (matches CPU RunningNorm)."""

    def __init__(self, P, window, device):
        self.buf = torch.zeros(P, window, device=device)
        self.W, self.ptr, self.n = window, 0, 0
        self.s = torch.zeros(P, device=device)
        self.ss = torch.zeros(P, device=device)

    def __call__(self, x):
        old = self.buf[:, self.ptr]
        if self.n == self.W:
            self.s -= old; self.ss -= old * old
        else:
            self.n += 1
        self.buf[:, self.ptr] = x
        self.s += x; self.ss += x * x
        self.ptr = (self.ptr + 1) % self.W
        mean = self.s / self.n
        var = (self.ss / self.n - mean * mean).clamp(min=0)
        return ((x - mean) / (var.sqrt() + 1e-6)).clamp(-3.0, 3.0)


class BatchedAgent:
    LAYERS = ("enc1", "enc2", "pi_head")     # the fast-weight layers

    def __init__(self, cfg: Config, params: dict, obs_dim: int, device):
        self.cfg, self.dev = cfg, device
        self.p = params
        P, d, bd = next(iter(params.values())).shape[0], cfg.d, cfg.bearer_dim
        self.P, self.d = P, d
        self.h = torch.zeros(P, d, device=device)
        self.b = torch.zeros(P, bd, device=device)
        self.onorm = BatchedRunningNorm(P, cfg.outcome_norm_window, device)
        self.step_count = 0
        self.last_dP = torch.zeros(P, device=device)
        shapes = {"enc1": (d, obs_dim), "enc2": (d, d), "pi_head": (N_ACTIONS, d)}
        self.P_fast = {k: torch.zeros(P, *s, device=device) for k, s in shapes.items()}
        self.elig = {k: torch.zeros(P, *s, device=device) for k, s in shapes.items()}
        self.last_outer = {k: torch.zeros(P, *s, device=device) for k, s in shapes.items()}
        self._cache = {}

    def _lin(self, name, x, fast=True):
        W = self.p[f"{name}.lin.weight"]
        b = self.p[f"{name}.lin.bias"]
        if fast:
            W = W + self.P_fast[name]
        y = _bmv(W, x) + b
        if fast:
            self._cache[name] = (x.detach(), y.detach())
        return y

    def _gru(self, prefix, x, h):
        Wi = self.p[f"{prefix}.weight_ih"]; Wh = self.p[f"{prefix}.weight_hh"]
        gi = _bmv(Wi, x) + self.p[f"{prefix}.bias_ih"]
        gh = _bmv(Wh, h) + self.p[f"{prefix}.bias_hh"]
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h

    @torch.no_grad()
    def act(self, obs):
        z = torch.tanh(self._lin("enc1", obs))
        z = torch.tanh(self._lin("enc2", z))
        film = _bmv(self.p["film.weight"], self.b) + self.p["film.bias"]
        gamma, beta = film.chunk(2, 1)
        z = z * (1.0 + gamma) + beta
        self.h = self._gru("gru", z, self.h)
        logits = self._lin("pi_head", self.h)
        a = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
        for name in self.LAYERS:                      # eligibility traces
            x, y = self._cache[name]
            outer = torch.einsum("po,pi->poi", y, x)
            nrm = outer.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-8
            norm_outer = outer / nrm
            self.last_outer[name] = norm_outer                # instantaneous term
            self.elig[name].mul_(self.cfg.fw_gamma_e).add_(norm_outer)
        return a

    @torch.no_grad()
    def plastic_update(self, o_raw, eps, deg):
        cfg = self.cfg
        o_n = self.onorm(o_raw)
        binp = torch.cat([self.h, torch.stack([o_n, eps, deg], 1)], 1)
        self.b = self._gru("bearer_cell", binp, self.b)

        e_norms = torch.stack([self.elig[k].flatten(1).norm(dim=1)
                               for k in self.LAYERS], 1)
        p_norms = torch.stack([self.P_fast[k].flatten(1).norm(dim=1)
                               for k in self.LAYERS], 1)
        zc = torch.cat([self.b, torch.stack([o_n, eps, deg], 1), e_norms, p_norms], 1)
        zc = torch.tanh(_bmv(self.p["controller.net.0.weight"], zc)
                        + self.p["controller.net.0.bias"])
        zc = torch.tanh(_bmv(self.p["controller.net.2.weight"], zc)
                        + self.p["controller.net.2.bias"])
        out = torch.sigmoid(_bmv(self.p["controller.net.4.weight"], zc)
                            + self.p["controller.net.4.bias"])
        n = len(self.LAYERS)
        learned = getattr(cfg, "learned_update", False)
        if learned:
            c_elig = out[:, 0:n] * cfg.eta_max
            c_inst = out[:, n:2 * n] * cfg.eta_max
            c_decay = cfg.delta_min + out[:, 2 * n:3 * n] * (cfg.delta_max - cfg.delta_min)
            c_omod = out[:, 3 * n:4 * n] * cfg.eta_max
        else:
            eta = out[:, :n] * cfg.eta_max
            delta = cfg.delta_min + out[:, n:] * (cfg.delta_max - cfg.delta_min)
        on3 = o_n.view(-1, 1, 1)

        dP = torch.zeros(self.P, device=self.dev)
        for i, name in enumerate(self.LAYERS):
            Pm = self.P_fast[name]
            if learned:
                # learned mix of a plasticity basis: eligibility-Hebbian +
                # instantaneous-Hebbian - decay - outcome-modulated forgetting.
                step = (c_elig[:, i].view(-1, 1, 1) * on3 * self.elig[name]
                        + c_inst[:, i].view(-1, 1, 1) * on3 * self.last_outer[name]
                        - c_decay[:, i].view(-1, 1, 1) * Pm
                        - c_omod[:, i].view(-1, 1, 1) * on3 * Pm)
            else:
                step = (eta[:, i].view(-1, 1, 1) * on3 * self.elig[name]
                        - delta[:, i].view(-1, 1, 1) * Pm)
            Pm.add_(step)
            # per-agent norm ceiling  ||P|| <= rho * ||W||
            wn = self.p[f"{name}.lin.weight"].flatten(1).norm(dim=1)
            pn = Pm.flatten(1).norm(dim=1)
            over = pn > cfg.fw_rho * wn
            if bool(over.any()):
                scale = torch.where(over, cfg.fw_rho * wn / (pn + 1e-12),
                                    torch.ones_like(pn))
                Pm.mul_(scale.view(-1, 1, 1))
            dP += step.flatten(1).norm(dim=1)
        self.last_dP = dP
        self.step_count += 1
        if cfg.project_every and self.step_count % cfg.project_every == 0:
            self._project_rank()

    @torch.no_grad()
    def _project_rank(self):
        r = self.cfg.fw_rank
        for name in self.LAYERS:
            Pm = self.P_fast[name]
            if min(Pm.shape[1:]) <= r:
                continue
            U, S, Vh = torch.linalg.svd(Pm, full_matrices=False)
            Pm.copy_(U[:, :, :r] @ torch.diag_embed(S[:, :r]) @ Vh[:, :r, :])


# ----------------------------------------------------------------------------
# Population evaluation
# ----------------------------------------------------------------------------

@torch.no_grad()
def eval_population(theta: torch.Tensor, cfg: Config, spec, scope, base,
                    lifetime: int, env_seed: int, device) -> torch.Tensor:
    """theta [P, D] -> fitness [P] = mean(-eps) over the last quarter."""
    P = theta.shape[0]
    env = BatchedForagerEnv(cfg, P, env_seed, device)
    params = unflatten_batch(theta, spec, scope, base)
    ag = BatchedAgent(cfg, params, env.obs_dim, device)
    obs = env.obs()
    tail_start = int(lifetime * 0.75)
    acc = torch.zeros(P, device=device)
    cnt = 0
    for t in range(lifetime):
        a = ag.act(obs)
        obs, F_gen, V_self, eps, _ = env.step(a)
        o = (1 - cfg.lambda_b) * F_gen + cfg.lambda_b * V_self
        ag.plastic_update(o, eps, env._degradation())
        if t >= tail_start:
            acc -= eps
            cnt += 1
    fit = acc / max(cnt, 1)
    # plasticity-death guard, same as the CPU path
    return torch.where(ag.last_dP <= 0, torch.full_like(fit, -1e6), fit)


def centered_ranks(x: torch.Tensor) -> torch.Tensor:
    ranks = torch.empty_like(x)
    ranks[x.argsort()] = torch.arange(len(x), device=x.device, dtype=x.dtype)
    return ranks / (len(x) - 1) - 0.5


def run_es_gpu(cfg: Config, generations: int, pop: int, lifetime: int,
               sigma: float, lr: float, scope: str, out_path: str,
               device, log_fn=print) -> np.ndarray:
    assert pop % 2 == 0, "pop must be even (antithetic pairs)"
    probe = BatchedForagerEnv(cfg, 1, 0, device)
    obs_dim = probe.obs_dim
    spec = param_spec(cfg, obs_dim)
    torch.manual_seed(cfg.seed)
    tmpl = PlasticAgent(cfg, obs_dim)
    base = {n: p.detach().to(device) for n, p in tmpl.named_parameters()}
    D = spec_dim(spec, scope)
    theta = torch.cat([base[n].reshape(-1) for n, _, _ in spec
                       if _in_scope(n, scope)]).clone()
    log_fn(f"[gpu-es] device={device} scope={scope} dim={D} pop={pop} "
           f"lifetime={lifetime}")

    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed + 99)
    best_fit, best_theta = -float("inf"), theta.clone()
    for g in range(generations):
        t0 = time.time()
        noise = torch.randn(pop // 2, D, device=device, generator=gen)
        eps_all = torch.cat([noise, -noise], 0)
        fits = eval_population(theta.unsqueeze(0) + sigma * eps_all, cfg, spec,
                               scope, base, lifetime, 10_000 + g, device)
        shaped = centered_ranks(fits)
        theta = theta + lr * (shaped.unsqueeze(1) * eps_all).sum(0) / (pop * sigma)
        gi = int(fits.argmax())
        if float(fits[gi]) > best_fit:
            best_fit = float(fits[gi])
            best_theta = theta.clone()
        np.savez(out_path, theta=best_theta.cpu().numpy().astype(np.float64),
                 scope=scope, env="forager", cfg=json.dumps(cfg.__dict__))
        log_fn(f"[gen {g:03d}] mean={fits.mean():+.4f} max={fits.max():+.4f} "
               f"best={best_fit:+.4f} ({time.time() - t0:.1f}s)")
    return best_theta.cpu().numpy()


# ----------------------------------------------------------------------------
# CLI: benchmark / parity / train
# ----------------------------------------------------------------------------

def _setup(cfg, device):
    probe = BatchedForagerEnv(cfg, 1, 0, device)
    spec = param_spec(cfg, probe.obs_dim)
    torch.manual_seed(cfg.seed)
    tmpl = PlasticAgent(cfg, probe.obs_dim)
    base = {n: p.detach().to(device) for n, p in tmpl.named_parameters()}
    return probe.obs_dim, spec, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--lifetime", type=int, default=2000)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--scope", choices=["meta", "all"], default="all")
    ap.add_argument("--out", default="champion_gpu.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available; running on CPU (check CUDA_VISIBLE_DEVICES)")
    else:
        print(f"[gpu] {torch.cuda.get_device_name(0)}")
    cfg = Config(seed=0, lambda_b=1.0, drift_every=0)

    if args.benchmark:
        obs_dim, spec, base = _setup(cfg, device)
        D = spec_dim(spec, "all")
        for pop in (8, 24, 64):
            theta = torch.cat([base[n].reshape(-1) for n, _, _ in spec]).unsqueeze(0) \
                .repeat(pop, 1) + 0.02 * torch.randn(pop, D, device=device)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()
            fit = eval_population(theta, cfg, spec, "all", base, 300, 1, device)
            torch.cuda.synchronize() if device.type == "cuda" else None
            dt = time.time() - t0
            print(f"  pop={pop:3d} lifetime=300 -> {dt:6.2f}s "
                  f"({pop * 300 / dt:8.0f} agent-steps/s)  fit_mean={fit.mean():+.4f}")
        return

    if args.parity:
        obs_dim, spec, base = _setup(cfg, device)
        D = spec_dim(spec, "all")
        theta = torch.cat([base[n].reshape(-1) for n, _, _ in spec]).unsqueeze(0) \
            .repeat(16, 1) + 0.02 * torch.randn(16, D, device=device)
        gfit = eval_population(theta, cfg, spec, "all", base, 300, 1, device)
        print(f"  GPU (16 genomes, lifetime 300): mean={gfit.mean():+.4f} "
              f"sd={gfit.std():+.4f} min={gfit.min():+.4f} max={gfit.max():+.4f}")
        from es_outer_loop import eval_genome
        from dataclasses import asdict
        cpu = [eval_genome((theta[i].cpu().numpy().astype(np.float32), asdict(cfg),
                            "forager", "all", 300, 1, cfg.seed)) for i in range(6)]
        cpu = np.array(cpu)
        print(f"  CPU (6 of the same genomes):     mean={cpu.mean():+.4f} "
              f"sd={cpu.std():+.4f} min={cpu.min():+.4f} max={cpu.max():+.4f}")
        print("  (streams differ, so compare scale/spread, not per-genome values)")
        return

    if args.train:
        run_es_gpu(cfg, args.generations, args.pop, args.lifetime, args.sigma,
                   args.lr, args.scope, args.out, device)
        print(f"champion -> {args.out}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
