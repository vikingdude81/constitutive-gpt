"""
CCA Milestone 1 -- Minimal Bearer-Coupling Experiment
=====================================================

Runnable skeleton implementing the Milestone 1 spec:

  * PCFEnv          -- Private Constitution Forager (32x32 torus, K resources,
                       H hazards, homeostasis, drift, lambda_b outcome dial)
  * Constitution    -- private (m, v, kappa, g), never observable
  * PlasticAgent    -- GRU backbone + bearer state (FiLM-injected) +
                       outcome-gated Hebbian fast weights + plasticity controller
  * run_pair()      -- self-coupled agent A vs yoked twin B (B updates on A's
                       normalized outcome stream); identical world seed & init
  * smoke test      -- short lifetime, prints homeostatic perf + plasticity norms

Design notes (map to spec sections):
  - Yoking is implemented at EXACTLY ONE line: the outcome value passed to
    plastic_update().  Everything else is identical between A and B.
  - Fast weights are stored as full matrices here for simplicity; rank-r
    projection is provided as an optional periodic step (project_rank).
    At d=256 full matrices are cheap; switch to factored A@B^T for scale.
  - W (base weights) training is left as a hook: meta-train with ES or PPO
    per the spec.  The plasticity loop below is the scientific core.
  - No resets, no death: out-of-band homeostasis degrades sensing/action.

Deps: torch, numpy.  CPU is fine at this scale.
"""

from __future__ import annotations

# Force CPU-only in EVERY process, including spawned ES workers. These nets are
# tiny, but on Windows/WDDM each per-worker CUDA context reserves system RAM up
# to the card's VRAM (an RTX 5090 = 32 GB each). A dozen workers then commit
# ~240 GB and freeze the machine. Set before importing torch so the spawn
# re-import in each worker inherits it. Override by exporting the var yourself.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class Config:
    # world
    grid: int = 32
    K: int = 6                 # resource types
    H: int = 2                 # hazard types
    view: int = 7              # egocentric window (odd)
    hazard_frac: float = 0.04
    respawn_p: float = 0.02    # per-type per-step spawn prob
    drift_every: int = 50_000
    drift_corr: float = 0.8
    # homeostasis
    band_E: tuple = (0.4, 0.8)
    band_I: tuple = (0.7, 1.0)
    passive_decay: float = 0.004
    integrity_regen: float = 0.001
    hazard_base_damage: float = 0.08
    # constitution draw
    const_sigma: float = 0.5
    # observation: when False, the homeostatic channels (E, I, eps) are dropped
    # from the ACTION observation, so the policy cannot regulate reactively and
    # must rely on the outcome-driven bearer/fast-weight pathway. E/I still feed
    # plastic_update, so the bearer manipulation's channel is intact. Honored by
    # the GPU BatchedForagerEnv.
    observe_homeo: bool = True
    # obligate selectivity: when > 0, eating a resource that is poison for THIS
    # body (negative m_dev) damages integrity by obligate_poison * (-m_dev),
    # which the energy channel cannot restore. Avoiding own-poison then requires
    # bearer-specific selectivity with no bearer-neutral substitute. Honored by
    # the GPU BatchedForagerEnv (real dynamics use own body; the wrong-bearer
    # counterfactual uses the alt body).
    obligate_poison: float = 0.0
    # outcome
    lambda_b: float = 1.0
    # agent
    d: int = 256
    bearer_dim: int = 256
    fw_gamma_e: float = 0.9    # eligibility decay
    fw_rank: int = 4           # target rank for optional projection
    fw_rho: float = 0.15       # ||P|| <= rho * ||W|| ceiling
    eta_max: float = 5e-3
    delta_min: float = 1e-5
    delta_max: float = 1e-3
    outcome_norm_window: int = 1000
    project_every: int = 500   # 0 = never project to low rank
    learned_update: bool = False  # Lever B: expand the plasticity controller from
                                  # 2 scalars/layer (eta, delta over a fixed Hebbian
                                  # rule) to 4 coefficients/layer that mix a basis of
                                  # plasticity terms (eligibility-Hebbian,
                                  # instantaneous-Hebbian, decay, outcome-modulated
                                  # forgetting), so ES can learn the update rule
                                  # itself, not just its gains (BEARER_OBSERVABILITY
                                  # lever 3). See RESULTS_LEVER_B.md.
    # run
    seed: int = 0
    device: str = "cpu"


N_ACTIONS = 10  # 8 moves + consume + wait
MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
A_CONSUME, A_WAIT = 8, 9


def band_dist(x: float, band: tuple) -> float:
    lo, hi = band
    return max(0.0, lo - x) + max(0.0, x - hi)


# ----------------------------------------------------------------------------
# Constitution -- private, fixed for life, learnable only through consequence
# ----------------------------------------------------------------------------

@dataclass
class Constitution:
    m_dev: np.ndarray      # metabolic deviation per resource, ~N(0, sigma)
    v_dev: np.ndarray      # hazard vulnerability multiplier deviation
    kappa: np.ndarray      # per-action energy cost (private)
    g: np.ndarray          # per-resource sensory noise scale

    @staticmethod
    def draw(cfg: Config, rng: np.random.Generator) -> "Constitution":
        return Constitution(
            m_dev=rng.normal(0.0, cfg.const_sigma, cfg.K),
            v_dev=rng.normal(0.0, cfg.const_sigma, cfg.H),
            kappa=np.clip(rng.normal(0.002, 0.001, N_ACTIONS), 0.0002, 0.006),
            g=np.abs(rng.normal(0.1, 0.08, cfg.K)),
        )


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------

class PCFEnv:
    """Private Constitution Forager.  One env instance per agent.

    step() returns (obs, o_generic_F, V_self, eps, info).  The lambda_b mix is
    applied by the caller so both mixed and component signals stay loggable.
    """

    def __init__(self, cfg: Config, constitution: Constitution, seed: int):
        self.cfg = cfg
        self.c = constitution
        self.alt_c: Constitution | None = None   # wrong-bearer counterfactual body
        self.rng = np.random.default_rng(seed)
        g = cfg.grid
        self.res = np.zeros((g, g), dtype=np.int64)          # 0 empty, 1..K
        self.haz = np.zeros((g, g), dtype=np.int64)          # 0 none, 1..H
        # population-mean profiles (drift targets); private deviations sit on top
        self.base_value = self.rng.uniform(0.05, 0.15, cfg.K)    # energy per unit
        self.base_damage = np.full(cfg.H, cfg.hazard_base_damage)
        self._seed_world()
        self.pos = np.array([g // 2, g // 2])
        self.E, self.I = 0.6, 1.0
        self.t = 0
        self.eps_prev = self._eps()

    # -- world construction -------------------------------------------------
    def _seed_world(self):
        g, cfg = self.cfg.grid, self.cfg
        n_haz = int(cfg.hazard_frac * g * g)
        for _ in range(n_haz):
            x, y = self.rng.integers(0, g, 2)
            self.haz[x, y] = self.rng.integers(1, cfg.H + 1)
        for _ in range(int(0.10 * g * g)):
            x, y = self.rng.integers(0, g, 2)
            if self.haz[x, y] == 0:
                self.res[x, y] = self.rng.integers(1, cfg.K + 1)

    def drift(self):
        """Re-draw population profiles with correlation drift_corr."""
        c = self.cfg.drift_corr
        self.base_value = np.clip(
            c * self.base_value
            + (1 - c) * self.rng.uniform(0.05, 0.15, self.cfg.K), 0.01, 0.3)
        self.base_damage = np.clip(
            c * self.base_damage
            + (1 - c) * self.rng.uniform(0.02, 0.14, self.cfg.H), 0.01, 0.3)
        # hazards relocate slowly
        idx = np.argwhere(self.haz > 0)
        for x, y in idx[self.rng.random(len(idx)) < 0.2]:
            self.haz[x, y] = 0
            nx, ny = self.rng.integers(0, self.cfg.grid, 2)
            self.haz[nx, ny] = self.rng.integers(1, self.cfg.H + 1)

    # -- internals ----------------------------------------------------------
    def _eps(self) -> float:
        return band_dist(self.E, self.cfg.band_E) + band_dist(self.I, self.cfg.band_I)

    def _degradation(self) -> float:
        """0 = in band; grows out of band.  Scales obs noise & action slip."""
        return min(1.0, 2.0 * self._eps())

    def obs(self) -> np.ndarray:
        cfg, g, v = self.cfg, self.cfg.grid, self.cfg.view
        r = v // 2
        C = cfg.K + cfg.H + 1                                # +1 wall/self plane unused->bias
        win = np.zeros((C, v, v), dtype=np.float32)
        deg = self._degradation()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = (self.pos[0] + dx) % g, (self.pos[1] + dy) % g
                k = self.res[x, y]
                if k > 0:
                    noise = self.rng.normal(0, self.c.g[k - 1] * (1 + deg))
                    win[k - 1, dx + r, dy + r] = np.clip(1.0 + noise, 0.0, 2.0)
                h = self.haz[x, y]
                if h > 0:
                    win[cfg.K + h - 1, dx + r, dy + r] = 1.0
        win[-1, r, r] = 1.0                                   # self marker
        homeo = np.array([self.E, self.I, self._eps()], dtype=np.float32)
        return np.concatenate([win.ravel(), homeo])

    @property
    def obs_dim(self) -> int:
        return (self.cfg.K + self.cfg.H + 1) * self.cfg.view ** 2 + 3

    # -- dynamics -----------------------------------------------------------
    def step(self, a: int):
        cfg, c, g = self.cfg, self.c, self.cfg.grid
        self.t += 1
        F_generic = 0.0
        E_pre, I_pre, eps_before = self.E, self.I, self.eps_prev
        consumed_k = 0                                        # 0 = nothing eaten

        # out-of-band action slip (part of the EVENT: shared by counterfactual)
        if self.rng.random() < 0.3 * self._degradation():
            a = A_WAIT

        # private + generic action cost
        self.E -= c.kappa[a]
        F_generic -= float(np.mean(c.kappa))                  # population-mean cost proxy

        if a < 8:
            self.pos = (self.pos + MOVES[a]) % g
        elif a == A_CONSUME:
            k = self.res[tuple(self.pos)]
            if k > 0:
                consumed_k = int(k)
                base = self.base_value[k - 1]
                gain_self = base * (1.0 + c.m_dev[k - 1])     # may be negative: "poison"
                self.E = float(np.clip(self.E + gain_self, 0.0, 1.0))
                F_generic += base                              # generic says: this was good
                self.res[tuple(self.pos)] = 0

        # hazards
        h = int(self.haz[tuple(self.pos)])
        if h > 0:
            dmg_self = self.base_damage[h - 1] * (1.0 + c.v_dev[h - 1])
            self.I = float(np.clip(self.I - dmg_self, 0.0, 1.0))
            F_generic -= self.base_damage[h - 1]

        # passive processes
        self.E = float(np.clip(self.E - cfg.passive_decay, 0.0, 1.0))
        if self.haz[tuple(self.pos)] == 0:
            self.I = float(np.clip(self.I + cfg.integrity_regen, 0.0, 1.0))

        # respawn
        for k in range(cfg.K):
            if self.rng.random() < cfg.respawn_p:
                x, y = self.rng.integers(0, g, 2)
                if self.res[x, y] == 0 and self.haz[x, y] == 0:
                    self.res[x, y] = k + 1

        if cfg.drift_every and self.t % cfg.drift_every == 0:
            self.drift()

        eps = self._eps()
        V_self = self.eps_prev - eps                          # homeostatic improvement
        self.eps_prev = eps

        # --- SAME-EVENT / WRONG-BEARER counterfactual ------------------------
        # Identical event facts (executed action, thing consumed, hazard hit,
        # starting condition); the consequence is re-evaluated through a
        # DIFFERENT private constitution.  This is the primary yoked signal:
        # the only manipulated variable is WHOSE BODY the event is valued for.
        V_wrong = None
        if self.alt_c is not None:
            c2 = self.alt_c
            E_cf = E_pre - c2.kappa[a]
            if consumed_k > 0:
                E_cf += self.base_value[consumed_k - 1] * (1.0 + c2.m_dev[consumed_k - 1])
            E_cf = float(np.clip(E_cf - cfg.passive_decay, 0.0, 1.0))
            I_cf = I_pre
            if h > 0:
                I_cf -= self.base_damage[h - 1] * (1.0 + c2.v_dev[h - 1])
            else:
                I_cf += cfg.integrity_regen
            I_cf = float(np.clip(I_cf, 0.0, 1.0))
            eps_cf = band_dist(E_cf, cfg.band_E) + band_dist(I_cf, cfg.band_I)
            V_wrong = eps_before - eps_cf

        return self.obs(), F_generic, V_self, eps, {"t": self.t,
                                                    "V_wrong": V_wrong}


# ----------------------------------------------------------------------------
# Fast-weight linear layer
# ----------------------------------------------------------------------------

class FastLinear(nn.Module):
    """y = x @ (W + P)^T + b, with eligibility trace e and Hebbian update.

    P and e are buffers (not autograd parameters): they are written by the
    neuromodulated update rule, not by backprop.  W is a normal parameter,
    trainable by the outer loop (ES/PPO) and frozen during deployment if the
    retrofit condition calls for it.
    """

    def __init__(self, d_in: int, d_out: int, cfg: Config):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.register_buffer("P", torch.zeros(d_out, d_in))
        self.register_buffer("e", torch.zeros(d_out, d_in))
        self.gamma_e = cfg.fw_gamma_e
        self.rho = cfg.fw_rho
        self.rank = cfg.fw_rank
        self._x = None
        self._y = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.lin.weight + self.P, self.lin.bias)
        self._x, self._y = x.detach(), y.detach()
        return y

    @torch.no_grad()
    def accumulate_trace(self):
        if self._x is None:
            return
        # rank-1 pre/post outer product per step; batch dim averaged out
        outer = torch.einsum("bo,bi->oi", self._y, self._x) / self._x.shape[0]
        outer = outer / (outer.norm() + 1e-8)                 # scale-free trace
        self.e.mul_(self.gamma_e).add_(outer)

    @torch.no_grad()
    def plastic_update(self, eta: float, delta: float, o_norm: float):
        """dP = eta * o_norm * e - delta * P   (the ONE line where outcomes
        enter the substrate; yoking substitutes the partner's o_norm here)."""
        self.P.add_(eta * o_norm * self.e - delta * self.P)
        # stability guard 1: norm ceiling
        w_norm = self.lin.weight.norm()
        p_norm = self.P.norm()
        if p_norm > self.rho * w_norm:
            self.P.mul_((self.rho * w_norm / p_norm).item())

    @torch.no_grad()
    def project_rank(self):
        """Optional: keep P near rank fw_rank (spec's low-rank constraint)."""
        U, S, Vh = torch.linalg.svd(self.P, full_matrices=False)
        r = self.rank
        self.P.copy_(U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :])


# ----------------------------------------------------------------------------
# Plasticity controller  (Theta_t)
# ----------------------------------------------------------------------------

class PlasticityController(nn.Module):
    def __init__(self, cfg: Config, n_layers: int):
        super().__init__()
        self.cfg = cfg
        self.n = n_layers
        in_dim = cfg.bearer_dim + 3 + 2 * n_layers   # b, (o, eps, deg), norms
        self.k = 4 if cfg.learned_update else 2       # coefficients per layer
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, self.k * n_layers),
        )

    def forward(self, b, o_norm, eps, deg, e_norms, p_norms):
        z = torch.cat([b,
                       torch.tensor([o_norm, eps, deg], dtype=torch.float32),
                       e_norms, p_norms])
        out = torch.sigmoid(self.net(z))
        n = self.n
        if self.k == 2:
            eta = out[:n] * self.cfg.eta_max
            delta = self.cfg.delta_min + out[n:] * (self.cfg.delta_max - self.cfg.delta_min)
            return eta, delta
        # learned rule: c_elig, c_inst (Hebbian gains), c_decay, c_omod (forgetting)
        c_elig = out[0:n] * self.cfg.eta_max
        c_inst = out[n:2 * n] * self.cfg.eta_max
        c_decay = self.cfg.delta_min + out[2 * n:3 * n] * (self.cfg.delta_max - self.cfg.delta_min)
        c_omod = out[3 * n:4 * n] * self.cfg.eta_max
        return c_elig, c_inst, c_decay, c_omod


# ----------------------------------------------------------------------------
# Running outcome normalizer
# ----------------------------------------------------------------------------

class RunningNorm:
    def __init__(self, window: int):
        self.buf: list[float] = []
        self.window = window

    def __call__(self, x: float) -> float:
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        mu = float(np.mean(self.buf))
        sd = float(np.std(self.buf)) + 1e-6
        return float(np.clip((x - mu) / sd, -3.0, 3.0))


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------

class PlasticAgent(nn.Module):
    """GRU backbone; encoder + policy head carry fast weights; bearer state
    gates the trunk via FiLM (conditional scale/shift), never as plain input
    tokens.  All continuant state lives on this module: (W, P, b, Theta, h)."""

    def __init__(self, cfg: Config, obs_dim: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.enc1 = FastLinear(obs_dim, d, cfg)
        self.enc2 = FastLinear(d, d, cfg)
        self.gru = nn.GRUCell(d, d)
        self.pi_head = FastLinear(d, N_ACTIONS, cfg)
        self.v_head = nn.Linear(d, 1)
        # bearer state machinery
        self.bearer_cell = nn.GRUCell(d + 3, cfg.bearer_dim)
        self.film = nn.Linear(cfg.bearer_dim, 2 * d)
        self.plastic_layers = [self.enc1, self.enc2, self.pi_head]
        self.controller = PlasticityController(cfg, len(self.plastic_layers))
        # continuant hidden states (never reset during a lifetime)
        self.h = torch.zeros(1, d)
        self.b = torch.zeros(1, cfg.bearer_dim)
        self.onorm = RunningNorm(cfg.outcome_norm_window)
        self.step_count = 0
        self.last_dP = 0.0

    # -- action -------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        """MUST stay under no_grad. self.h is a continuant state reassigned
        every step (h <- GRU(z, h)); with autograd on, each step chains a graph
        to the previous one and nothing is ever freed. That leaked 9 tensors
        (~0.37 MB) per agent per step: harmless over an 800-step ES lifetime,
        but ~19 GB per cell over a 25k-step sweep, and ~220 GB across 12
        workers, which OOMs the machine. Nothing here backprops (ES + Hebbian
        plasticity only), so no_grad is free."""
        x = torch.from_numpy(obs).float().unsqueeze(0)
        z = torch.tanh(self.enc1(x))
        z = torch.tanh(self.enc2(z))
        gamma, beta = self.film(self.b).chunk(2, dim=-1)      # bearer gates trunk
        z = z * (1.0 + gamma) + beta
        self.h = self.gru(z, self.h)
        logits = self.pi_head(self.h)
        a = torch.distributions.Categorical(logits=logits).sample().item()
        for l in self.plastic_layers:
            l.accumulate_trace()
        return a

    # -- the closed loop ----------------------------------------------------
    def plastic_update(self, o_raw: float, eps: float, deg: float):
        """o_raw is the (lambda_b-mixed) outcome.  For the YOKED twin, the
        caller passes the PARTNER's o_raw here.  Nothing else differs."""
        o_n = self.onorm(o_raw)
        with torch.no_grad():
            # bearer state update: b <- GRU(b; [h, o, eps, deg])
            binp = torch.cat([self.h,
                              torch.tensor([[o_n, eps, deg]], dtype=torch.float32)],
                             dim=-1)
            self.b = self.bearer_cell(binp, self.b)
            e_norms = torch.tensor([l.e.norm() for l in self.plastic_layers])
            p_norms = torch.tensor([l.P.norm() for l in self.plastic_layers])
            eta, delta = self.controller(self.b.squeeze(0), o_n, eps, deg,
                                         e_norms, p_norms)
        dP_total = 0.0
        for i, l in enumerate(self.plastic_layers):
            before = l.P.clone()
            l.plastic_update(eta[i].item(), delta[i].item(), o_n)
            dP_total += (l.P - before).norm().item()
        self.last_dP = dP_total                                # guard-2 monitor
        self.step_count += 1
        if self.cfg.project_every and self.step_count % self.cfg.project_every == 0:
            for l in self.plastic_layers:
                l.project_rank()

    # -- surgery hooks for the APHR battery ---------------------------------
    def export_history(self) -> dict:
        return {"P": [l.P.clone() for l in self.plastic_layers],
                "b": self.b.clone(), "h": self.h.clone()}

    def import_history(self, hist: dict):
        for l, P in zip(self.plastic_layers, hist["P"]):
            l.P.copy_(P)
        self.b = hist["b"].clone()
        self.h = hist["h"].clone()

    def ablate(self, what: str):
        if what in ("P", "all"):
            for l in self.plastic_layers:
                l.P.zero_()
        if what in ("b", "all"):
            self.b.zero_()


# ----------------------------------------------------------------------------
# Pair runner: self-coupled A vs yoked B
# ----------------------------------------------------------------------------

@dataclass
class PairLog:
    perf_A: list = field(default_factory=list)   # -eps (higher = better)
    perf_B: list = field(default_factory=list)
    dP_A: list = field(default_factory=list)     # plasticity-death monitor
    dP_B: list = field(default_factory=list)


def run_pair(cfg: Config, steps: int, mode: str = "wrong_bearer",
             log_every: int = 500, init_fn=None) -> PairLog:
    """A is always self-coupled.  B is the manipulated agent:

      mode="self"         B updates on its own bearer-relative outcome
                          (matched control).
      mode="wrong_bearer" PRIMARY yoked condition (same-event/wrong-bearer):
                          B lives its own life, but every outcome's bearer-
                          indexed component is re-valued through A's
                          constitution.  Same events, same information volume,
                          same machinery -- wrong body.
      mode="stream"       secondary/classic yoked control: B updates on A's
                          outcome stream (confounds wrong-bearer with
                          wrong-event; keep for comparison only).

    At lambda_b = 0 modes "self" and "wrong_bearer" are IDENTICAL by
    construction (the bearer-indexed component has zero weight), so the
    headline curve is pinned to Delta P(0) = 0 exactly.

    init_fn(agent), if given, is applied to both agents after construction
    (e.g. to load an ES champion via set_flat_params)."""
    assert mode in ("self", "wrong_bearer", "stream")
    rngA = np.random.default_rng(cfg.seed + 1)
    rngB = np.random.default_rng(cfg.seed + 2)
    envA = PCFEnv(cfg, Constitution.draw(cfg, rngA), seed=cfg.seed)
    envB = PCFEnv(cfg, Constitution.draw(cfg, rngB), seed=cfg.seed)
    if mode == "wrong_bearer":
        envB.alt_c = envA.c                        # value B's events with A's body

    torch.manual_seed(cfg.seed)                    # identical init for A...
    A = PlasticAgent(cfg, envA.obs_dim)
    torch.manual_seed(cfg.seed)                    # ...and B
    B = PlasticAgent(cfg, envB.obs_dim)
    if init_fn is not None:
        init_fn(A)
        init_fn(B)

    obsA, obsB = envA.obs(), envB.obs()
    log = PairLog()
    lam = cfg.lambda_b

    for t in range(steps):
        aA = A.act(obsA)
        aB = B.act(obsB)
        obsA, F_A, V_A, epsA, infoA = envA.step(aA)
        obsB, F_B, V_B, epsB, infoB = envB.step(aB)
        oA = (1 - lam) * F_A + lam * V_A

        # >>> the yoking lines: bearer-indexed component's OWNER is the only
        #     manipulated variable
        if mode == "self":
            oB_upd = (1 - lam) * F_B + lam * V_B
        elif mode == "wrong_bearer":
            oB_upd = (1 - lam) * F_B + lam * infoB["V_wrong"]
        else:  # stream
            oB_upd = oA

        A.plastic_update(oA, epsA, envA._degradation())
        B.plastic_update(oB_upd, epsB, envB._degradation())

        if t % log_every == 0:
            log.perf_A.append(-epsA)
            log.perf_B.append(-epsB)
            log.dP_A.append(A.last_dP)
            log.dP_B.append(B.last_dP)
    return log


# ----------------------------------------------------------------------------
# Outer-loop hooks (spec section 3) -- stubs
# ----------------------------------------------------------------------------

def es_fitness(cfg: Config, controller_vec: torch.Tensor, lifetime: int = 2000) -> float:
    """Hook: load controller_vec into agent.controller (+ film/bearer_cell),
    run a short lifetime, return mean(-eps) over the last quarter.  Wrap this
    in your ES loop (pop 64, sigma 0.02).  Left as an exercise wiring step."""
    raise NotImplementedError


def ppo_update_W(agent: PlasticAgent, rollout) -> None:
    """Hook: grown-in condition trains W concurrently with standard PPO on
    the mixed outcome as reward.  Keep fast-weight buffers out of the
    optimizer (they already are: buffers, not parameters)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Config(seed=0, lambda_b=1.0, drift_every=0)  # drift off for smoke
    print("smoke test: 2000 steps, lambda_b=1.0, same-event/wrong-bearer pair")
    log = run_pair(cfg, steps=2000, mode="wrong_bearer", log_every=250)
    print(f"  perf A (self) : {np.mean(log.perf_A):+.4f}")
    print(f"  perf B (yoked): {np.mean(log.perf_B):+.4f}")
    print(f"  |dP| A last   : {log.dP_A[-1]:.5f}  (should be > 0: plasticity alive)")
    print(f"  |dP| B last   : {log.dP_B[-1]:.5f}")

    # quick surgery sanity: history exchange should perturb behavior state
    # (full R-metric needs developed agents; this just checks the plumbing)
    cfgB = Config(seed=1, lambda_b=1.0, drift_every=0)
    print("plumbing check: export/import/ablate history hooks ...", end=" ")
    env = PCFEnv(cfgB, Constitution.draw(cfgB, np.random.default_rng(7)), seed=7)
    ag = PlasticAgent(cfgB, env.obs_dim)
    o = env.obs()
    for _ in range(50):
        a = ag.act(o)
        o, Fg, Vs, eps, _ = env.step(a)
        ag.plastic_update((1 - cfgB.lambda_b) * Fg + cfgB.lambda_b * Vs,
                          eps, env._degradation())
    h = ag.export_history()
    ag.ablate("all")
    assert all(P.norm() == 0 for P in [l.P for l in ag.plastic_layers])
    ag.import_history(h)
    assert any(l.P.norm() > 0 for l in ag.plastic_layers)
    print("ok")
