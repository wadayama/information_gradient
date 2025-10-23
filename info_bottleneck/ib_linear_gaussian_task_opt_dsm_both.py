
# ib_linear_gaussian_task_opt_dsm_both.py
# -*- coding: utf-8 -*-
"""
Linear-Gaussian task-oriented IB with BOTH scores learned by DSM.
  - Channel: Y = A X + Z, Z ~ N(0, t I_m)
  - Task:    T = W X
  - Input:   X ~ N(0, Sigma_x)  (default Sigma_x = I_n)
  - Objective: L_IB(A) = I(T;Y) - beta * I(X;Y)
  - Constraint: ||A||_F <= P
  - Optimizer: projected gradient ascent on A using score-based estimator
        ∇_A L_IB ≈ E[ ( s_ψ(Y,T) + (beta-1) s_ϕ(Y) ) X^T ]
      where s_ϕ(y) ≈ ∇_y log p(y), s_ψ(y,t) ≈ ∇_y log p(y|t) are learned via DSM.
  - Eval per-iteration: exact closed-forms for I(T;Y), I(X;Y), and L_IB.
  - Optional: log MSE to the TRUE conditional score (for diagnostics only).

Usage (defaults suffice):
    python3 ib_linear_gaussian_task_opt_dsm_both.py

Outputs:
  out_dir/
    - history_plot.png   (iteration vs nats: L_IB, I(T;Y), I(X;Y))
    - history.json       (metrics and optional MSE traces)
    - matrices.pt        (A, W, Sigma_x, cfg)
"""

import math
import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------- Utilities ----------------------------------

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def fro_norm(A: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(A * A))


def project_to_fro_ball(A: torch.Tensor, P: float) -> torch.Tensor:
    nrm = fro_norm(A).item()
    if nrm <= P or P <= 0:
        return A
    return A * (P / nrm)


def symmetric_psd(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    M = 0.5 * (M + M.T)
    return M + eps * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)


def stable_inv(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    M = symmetric_psd(M, eps)
    return torch.linalg.inv(M)


def logdet_psd(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    M = symmetric_psd(M, eps)
    sign, logabsdet = torch.linalg.slogdet(M)
    return logabsdet


# -------------------- Closed-form MI for Gaussian blocks -------------------

def mi_XY_linear_gaussian(A: torch.Tensor, Sigma_x: torch.Tensor, noise_var: float) -> torch.Tensor:
    """
    I(X;Y) for Y = A X + Z, Z ~ N(0, noise_var I_m), X ~ N(0, Sigma_x).
    I(X;Y) = 0.5 * log det( I_m + (1/noise_var) A Sigma_x A^T )
    """
    m = A.shape[0]
    I_m = torch.eye(m, device=A.device, dtype=A.dtype)
    inside = I_m + (1.0 / noise_var) * (A @ Sigma_x @ A.T)
    return 0.5 * logdet_psd(inside)


def mi_TY_linear_gaussian(A: torch.Tensor, W: torch.Tensor, Sigma_x: torch.Tensor, noise_var: float) -> torch.Tensor:
    """
    I(T;Y) with T = W X, Y = A X + Z, X ~ N(0, Sigma_x), Z ~ N(0, noise_var I).
    I(T;Y) = 0.5*log det( I_k + Sigma_T^{-1} Sigma_TY Sigma_Y^{-1} Sigma_YT )
    """
    k = W.shape[0]
    I_k = torch.eye(k, device=A.device, dtype=A.dtype)

    Sigma_T = W @ Sigma_x @ W.T
    Sigma_Y = A @ Sigma_x @ A.T + noise_var * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    Sigma_TY = W @ Sigma_x @ A.T

    Sigma_T_inv = stable_inv(Sigma_T)
    Sigma_Y_inv = stable_inv(Sigma_Y)

    inside = I_k + Sigma_T_inv @ Sigma_TY @ Sigma_Y_inv @ Sigma_TY.T
    return 0.5 * logdet_psd(inside)


# -------- True conditional score (for optional diagnostics only) ----------

def true_conditional_score_Y_given_T(y: torch.Tensor,
                                     t: torch.Tensor,
                                     A: torch.Tensor,
                                     W: torch.Tensor,
                                     Sigma_x: torch.Tensor,
                                     noise_var: float) -> torch.Tensor:
    """
    Ground-truth s_{Y|T}(y|t) for linear-Gaussian model (used only for diagnostics).
    """
    device = y.device
    m = A.shape[0]

    Sigma_T = W @ Sigma_x @ W.T
    Sigma_T_inv = stable_inv(Sigma_T)

    # E[Y|T] = A Sigma_x W^T Sigma_T^{-1} t
    M = A @ Sigma_x @ W.T @ Sigma_T_inv
    mu = t @ M.T  # (B,m)

    middle = Sigma_x - Sigma_x @ W.T @ Sigma_T_inv @ W @ Sigma_x
    Sigma_Y_given_T = A @ middle @ A.T + noise_var * torch.eye(m, device=device, dtype=A.dtype)
    Sigma_Y_given_T_inv = stable_inv(Sigma_Y_given_T)

    diff = y - mu
    s = - diff @ Sigma_Y_given_T_inv.T
    return s


# --------------------------- DSM score networks ---------------------------

class MLPScore(nn.Module):
    """Unconditional score s_Y(y)."""
    def __init__(self, m: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        dims = [m] + [hidden] * (depth - 1) + [m]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
    def forward(self, y):
        return self.net(y)


class CondScore(nn.Module):
    """Conditional score s_{Y|T}(y, t). Input is concatenated [y; t]."""
    def __init__(self, m: int, k: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        dims = [m + k] + [hidden] * (depth - 1) + [m]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
    def forward(self, y, t):
        return self.net(torch.cat([y, t], dim=-1))


def dsm_loss_unconditional(score_net: nn.Module,
                           y: torch.Tensor,
                           sigma_dsm: float) -> torch.Tensor:
    """
    Basic DSM for unconditional score with fixed noise level.
    """
    eps = torch.randn_like(y)
    y_tilde = y + sigma_dsm * eps
    target = - (y_tilde - y) / (sigma_dsm ** 2)
    pred = score_net(y_tilde)
    return 0.5 * torch.mean((pred - target)**2)


def dsm_loss_conditional(score_net: nn.Module,
                         y: torch.Tensor,
                         t: torch.Tensor,
                         sigma_dsm: float) -> torch.Tensor:
    """
    Conditional DSM: perturb y only; t is kept clean and concatenated.
    """
    eps = torch.randn_like(y)
    y_tilde = y + sigma_dsm * eps
    target = - (y_tilde - y) / (sigma_dsm ** 2)
    pred = score_net(y_tilde, t)
    return 0.5 * torch.mean((pred - target)**2)


# ------------------------------- Sampling ---------------------------------

def sample_batch(Sigma_x: torch.Tensor,
                 A: torch.Tensor,
                 W: torch.Tensor,
                 noise_var: float,
                 batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw a batch: x ~ N(0, Sigma_x), t = W x, y = A x + z, z ~ N(0, noise_var I).
    """
    device = A.device
    n = Sigma_x.shape[0]
    m = A.shape[0]

    Lx = torch.linalg.cholesky(Sigma_x + 1e-8 * torch.eye(n, device=device, dtype=A.dtype))
    z = torch.randn(batch_size, n, device=device, dtype=A.dtype)
    x = z @ Lx.T
    t = x @ W.T
    y = x @ A.T + math.sqrt(noise_var) * torch.randn(batch_size, m, device=device, dtype=A.dtype)
    return x, t, y


# -------------------------- Training / Optimization -----------------------

@dataclass
class Config:
    n: int = 8        # dim(X)
    m: int = 8        # dim(Y)
    k: int = 4        # dim(T)
    noise_var: float = 0.5
    beta: float = 0.25
    P: float = 5.0                 # Frobenius norm limit for A
    steps: int = 20                # projected ascent iterations
    batch: int = 512
    dsm_steps_per_iter: int = 200  # DSM updates every iteration
    dsm_sigma: float = 0.1         # DSM perturbation sigma
    lr_A: float = 0.05             # ascent step size
    lr_dsm: float = 1e-3           # DSM optimizer LR
    hidden: int = 256              # score net width
    depth: int = 3                 # score net depth
    device: str = "cpu"
    seed: int = 42
    out_dir: str = "./outputs_ib"
    log_true_cond_score: bool = False   # if True, log MSE vs true s_{Y|T}


def run(cfg: Config) -> Dict[str, list]:
    device = torch.device(cfg.device)
    torch.set_default_dtype(torch.float64)
    set_seed(cfg.seed)

    n, m, k = cfg.n, cfg.m, cfg.k
    Sigma_x = torch.eye(n, device=device)

    # Initialize matrices
    A = 0.1 * torch.randn(m, n, device=device)
    A = project_to_fro_ball(A, cfg.P)
    W = torch.randn(k, n, device=device)

    # Score networks
    score_y = MLPScore(m=m, hidden=cfg.hidden, depth=cfg.depth).to(device)
    score_yt = CondScore(m=m, k=k, hidden=cfg.hidden, depth=cfg.depth).to(device)
    opt_y = torch.optim.Adam(score_y.parameters(), lr=cfg.lr_dsm)
    opt_yt = torch.optim.Adam(score_yt.parameters(), lr=cfg.lr_dsm)

    hist = {"iter": [], "I_TY": [], "I_XY": [], "L_IB": [], "froA": []}
    if cfg.log_true_cond_score:
        hist["mse_cond_score"] = []

    for it in range(cfg.steps):
        # 1) Train DSMs at current A (few steps)
        score_y.train(); score_yt.train()
        for _ in range(cfg.dsm_steps_per_iter):
            with torch.no_grad():
                x, t, y = sample_batch(Sigma_x, A, W, cfg.noise_var, cfg.batch)
            # unconditional
            loss_u = dsm_loss_unconditional(score_y, y, cfg.dsm_sigma)
            opt_y.zero_grad(set_to_none=True); loss_u.backward(); opt_y.step()
            # conditional
            loss_c = dsm_loss_conditional(score_yt, y, t, cfg.dsm_sigma)
            opt_yt.zero_grad(set_to_none=True); loss_c.backward(); opt_yt.step()

        # 2) Gradient estimate
        score_y.eval(); score_yt.eval()
        with torch.no_grad():
            x, t, y = sample_batch(Sigma_x, A, W, cfg.noise_var, cfg.batch)
            sY = score_y(y)             # learned unconditional
            sYgT = score_yt(y, t)       # learned conditional

            if cfg.log_true_cond_score:
                s_true = true_conditional_score_Y_given_T(y, t, A, W, Sigma_x, cfg.noise_var)
                mse = torch.mean((sYgT - s_true)**2).item()
                hist["mse_cond_score"].append(float(mse))

            G = torch.einsum("bm,bn->mn", (sYgT + (cfg.beta - 1.0) * sY), x) / x.shape[0]

        # 3) Ascent + projection
        A = A + cfg.lr_A * G
        A = project_to_fro_ball(A, cfg.P)

        # 4) Closed-form evaluation
        with torch.no_grad():
            I_XY = mi_XY_linear_gaussian(A, Sigma_x, cfg.noise_var)
            I_TY = mi_TY_linear_gaussian(A, W, Sigma_x, cfg.noise_var)
            L_IB = I_TY - cfg.beta * I_XY

        hist["iter"].append(it)
        hist["I_TY"].append(float(I_TY.item()))
        hist["I_XY"].append(float(I_XY.item()))
        hist["L_IB"].append(float(L_IB.item()))
        hist["froA"].append(float(fro_norm(A).item()))

        line = f"[{it:03d}] I(T;Y)={hist['I_TY'][-1]:.4f}, I(X;Y)={hist['I_XY'][-1]:.4f}, L_IB={hist['L_IB'][-1]:.4f}, ||A||_F={hist['froA'][-1]:.4f}"
        if cfg.log_true_cond_score:
            line += f", MSE_cond={hist['mse_cond_score'][-1]:.6f}"
        print(line)

    return {"A": A, "W": W, "Sigma_x": Sigma_x, "hist": hist, "cfg": cfg}


def plot_history(hist: Dict[str, list], out_pdf: str):
    it = np.array(hist["iter"])
    plt.figure()
    plt.plot(it, hist["L_IB"], marker='o', label="L_IB = I(T;Y) - beta I(X;Y)")
    plt.plot(it, hist["I_TY"], marker='s', label="I(T;Y)")
    plt.plot(it, hist["I_XY"], marker='^', label="I(X;Y)")
    plt.xlabel("Iteration")
    plt.ylabel("nats")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def save_history(hist: Dict[str, list], out_json: str):
    with open(out_json, "w") as f:
        json.dump(hist, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--noise_var", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--P", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--dsm_steps_per_iter", type=int, default=200)
    parser.add_argument("--dsm_sigma", type=float, default=0.1)
    parser.add_argument("--lr_A", type=float, default=0.05)
    parser.add_argument("--lr_dsm", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./outputs_ib_both_dsm")
    parser.add_argument("--log_true_cond_score", action="store_true")
    args = parser.parse_args()

    cfg = Config(n=args.n, m=args.m, k=args.k,
                 noise_var=args.noise_var, beta=args.beta, P=args.P,
                 steps=args.steps, batch=args.batch,
                 dsm_steps_per_iter=args.dsm_steps_per_iter, dsm_sigma=args.dsm_sigma,
                 lr_A=args.lr_A, lr_dsm=args.lr_dsm,
                 hidden=args.hidden, depth=args.depth,
                 device=args.device, seed=args.seed,
                 out_dir=args.out_dir, log_true_cond_score=args.log_true_cond_score)

    os.makedirs(cfg.out_dir, exist_ok=True)
    results = run(cfg)
    hist = results["hist"]

    out_pdf = os.path.join(cfg.out_dir, "history_plot.pdf")
    out_json = os.path.join(cfg.out_dir, "history.json")
    plot_history(hist, out_pdf)
    save_history(hist, out_json)

    torch.save({"A": results["A"], "W": results["W"], "Sigma_x": results["Sigma_x"], "cfg": cfg.__dict__},
               os.path.join(cfg.out_dir, "matrices.pt"))
    print(f"Saved: {out_pdf}, {out_json}, matrices.pt")


if __name__ == "__main__":
    main()
