# tanhA_mi_kde.py
# Estimate MI I(X;Y) for Y = tanh(A X) + Z via KDE (with whitening + LOO).
# Focused on MI *evaluation only* (no gradients/optimization). Default n=4.

from __future__ import annotations
import numpy as np
from numpy.linalg import eigh
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SK = True
except Exception:
    _HAS_SK = False


@dataclass
class KDEConfig:
    n_dim: int = 4
    bandwidth_scale_grid: Sequence[float] = (0.5, 0.7, 1.0, 1.4, 2.0)
    max_samples_for_kde: int = 6000
    seed: Optional[int] = 1234
    whiten: bool = True
    use_ledoitwolf: bool = True
    verbose: bool = True


def _shrink_cov(Y: np.ndarray, use_ledoitwolf: bool = True) -> np.ndarray:
    if use_ledoitwolf and _HAS_SK:
        lw = LedoitWolf().fit(Y)
        return lw.covariance_
    Yc = Y - Y.mean(axis=0, keepdims=True)
    return (Yc.T @ Yc) / Yc.shape[0]


def _inv_sqrt_from_cov(Sigma: np.ndarray) -> Tuple[np.ndarray, float]:
    vals, vecs = eigh(Sigma)
    eps = 1e-12
    vals = np.clip(vals, eps, None)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    logdetW = -0.5 * np.sum(np.log(vals))
    return inv_sqrt, logdetW


def _pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X * X, axis=1, keepdims=True)
    D2 = norms + norms.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    return D2


def _scott_bandwidth_base(N: int, d: int) -> float:
    return N ** (-1.0 / (d + 4.0))


def _loo_log_probs_from_D2(D2: np.ndarray, d: int, h: float) -> np.ndarray:
    N = D2.shape[0]
    K = np.exp(-D2 / (2.0 * (h ** 2)))
    np.fill_diagonal(K, 0.0)
    S = K.sum(axis=1)
    c = -(d / 2.0) * np.log(2.0 * np.pi) - d * np.log(h)
    log_p = c + np.log(S) - np.log(N - 1)
    return log_p


def entropy_kde_whitened(Y: np.ndarray, cfg: KDEConfig) -> Tuple[float, float]:
    N, d = Y.shape
    if cfg.whiten:
        mu = Y.mean(axis=0, keepdims=True)
        Yc = Y - mu
        Sigma = _shrink_cov(Yc, cfg.use_ledoitwolf)
        W, logdetW = _inv_sqrt_from_cov(Sigma)
        Yw = (Yc @ W.T)
    else:
        Yw = Y.copy()
        logdetW = 0.0

    if N > cfg.max_samples_for_kde:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(N, size=cfg.max_samples_for_kde, replace=False)
        Yw_eval = Yw[idx]
    else:
        Yw_eval = Yw

    N_eval = Yw_eval.shape[0]
    D2 = _pairwise_squared_distances(Yw_eval)
    h0 = _scott_bandwidth_base(N_eval, d)

    best_ll = -np.inf
    best_h = None
    for scale in cfg.bandwidth_scale_grid:
        h = h0 * float(scale)
        log_p = _loo_log_probs_from_D2(D2, d, h)
        ll = np.mean(log_p)
        if ll > best_ll:
            best_ll = ll
            best_h = h

    log_p_best = _loo_log_probs_from_D2(D2, d, best_h)
    H_tilde = -np.mean(log_p_best)
    H_hat = H_tilde - logdetW
    return float(H_hat), float(best_h)


def estimate_mi_from_samples(Y: np.ndarray, noise_variance_t: float, cfg: Optional[KDEConfig] = None) -> Tuple[float, float, float]:
    if cfg is None:
        cfg = KDEConfig()
    d = Y.shape[1]
    H_hat, h_star = entropy_kde_whitened(Y, cfg)
    H_Z = 0.5 * d * np.log(2.0 * np.pi * np.e * noise_variance_t)
    I_hat = H_hat - H_Z
    return float(I_hat), float(H_hat), float(h_star)


def sample_Y_tanhAX(A: np.ndarray, N: int, t: float, seed: Optional[int] = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    X = rng.normal(size=(N, n))
    Z = rng.normal(scale=np.sqrt(t), size=(N, n))
    Y = np.tanh(X @ A.T) + Z
    return Y


def make_A_with_frobenius(n: int, fro_norm: float, seed: Optional[int] = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    current = np.linalg.norm(A, ord="fro")
    if current == 0.0:
        return np.eye(n) * (fro_norm / np.sqrt(n))
    return A * (fro_norm / current)


if __name__ == "__main__":
    cfg = KDEConfig(n_dim=4, verbose=True)
    n = cfg.n_dim
    N = 6000
    t = 0.1
    fro_A = 2.0

    A = make_A_with_frobenius(n, fro_A, seed=cfg.seed)
    if cfg.verbose:
        print(f"[demo] n={n}, N={N}, t={t}")
        print(f"[demo] Frobenius(A)={np.linalg.norm(A,'fro'):.4f}")

    Y = sample_Y_tanhAX(A, N, t, seed=cfg.seed)
    I_hat, H_hat, h_star = estimate_mi_from_samples(Y, t, cfg=cfg)
    if cfg.verbose:
        print(f"[KDE-MI] H(Y) ≈ {H_hat:.6f} nats")
        print(f"[KDE-MI] I(X;Y) ≈ {I_hat:.6f} nats  (bandwidth h* = {h_star:.4f})")
