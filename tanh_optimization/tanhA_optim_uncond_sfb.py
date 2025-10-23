
import math, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from numpy.linalg import eigh

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# =========================
# Config
# =========================
SEED = 321
DEVICE = "cpu"

# Problem setup
DIM = 12                # n = m = 12
SIGMA_X = 1.0           # X ~ N(0, σ_x^2 I)
T_NOISE = 0.5           # Channel noise variance t (fixed in optimization)
P_FROB = 5.0            # Frobenius constraint ||A||_F ≤ P

# Alternating optimization
OUTER_ITERS = 100       # number of iterations
STEP_SIZE = 0.2         # fixed step size (no backtracking)
EVAL_INTERVAL = 5       # evaluate MI every N iterations (for speed)

# DSM score (unconditional; input is y only), used for gradient at fixed T_NOISE
HIDDEN = 256
LR = 3e-3
BATCH = 4096
EPOCHS_PER_OUTER = 400
SIGMA_TRAIN_REL = 0.1   # σ_train = σ_eval = this * sqrt(T_NOISE)
GRAD_CLIP = 1.0
USE_STEIN_CALIB = True

# MC for gradient estimate
N_MC_GRAD = 50000

# KDE MI estimation parameters (per outer iteration)
KDE_SAMPLES = 6000                        # samples for KDE-based MI estimation
KDE_BANDWIDTH_GRID = [0.5, 0.7, 1.0, 1.4, 2.0]  # bandwidth scale grid
KDE_USE_WHITENING = True
KDE_USE_LEDOITWOLF = True

# Outputs
SAVE_PDF = "optim_MI_tanhA_vs_iter.pdf"
SAVE_NPZ = "optim_MI_tanhA_vs_iter.npz"

torch.manual_seed(SEED); np.random.seed(SEED)

# =========================
# Helpers
# =========================
def sample_X(n, device=DEVICE):
    return torch.randn(n, DIM, device=device) * SIGMA_X

def sample_Z(n, var, device=DEVICE):
    return torch.randn(n, DIM, device=device) * math.sqrt(var)

def forward_tanhA(A, X, Z):
    # returns (pre-activation U, noisy output Y)
    U = X @ A.T
    Y0 = torch.tanh(U)
    return U, Y0 + Z

def project_frobenius(A, P):
    fro = torch.linalg.norm(A).item()
    if fro <= P or fro == 0.0:
        return A
    return A * (P / fro)

def init_A_spectrum(dim, P, scale=0.8, seed=321):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    V, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    s = np.geomspace(3.0, 0.3, dim)
    A = (U * s) @ V.T
    fro = np.linalg.norm(A, "fro")
    if fro > 0:
        A = A * (scale * P / fro)
    return torch.tensor(A, dtype=torch.float32, device=DEVICE)

# =========================
# KDE-based MI estimation
# =========================
def _shrink_cov(Y, use_ledoitwolf=True):
    if use_ledoitwolf and _HAS_SK:
        lw = LedoitWolf().fit(Y)
        return lw.covariance_
    Yc = Y - Y.mean(axis=0, keepdims=True)
    return (Yc.T @ Yc) / Yc.shape[0]

def _inv_sqrt_from_cov(Sigma):
    vals, vecs = eigh(Sigma)
    eps = 1e-12
    vals = np.clip(vals, eps, None)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    logdetW = -0.5 * np.sum(np.log(vals))
    return inv_sqrt, logdetW

def _pairwise_squared_distances(X):
    norms = np.sum(X * X, axis=1, keepdims=True)
    D2 = norms + norms.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    return D2

def _scott_bandwidth_base(N, d):
    return N ** (-1.0 / (d + 4.0))

def _loo_log_probs_from_D2(D2, d, h):
    N = D2.shape[0]
    K = np.exp(-D2 / (2.0 * (h ** 2)))
    np.fill_diagonal(K, 0.0)
    S = K.sum(axis=1)
    c = -(d / 2.0) * np.log(2.0 * np.pi) - d * np.log(h)
    log_p = c + np.log(S) - np.log(N - 1)
    return log_p

def entropy_kde_whitened(Y, bandwidth_grid, whiten=True, use_ledoitwolf=True):
    N, d = Y.shape
    if whiten:
        mu = Y.mean(axis=0, keepdims=True)
        Yc = Y - mu
        Sigma = _shrink_cov(Yc, use_ledoitwolf)
        W, logdetW = _inv_sqrt_from_cov(Sigma)
        Yw = (Yc @ W.T)
    else:
        Yw = Y.copy()
        logdetW = 0.0

    D2 = _pairwise_squared_distances(Yw)
    h0 = _scott_bandwidth_base(N, d)

    best_ll = -np.inf
    best_h = None
    for scale in bandwidth_grid:
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

def mutual_info_kde_estimate(A):
    # Sample Y = tanh(AX) + Z
    with torch.no_grad():
        X = sample_X(KDE_SAMPLES)
        Z = sample_Z(KDE_SAMPLES, T_NOISE)
        U, Y = forward_tanhA(A, X, Z)
        Y_np = Y.cpu().numpy()

    # Estimate H(Y) via KDE
    H_hat, h_star = entropy_kde_whitened(
        Y_np,
        bandwidth_grid=KDE_BANDWIDTH_GRID,
        whiten=KDE_USE_WHITENING,
        use_ledoitwolf=KDE_USE_LEDOITWOLF
    )

    # H(Z) for Gaussian noise
    d = Y_np.shape[1]
    H_Z = 0.5 * d * np.log(2.0 * np.pi * np.e * T_NOISE)

    # I(X;Y) = H(Y) - H(Z)
    I_hat = H_hat - H_Z
    return float(I_hat)

def mutual_info_sfb_estimate(A):
    # Estimate I(X;Y_{t*}) for Y_u = tanh(A X) + Z_u via SFB:
    # I = 0.5 ∫_{t*}^{∞} ( J(Y_u) - m/u ) du  ≈  integral over [t*, t_max], tail≈0.
    t_star = T_NOISE
    t_max = t_star * SFB_T_MAX_SCALE
    u_grid = torch.logspace(math.log10(t_star), math.log10(t_max), SFB_M)  # [t*, ..., t_max]

    # Train per-u unconditional score; estimate Fisher J(Y_u) = E || s_u(Y_u) ||^2
    J_vals = []
    for u in u_grid:
        u_val = float(u.item())
        model = ScoreNet(dim=DIM, hidden=HIDDEN).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.99), weight_decay=1e-4)
        sigma_abs = SFB_SIGMA_TRAIN_REL * math.sqrt(u_val)

        # Per-u DSM training
        for ep in range(SFB_EPOCHS_PER_U):
            Xb = sample_X(SFB_BATCH); Zb = sample_Z(SFB_BATCH, u_val)
            Ub, Yb = forward_tanhA(A, Xb, Zb)
            loss = dsm_loss_single_sigma(model, Yb, sigma_abs)
            opt.zero_grad(); loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # Fisher estimation at u
        with torch.no_grad():
            Xe = sample_X(SFB_EVAL_SAMPLES); Ze = sample_Z(SFB_EVAL_SAMPLES, u_val)
            Ue, Ye = forward_tanhA(A, Xe, Ze)
            s_eval = model(Ye)
            if USE_STEIN_CALIB:
                c = stein_calibration_factor(Ye, s_eval)
                s_eval = s_eval * c
            J_hat = torch.mean(torch.sum(s_eval**2, dim=1)).item()
            J_vals.append(J_hat)

    J_vals = torch.tensor(J_vals)
    g_vals = 0.5 * (DIM / u_grid - J_vals) # de Bruijn / I–MMSE form

    # Log-domain trapezoid integration for ∫_{t*}^{t_max} g(u) du
    u_log = torch.log(u_grid); du = (u_log[1] - u_log[0]).item()
    integrand = g_vals * u_grid  # g(e^v) e^v
    area = 0.0
    for k in range(len(u_grid)-1):
        area += 0.5 * (integrand[k].item() + integrand[k+1].item()) * du

    I_hat = area  # tail ≈ 0
    return float(I_hat)

# =========================
# Score network & DSM
# =========================
class ScoreNet(nn.Module):
    def __init__(self, dim=DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, y):
        return self.net(y)

def dsm_loss_single_sigma(model, y_clean, sigma_abs):
    eps = torch.randn_like(y_clean) * sigma_abs
    y_tilde = y_clean + eps
    s_pred = model(y_tilde)
    target = -(y_tilde - y_clean) / (sigma_abs**2)
    loss = 0.5 * ((s_pred - target)**2).sum(dim=1).mean()
    return loss

def stein_calibration_factor(Y, s_pred):
    num = -Y.size(1)  # -m
    den = torch.mean(torch.sum(Y * s_pred, dim=1)).item()
    if abs(den) < 1e-12:
        return 1.0
    return num / den

# =========================
# Alternating optimization (fixed step)
# =========================
def main():
    # Initialize A
    A = init_A_spectrum(DIM, P_FROB)
    A = project_frobenius(A, P_FROB)

    iters, I_hist = [], []

    sigma_abs_grad = SIGMA_TRAIN_REL * math.sqrt(T_NOISE)

    for k in range(OUTER_ITERS):
        # ---- (1) Train unconditional score for gradient at fixed t = T_NOISE ----
        model = ScoreNet(dim=DIM, hidden=HIDDEN).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.99), weight_decay=1e-4)

        for ep in range(EPOCHS_PER_OUTER):
            Xb = sample_X(BATCH); Zb = sample_Z(BATCH, T_NOISE)
            Ub, Yb = forward_tanhA(A, Xb, Zb)
            loss = dsm_loss_single_sigma(model, Yb, sigma_abs_grad)
            opt.zero_grad(); loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # ---- (2) VJP gradient estimate & projected ascent (fixed step) ----
        with torch.no_grad():
            X = sample_X(N_MC_GRAD); Z = sample_Z(N_MC_GRAD, T_NOISE)
            U, Y = forward_tanhA(A, X, Z)
            s_eval = model(Y)
            if USE_STEIN_CALIB:
                c = stein_calibration_factor(Y, s_eval)
                s_eval = s_eval * c
            # Jacobian factor: diag(sech^2(U)) with sech^2 = 1 - tanh^2(U)
            sech2 = 1.0 - torch.tanh(U)**2
            # Gradient: ∇_A I ≈ - E[ (s(Y) ⊙ sech^2(U)) X^T ]
            G_hat = - ( (sech2 * s_eval).T @ X ) / N_MC_GRAD  # (m,n)
            A = project_frobenius(A + STEP_SIZE * G_hat, P_FROB)

        # ---- (3) Estimate MI via KDE at current A (every EVAL_INTERVAL iterations) ----
        if (k + 1) % EVAL_INTERVAL == 0 or k == 0 or k == OUTER_ITERS - 1:
            I_est = mutual_info_kde_estimate(A)
            iters.append(k); I_hist.append(I_est)
            print(f"[{k+1}/{OUTER_ITERS}] I_hat (KDE) = {I_est:.6f}")
        else:
            print(f"[{k+1}/{OUTER_ITERS}] (gradient update only, no MI eval)")

    # ---- Plot ----
    plt.figure()
    plt.plot(iters, I_hist, marker="o", label="Projected ascent (tanh(Ax); KDE MI)")
    plt.xlabel("Iteration"); plt.ylabel("Estimated mutual information  I(X;Y_{t^*})  [nats]")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(SAVE_PDF)

    np.savez(SAVE_NPZ, iters=np.array(iters), I_hist=np.array(I_hist),
             DIM=DIM, SIGMA_X=SIGMA_X, T_NOISE=T_NOISE, P_FROB=P_FROB,
             OUTER_ITERS=OUTER_ITERS, STEP_SIZE=STEP_SIZE,
             HIDDEN=HIDDEN, BATCH=BATCH, EPOCHS_PER_OUTER=EPOCHS_PER_OUTER,
             SIGMA_TRAIN_REL=SIGMA_TRAIN_REL, N_MC_GRAD=N_MC_GRAD,
             KDE_SAMPLES=KDE_SAMPLES)

if __name__ == "__main__":
    main()
