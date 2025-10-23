
import math, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt

# ---------------- Config ----------------
SEED = 123
DEVICE = "cpu"

# Problem
DIM = 8               # n = m = 8
SIGMA_X = 1.0         # X ~ N(0, σ_x^2 I)
NOISE_T = 0.5         # Z ~ N(0, t I)
P_FROB = 5.0          # Frobenius norm constraint on A, ||A||_F <= P
OUTER_ITERS = 30      # number of alternating iterations

# Score net & DSM (unconditional; input is y only)
SIGMA_TRAIN_REL = 0.1  # fixed σ_train = σ_eval = this * sqrt(t)
EPOCHS_PER_OUTER = 400  # training epochs per outer iteration
BATCH = 4096
LR = 3e-3
HIDDEN = 256
GRAD_CLIP = 1.0
USE_STEIN_CALIB = True

# Ascent: FIXED step size, NO backtracking
STEP_SIZE = 0.3       # tune as needed (smaller for smoother growth)

# MC for gradient estimation
N_MC = 50000          # samples per outer iteration for gradient estimate

# Plot
SAVE_PDF = "optim_I_vs_iter_simple.pdf"
SAVE_NPZ = "optim_I_vs_iter_simple.npz"

torch.manual_seed(SEED); np.random.seed(SEED)

# --------------- Helpers ----------------
def make_A_spectrum(dim, seed=SEED):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    V, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    s = np.geomspace(3.0, 0.3, dim)  # mildly ill-conditioned
    A = (U * s) @ V.T
    # scale to satisfy constraint with margin
    fro = np.linalg.norm(A, "fro")
    if fro > 0:
        A = A * (0.8 * P_FROB / fro)
    return torch.tensor(A, dtype=torch.float32, device=DEVICE)

def sample_X(n):
    return torch.randn(n, DIM, device=DEVICE) * SIGMA_X

def sample_Z(n):
    return torch.randn(n, DIM, device=DEVICE) * math.sqrt(NOISE_T)

def forward(A, X, Z):
    return X @ A.T + Z  # Y = A X + Z

def sigma_train_abs():
    return SIGMA_TRAIN_REL * math.sqrt(NOISE_T)

def mutual_info(A):
    # I = 1/2 log det( ΣY ) - 1/2 log det( t I ); ΣY = σ_x^2 A A^T + t I
    with torch.no_grad():
        AA = A @ A.T
        SigmaY = (SIGMA_X**2) * AA + NOISE_T * torch.eye(DIM, device=DEVICE)
        L = torch.linalg.cholesky(SigmaY)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        logdet_noise = DIM * math.log(NOISE_T)
        return 0.5 * (logdet - logdet_noise).item()

def project_frobenius(A, P):
    fro = torch.linalg.norm(A).item()
    if fro <= P or fro == 0.0:
        return A
    return A * (P / fro)

# --------------- Unconditional score net (y only) ---------------
class ScoreNet(nn.Module):
    def __init__(self, dim=DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, y_tilde):
        return self.net(y_tilde)

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

# --------------- Alternating optimization (simple) ---------------
def main():
    # Initialize A
    A = make_A_spectrum(DIM)
    A = project_frobenius(A, P_FROB)

    iters, I_hist = [], []
    I_star = 0.5 * DIM * math.log(1.0 + (SIGMA_X**2 / NOISE_T) * (P_FROB**2 / DIM))  # isotropic optimum
    sigma_abs = sigma_train_abs()

    # Common random numbers for smoother curves
    rng_grad = torch.Generator(device=DEVICE).manual_seed(SEED + 999)

    for k in range(OUTER_ITERS):
        # (1) Train unconditional score net on current A
        model = ScoreNet().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-4)

        for ep in range(EPOCHS_PER_OUTER):
            Xb = sample_X(BATCH)
            Zb = sample_Z(BATCH)
            Yb = forward(A, Xb, Zb)
            loss = dsm_loss_single_sigma(model, Yb, sigma_abs)
            opt.zero_grad(); loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # (2) Gradient estimate (VJP) & FIXED-STEP update + projection
        with torch.no_grad():
            X = torch.randn((N_MC, DIM), generator=rng_grad, device=DEVICE) * SIGMA_X
            Z = torch.randn((N_MC, DIM), generator=rng_grad, device=DEVICE) * math.sqrt(NOISE_T)
            Y = forward(A, X, Z)
            s_eval = model(Y)
            if USE_STEIN_CALIB:
                c = stein_calibration_factor(Y, s_eval)
                s_eval = s_eval * c

            G_hat = - (s_eval.T @ X) / N_MC  # ∇_A I ≈ -E[s(Y) X^T]
            A = project_frobenius(A + STEP_SIZE * G_hat, P_FROB)

        # (3) Log current MI
        I_curr = mutual_info(A)
        iters.append(k)
        I_hist.append(I_curr)
        print(f"[{k+1}/{OUTER_ITERS}] I(A) = {I_curr:.6f}  (I*={I_star:.6f})")

    # Plot
    plt.figure()
    plt.plot(iters, I_hist, marker="o", label="Projected ascent (VJP + DSM, unconditional)")
    plt.axhline(I_star, linestyle="--", label="Theoretical optimum (isotropic case)")
    plt.xlabel("Iteration")
    plt.ylabel("Mutual information  I(A)  [nats]")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(SAVE_PDF)
    np.savez(SAVE_NPZ, iters=np.array(iters), I_hist=np.array(I_hist), I_star=I_star,
             DIM=DIM, SIGMA_X=SIGMA_X, NOISE_T=NOISE_T, P_FROB=P_FROB,
             OUTER_ITERS=OUTER_ITERS, EPOCHS_PER_OUTER=EPOCHS_PER_OUTER, BATCH=BATCH,
             HIDDEN=HIDDEN, SIGMA_TRAIN_REL=SIGMA_TRAIN_REL, N_MC=N_MC, STEP_SIZE=STEP_SIZE)

if __name__ == "__main__":
    main()
