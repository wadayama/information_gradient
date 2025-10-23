
import math, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt

# ---------------- Config ----------------
SEED = 42
DIM  = 8             # n = m = 8
SIGMA_X = 1.0        # X ~ N(0, σ_x^2 I)
NOISE_T = 0.5        # Z ~ N(0, t I)
ALPHA_MIN, ALPHA_MAX, NUM_ALPHA = 0.0, 3.0, 25
ALPHAS_EVAL = None   # if None, use linspace above
N_MC = 100_000       # MC samples per alpha for (2)(3)

# DSM training (single fixed sigma; NO conditioning on sigma or alpha)
EPOCHS = 1000
BATCH = 4096
LR = 3e-3
HIDDEN = 256
SIGMA_TRAIN_REL = 0.1   # train at σ_train = this * sqrt(t)
USE_STEIN_CALIB = True   # Stein-based scale calibration at eval (optional)

DEVICE = 'cpu'
torch.manual_seed(SEED); np.random.seed(SEED)

# --------------- Helpers ----------------
def make_A(dim, seed=SEED):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    V, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    s = np.geomspace(3.0, 0.3, dim)
    return (U * s) @ V.T, s

def analytic_grad(alpha, svals, sigma_x=SIGMA_X, t=NOISE_T):
    num = alpha * (sigma_x**2) * (svals**2)
    den = t + (alpha**2) * (sigma_x**2) * (svals**2)
    return float(np.sum(num/den))

def sample_X(n):
    return torch.randn(n, DIM) * SIGMA_X

def sample_noise(n):
    return torch.randn(n, DIM) * math.sqrt(NOISE_T)

def forward(alpha, A, X, Z):
    return alpha * (X @ A.T) + Z

def true_score_y(alpha, A, y):
    with torch.no_grad():
        SigmaY = (alpha**2) * (SIGMA_X**2) * (A @ A.T) + NOISE_T * torch.eye(A.size(0), device=y.device)
        s = torch.linalg.solve(SigmaY, y.T).T
        return -s

# --------- Score network (DSM, NO conditioning on α or σ) ---------
class ScoreNetDSM_NoCond(nn.Module):
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

def stein_calibration_factor(y, s_pred):
    num = -y.size(1)  # -d
    den = torch.mean(torch.sum(y * s_pred, dim=1)).item()
    if abs(den) < 1e-12:
        return 1.0
    return num / den

# --------------- Main experiment ----------------
def main():
    A_np, svals = make_A(DIM)
    A = torch.tensor(A_np, dtype=torch.float32, device='cpu')

    if ALPHAS_EVAL is None:
        alphas = torch.linspace(ALPHA_MIN, ALPHA_MAX, NUM_ALPHA)
    else:
        alphas = torch.tensor(ALPHAS_EVAL, dtype=torch.float32)

    # (1) Analytic
    g_true = torch.tensor([analytic_grad(float(a), svals) for a in alphas])

    # (2) VJP with true score
    g_vjp_true = []
    for a in alphas:
        X = sample_X(N_MC)
        Z = sample_noise(N_MC)
        Y = forward(a, A, X, Z)
        sY = true_score_y(a, A, Y)
        vjp = ((X @ A.T) * sY).sum(dim=1)
        grad_hat = - vjp.mean().item()
        g_vjp_true.append(grad_hat)
    g_vjp_true = torch.tensor(g_vjp_true)

    # (3) Per-alpha DSM models (no α or σ conditioning; fixed σ_train = σ_eval)
    g_vjp_dsm = []
    sigma_abs = SIGMA_TRAIN_REL * math.sqrt(NOISE_T)

    for idx, a in enumerate(alphas):
        model = ScoreNetDSM_NoCond().to('cpu')
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-4)

        for ep in range(EPOCHS):
            y_clean = forward(a, A, sample_X(BATCH), sample_noise(BATCH))
            loss = dsm_loss_single_sigma(model, y_clean, sigma_abs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (ep+1) % max(1, EPOCHS//5) == 0:
                print(f"[α={float(a):.3f}] epoch {ep+1}/{EPOCHS}  DSM loss={loss.item():.4f}")

        with torch.no_grad():
            X = sample_X(N_MC)
            Z = sample_noise(N_MC)
            Y = forward(a, A, X, Z)
            s_eval = model(Y)

            if USE_STEIN_CALIB:
                c = stein_calibration_factor(Y, s_eval)
                s_eval = s_eval * c

            vjp = ((X @ A.T) * s_eval).sum(dim=1)
            grad_hat = - vjp.mean().item()
            g_vjp_dsm.append(grad_hat)

    g_vjp_dsm = torch.tensor(g_vjp_dsm)

    # Plot
    plt.figure()
    plt.plot(alphas.numpy(), g_true.numpy(), marker='o', label="(1) Analytic")
    plt.plot(alphas.numpy(), g_vjp_true.numpy(), marker='s', linestyle="--", label="(2) VJP w/ TRUE score")
    plt.plot(alphas.numpy(), g_vjp_dsm.numpy(), marker='^', linestyle="-.", label="(3) DSM (no α/σ conditioning)")
    plt.xlabel("α"); plt.ylabel("∂I/∂α  [nats per unit α]")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("alpha_vs_info_gradient_DSM_no_cond.pdf")

    # Save raw data
    out = {"alphas": alphas.numpy(),
           "g_true": g_true.numpy(),
           "g_vjp_true": g_vjp_true.numpy(),
           "g_vjp_dsm": g_vjp_dsm.numpy(),
           "DIM": DIM, "SIGMA_X": SIGMA_X, "NOISE_T": NOISE_T, "N_MC": N_MC,
           "EPOCHS": EPOCHS, "BATCH": BATCH,
           "SIGMA_TRAIN_REL": SIGMA_TRAIN_REL,
           "USE_STEIN_CALIB": USE_STEIN_CALIB}
    np.savez("alpha_vs_info_gradient_DSM_no_cond.npz", **out)

if __name__ == "__main__":
    main()
