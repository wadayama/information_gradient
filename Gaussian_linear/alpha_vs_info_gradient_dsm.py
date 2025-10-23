
import math, numpy as np, torch, torch.nn as nn, torch.autograd as autograd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
SEED = 42
DIM  = 8             # dimension
SIGMA_X = 1.0        # X ~ N(0, σ_x^2 I)
NOISE_T = 0.5        # Z ~ N(0, t I)
ALPHA_MIN, ALPHA_MAX, NUM_ALPHA = 0.0, 3.0, 31
N_MC = 100_000       # MC samples per alpha for (2)(3)
TRAIN_DSM = True
EPOCHS = 1000        # increase for better DSM accuracy
BATCH = 4096
LR = 3e-3
HIDDEN = 256
DEVICE = 'cpu'

# DSM noise schedule
SIGMAS = np.geomspace(1.0, 0.05, 8)   # relative to sqrt(t); actual σ = s * sqrt(t)
SIGMA_EVAL_REL = 0.05                 # evaluate learned score at σ_eval = 0.05 * sqrt(t)

torch.manual_seed(SEED); np.random.seed(SEED)

# --------------- Helpers ----------------
def make_A(dim, seed=SEED):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    V, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    s = np.geomspace(3.0, 0.3, dim)  # singular values (tunable)
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
    # Y = α A X + Z
    return alpha * (X @ A.T) + Z

def true_score_y(alpha, A, y):
    # s_Y(y) = - Σ_Y^{-1} y, Σ_Y = α^2 σ_x^2 A A^T + t I
    with torch.no_grad():
        SigmaY = (alpha**2) * (SIGMA_X**2) * (A @ A.T) + NOISE_T * torch.eye(A.size(0), device=y.device)
        s = torch.linalg.solve(SigmaY, y.T).T
        return -s

# --------- DSM score network (conditional on α and σ) ---------
class ScoreNetDSM(nn.Module):
    def __init__(self, dim=DIM, hidden=HIDDEN):
        super().__init__()
        # input: y_tilde (dim) + alpha (1) + log_sigma (1) => dim+2
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, y_tilde, alpha, log_sigma):
        # Ensure alpha and log_sigma are properly shaped
        if alpha.dim() == 0:
            alpha = alpha.unsqueeze(0).expand(y_tilde.size(0))
        elif alpha.size(0) == 1:
            alpha = alpha.expand(y_tilde.size(0))

        if log_sigma.dim() == 0:
            log_sigma = log_sigma.unsqueeze(0).expand(y_tilde.size(0))
        elif log_sigma.size(0) == 1:
            log_sigma = log_sigma.expand(y_tilde.size(0))

        a = alpha.view(-1, 1)
        ls = log_sigma.view(-1, 1)
        return self.net(torch.cat([y_tilde, a, ls], dim=1))

def dsm_loss(model, y_clean, alpha, sigmas_rel):
    # Sample σ from schedule (per-batch), construct y_tilde = y + ε, target = -(y_tilde - y)/σ^2
    N = y_clean.size(0)
    idx = torch.randint(low=0, high=len(sigmas_rel), size=(N,), device=y_clean.device)
    sigmas = torch.tensor(sigmas_rel, device=y_clean.device, dtype=torch.float32)[idx] * math.sqrt(NOISE_T)   # absolute σ
    eps = torch.randn_like(y_clean) * sigmas.view(-1,1)
    y_tilde = y_clean + eps
    log_sigma = torch.log(sigmas + 1e-12)

    # Ensure alpha is float32
    if alpha.dtype != torch.float32:
        alpha = alpha.to(torch.float32)

    s_pred = model(y_tilde, alpha, log_sigma)
    target = -(y_tilde - y_clean) / (sigmas.view(-1,1)**2)
    loss = 0.5 * ((s_pred - target)**2).sum(dim=1).mean()
    return loss

# --------------- Main experiment ----------------
def main():
    A_np, svals = make_A(DIM)
    A = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)
    alphas = torch.linspace(ALPHA_MIN, ALPHA_MAX, NUM_ALPHA, device=DEVICE)

    # (1) Analytic
    g_true = torch.tensor([analytic_grad(float(a), svals) for a in alphas], device=DEVICE)

    # (2) VJP with true score
    g_vjp_true = []
    for a in alphas:
        X = sample_X(N_MC).to(DEVICE)
        Z = sample_noise(N_MC).to(DEVICE)
        Y = forward(a, A, X, Z)
        sY = true_score_y(a, A, Y)
        vjp = ((X @ A.T) * sY).sum(dim=1)
        grad_hat = - vjp.mean().item()
        g_vjp_true.append(grad_hat)
    g_vjp_true = torch.tensor(g_vjp_true, device=DEVICE)

    # (3) VJP with DSM-learned score
    g_vjp_dsm = None
    if TRAIN_DSM:
        model = ScoreNetDSM().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        # Train across mixed α
        alpha_train = alphas
        for ep in range(EPOCHS):
            a = alpha_train[torch.randint(len(alpha_train), (1,))]
            y_clean = forward(a, A, sample_X(BATCH).to(DEVICE), sample_noise(BATCH).to(DEVICE))
            loss = dsm_loss(model, y_clean, a.detach(), SIGMAS)
            opt.zero_grad(); loss.backward(); opt.step()
            if (ep+1) % max(1, EPOCHS//10) == 0:
                print(f"[{ep+1}/{EPOCHS}] DSM loss = {loss.item():.4f}")

        with torch.no_grad():
            g_vjp_dsm = []
            sigma_eval = SIGMA_EVAL_REL * math.sqrt(NOISE_T)
            log_sigma_eval = torch.tensor(math.log(sigma_eval), device=DEVICE)
            for a in alphas:
                X = sample_X(N_MC).to(DEVICE)
                Z = sample_noise(N_MC).to(DEVICE)
                Y = forward(a, A, X, Z)
                sY = model(Y, a, log_sigma_eval)  # approximate s_Y at small σ
                vjp = ((X @ A.T) * sY).sum(dim=1)
                grad_hat = - vjp.mean().item()
                g_vjp_dsm.append(grad_hat)
            g_vjp_dsm = torch.tensor(g_vjp_dsm, device=DEVICE)

    # --------- Plot: α vs ∂I/∂α ---------
    plt.figure()
    plt.plot(alphas.cpu(), g_true.cpu(), label="(1) Analytic")
    plt.plot(alphas.cpu(), g_vjp_true.cpu(), linestyle="--", label="(2) VJP w/ TRUE score")
    if g_vjp_dsm is not None:
        plt.plot(alphas.cpu(), g_vjp_dsm.cpu(), linestyle="-.", label="(3) VJP w/ DSM-learned score")
    plt.xlabel("α"); plt.ylabel("∂I/∂α  [nats per unit α]")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("alpha_vs_info_gradient_DSM.pdf")

    # Save raw data
    out = {"alphas": alphas.cpu().numpy(),
           "g_true": g_true.cpu().numpy(),
           "g_vjp_true": g_vjp_true.cpu().numpy(),
           "g_vjp_dsm": None if g_vjp_dsm is None else g_vjp_dsm.cpu().numpy(),
           "DIM": DIM, "SIGMA_X": SIGMA_X, "NOISE_T": NOISE_T, "N_MC": N_MC,
           "EPOCHS": EPOCHS, "BATCH": BATCH,
           "SIGMAS_rel": SIGMAS, "SIGMA_EVAL_REL": SIGMA_EVAL_REL}
    np.savez("alpha_vs_info_gradient_DSM.npz", **out)

if __name__ == "__main__":
    main()
