#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def I_analytic(alpha: float, sigma_x: float, t: float) -> float:
    return 0.5 * np.log(1.0 + (alpha ** 2) * (sigma_x ** 2) / t)

def dIdalpha_analytic(alpha: float, sigma_x: float, t: float) -> float:
    return (alpha * (sigma_x ** 2)) / (t + (alpha ** 2) * (sigma_x ** 2))

def grad_estimate_vjp(alphas, sigma_x, t, N, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=sigma_x, size=N)
    Z = rng.normal(loc=0.0, scale=np.sqrt(t), size=N)
    grad_est = np.empty_like(alphas, dtype=float)
    for i, alpha in enumerate(alphas):
        Y = alpha * X + Z
        var_Y = (alpha ** 2) * (sigma_x ** 2) + t
        sY = - Y / var_Y
        grad_est[i] = - np.mean(X * sY)
    return grad_est

def main():
    sigma_x = 1.0
    t = 0.5
    alpha_min, alpha_max, num_alpha = 0.0, 3.0, 61
    N = 200_000
    seed = 42

    alphas = np.linspace(alpha_min, alpha_max, num_alpha)
    I_true = np.array([I_analytic(a, sigma_x, t) for a in alphas])
    dIda_true = np.array([dIdalpha_analytic(a, sigma_x, t) for a in alphas])

    dIda_est = grad_estimate_vjp(alphas, sigma_x, t, N=N, seed=seed)

    I_est = np.zeros_like(alphas)
    for k in range(1, len(alphas)):
        a0, a1 = alphas[k-1], alphas[k]
        g0, g1 = dIda_est[k-1], dIda_est[k]
        I_est[k] = I_est[k-1] + 0.5 * (g0 + g1) * (a1 - a0)

    plt.figure()
    plt.plot(alphas, I_true, label="Analytic I(α)")
    plt.plot(alphas, I_est, linestyle="--", marker="", label="Proposed (path-integral from ∂I/∂α)")
    plt.xlabel("α")
    plt.ylabel("I(X; Y_t) [nats]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("I_vs_alpha_linear_gaussian.pdf")

    plt.figure()
    plt.plot(alphas, dIda_true, label="Analytic ∂I/∂α")
    plt.plot(alphas, dIda_est, linestyle="--", marker="", label="Proposed score estimator")
    plt.xlabel("α")
    plt.ylabel("∂I/∂α [nats per unit α]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dIdalpha_vs_alpha_linear_gaussian.pdf")

    np.savez(
        "vi_a_1d_gaussian_results.npz",
        alphas=alphas,
        I_true=I_true,
        I_est=I_est,
        dIda_true=dIda_true,
        dIda_est=dIda_est,
        sigma_x=sigma_x,
        t=t,
        N=N,
        seed=seed,
    )

if __name__ == "__main__":
    main()
