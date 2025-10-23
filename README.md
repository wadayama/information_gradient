# Information Gradient for Nonlinear Gaussian Channels

Experimental code for "Information Gradient for Nonlinear Gaussian Channel with Applications to Task-Oriented Communication" by Tadashi Wadayama.

## Overview

A gradient-based framework for optimizing parametric nonlinear Gaussian channels via mutual information maximization using Score-to-Fisher Bridge (SFB) methodology.

## Repository Structure

```
experiments/
├── Gaussian_analytical/       # Analytical validation
├── Gaussian_linear/           # Linear vector channel with DSM
├── Gaussian_linear_per_alpha/ # Per-parameter DSM
├── A_optimization/            # Matrix optimization
├── tanh_optimization/         # Nonlinear channel
└── info_bottleneck/          # Information Bottleneck
```

## Key Formula

Information gradient:
```
grad_eta I(X; Y_t) = -E[D f_eta(X)^T s_Y_t(Y_t)]
```
where `s_Y_t(y)` is the score function learned via Denoising Score Matching (DSM).

## Requirements

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

Main dependencies:
- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, SciPy

## Quick Start

### 1. Validate with analytical solution
```bash
cd Gaussian_analytical
uv run python vi_a_1d_gaussian_experiment.py
```

### 2. Optimize linear channel
```bash
cd A_optimization
uv run python alphaA_optim_uncond_simple.py
```

### 3. Optimize nonlinear (tanh) channel
```bash
cd tanh_optimization
uv run python tanhA_optim_uncond_sfb.py
```

### 4. Information Bottleneck
```bash
cd info_bottleneck
uv run python ib_linear_gaussian_task_opt_dsm_both.py
```

## Citation

```bibtex
@article{wadayama2025information,
  title={Information Gradient for Nonlinear Gaussian Channel with Applications to Task-Oriented Communication},
  author={Wadayama, Tadashi},
  year={2025}
}
```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

Tadashi Wadayama - wadayama@nitech.ac.jp