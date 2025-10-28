# Flow Matching Tutorial


A hands-on tutorial on **Flow Matching** for generative modeling. Learn to build, train, and understand flow-based generative models through progressive Jupyter notebooks.

## What is Flow Matching?

Flow matching learns to transport samples from a simple source distribution (e.g., Gaussian noise) to a complex target distribution by following learned velocity fields. It provides a simple and efficient framework for generative modeling with strong connections to optimal transport theory.

---

## Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| **[Flow Matching Basics](notebooks/01_flow_matching_basics.ipynb)** | Introduction to flow matching with straight-line interpolation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/01_flow_matching_basics.ipynb) |
| **[Optimal Transport Pairing](notebooks/02_minibatch_ot.ipynb)** | Improve training efficiency with optimal transport couplings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/02_minibatch_ot.ipynb) |
| **[Latent Distribution Choice](notebooks/03_latent_choice.ipynb)** | Explore componentwise noise adaptation to target geometry | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/03_latent_choice.ipynb) |
| **[Conditional Sampling](notebooks/04_conditional_sampling.ipynb)** | Learn conditional generation with Y-penalized optimal transport | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/04_conditional_sampling.ipynb) |

---

## Getting Started

### Option 1: Google Colab (Recommended)

Click the badge above or open individual notebooks directly in Colab. Each notebook includes setup cells that install dependencies automatically.

### Option 2: Local Installation

```bash
git clone https://github.com/yourusername/tutorial_fm.git
cd tutorial_fm
pip install -r requirements.txt
jupyter notebook
```

---

## Lectures

The tutorial is accompanied by a series of lecture slides covering the theoretical foundations:

| Lecture | Topic | PDF |
|---------|-------|-----|
| **Lecture 1** | Optimal Transport | [1_HS_AMSTERDAM_OT.pdf](lectures/1_HS_AMSTERDAM_OT.pdf) |
| **Lecture 2** | Flow Matching | [2_HS_AMSTERDAM_FM.pdf](lectures/2_HS_AMSTERDAM_FM.pdf) |
| **Lecture 3** | 1D Processes and Noise Adaptation | [3_HS_AMSTERDAM_1D.pdf](lectures/3_HS_AMSTERDAM_1D.pdf) |
| **Lecture 4** | Bayesian Inference and Conditional Flow Matching | [4_HS_AMSTERDAM_Bayesian.pdf](lectures/4_HS_AMSTERDAM_Bayesian.pdf) |

These lectures provide the mathematical background and theory that complements the hands-on notebooks.

---

## References

### Primary Tutorial Sources

1. Flow Matching: Markov Kernels, Stochastic Processes and Transport Plans — Christian Wald, Gabriele Steidl, 2025 | [arXiv:2501.16839](https://arxiv.org/abs/2501.16839)
   - Used throughout the flow matching basics and optimal transport notebooks.
2. Adapting Noise to Data: Generative Flows from 1D Processes — Jannis Chemseddine, Gregor Kornhardt, Richard Duong, Gabriele Steidl, 2025 | [arXiv:2510.12636](https://arxiv.org/abs/2510.12636)
   - Used in the latent choice notebook.
3. Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching — Jannis Chemseddine, Paul Hagemann, Gabriele Steidl, Christian Wald, 2024 |[arXiv:2403.18705](https://arxiv.org/abs/2403.18705)
    - Used in the conditional sampling notebook.

### Related Work

- Flow Matching for Generative Modeling — Lipman et al., ICLR 2023 | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Building Normalizing Flows with Stochastic Interpolants — Albergo et al., ICLR 2023 | [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow — Liu et al., ICLR 2023 | [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Heavy-Tailed Diffusion Models — Pandey et al., 2024 | [arXiv:2410.14171](https://arxiv.org/abs/2410.14171)
   - Inspired notebook latent_choice.

### Libraries
- POT: Python Optimal Transport — Flamary et al., JMLR 2021 | [pythonot.github.io](https://pythonot.github.io/)
- GeomLoss: Geometric Losses for Deep Learning — Feydy et al. | [kernel-operations.io/geomloss](https://www.kernel-operations.io/geomloss/)

---


## Contact

Questions or suggestions? Write me an email: [chemseddine@math.tu-berlin.de](mailto:chemseddine@math.tu-berlin.de)
