# Faster MNIST CNN Embedding Power

This experiment trains a small CNN on the MNIST train split, freezes it, and evaluates
power on the MNIST test split only. It reuses the optimized MMMD computation pattern
from the faster kernel reproduction while adding:

- GPU-accelerated CNN training and embedding extraction
- Gaussian-only matched method comparisons
- Sigma condition-number diagnostics
- Automatic plotting after experiment completion

Outputs live under `Results/`.
