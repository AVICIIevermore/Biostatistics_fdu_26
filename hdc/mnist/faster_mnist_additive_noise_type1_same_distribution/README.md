# MNIST Additive Noise Type-I Error

This experiment adds null-distribution checks that are missing from the original MNIST additive-noise setup.

## Null Scenarios

- `{1,2,3}` vs `{1,2,3}`
- `{1,2,8}` vs `{1,2,8}`

For each scenario, both groups are drawn independently from the same digit pool and perturbed with the same noise level using independent Gaussian noise realizations.

## Smoke-Test Configuration

- Noise levels: `0, 0.5, 1.0`
- Outer repetitions `n.rep`: `2`
- Inner Monte Carlo iterations `n.iter`: `20`
- Resample size: `30`
- CPU cores: `4`
- Alpha: `0.05`
- Seeds: `20260522`, `20260523`

## Run

Kernel-based tests:

```bash
cd /home/dechao/kernel_two-sample/own_reproduction/mnist_additive_noise_type1_same_distribution/Code/Kernel_Based_Tests
conda run -n ktst-mnist Rscript Body.R
```

FR graph-based test:

```bash
cd /home/dechao/kernel_two-sample/own_reproduction/mnist_additive_noise_type1_same_distribution/Code/Graph_Based_Tests
conda run -n ktst-mnist Rscript Body.R
```

## Outputs

Results are written separately to:

- `Results/set123_vs_set123/`
- `Results/set128_vs_set128/`
