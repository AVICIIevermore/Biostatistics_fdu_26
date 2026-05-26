# MNIST Additive Noise Reproduction

This experiment reproduces the author MNIST additive-noise baseline on a tiny smoke-test scale.

## Scope

- Keeps the original comparison targets: kernel-based tests plus FR graph-based test.
- Keeps the original alternative: `{1,2,3}` vs `{1,2,8}`.
- Uses smoke-test parameters only to verify the pipeline runs end-to-end.

## Smoke-Test Configuration

- Noise levels: `0, 0.5, 1.0`
- Outer repetitions `n.rep`: `2`
- Inner Monte Carlo iterations `n.iter`: `20`
- Resample size: `30`
- CPU cores: `4`
- Alpha: `0.05`
- Seed: `20260521`

## Run

Kernel-based tests:

```bash
cd /home/dechao/kernel_two-sample/own_reproduction/mnist_additive_noise_reproduction/Code/Kernel_Based_Tests
conda run -n ktst-mnist Rscript Body.R
```

FR graph-based test:

```bash
cd /home/dechao/kernel_two-sample/own_reproduction/mnist_additive_noise_reproduction/Code/Graph_Based_Tests
conda run -n ktst-mnist Rscript Body.R
```

## Outputs

Results are written to `Results/` in this experiment directory.
