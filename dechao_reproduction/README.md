# Dechao reproduction experiments

This directory is organized around three experiment families:

- `mnist/`: original MNIST additive-noise reproductions, faster MNIST reruns, CNN embedding experiments, shared CNN model metadata, and MNIST-specific plans.
- `medmnist_pathmnist28/`: PathMNIST 28x28 class-mixture MMMD experiments, including CNN training metadata, smoke tests, power, and Type-I checks.
- `pathmnist_center_shift_mmmd/`: PathMNIST source-vs-external center-shift MMMD experiments, including split sanity checks, CNN training metadata, smoke tests, power, and Type-I checks.

Tracked content includes experiment code, configuration files, README/plan files, summary CSVs, audit CSVs, and figures needed to inspect reported power and Type-I behavior.

Intentionally not tracked:

- Raw data/cache files under `data/` and local `.npz` split caches.
- CNN model weights such as `*.pt` and `*.pth`.
- R checkpoints such as `checkpoint.rds`.
- Large per-resample diagnostics such as `*_sigma_diagnostics.csv` and `*_all_inner_iters.csv`.
- Full raw PathMNIST per-test result tables; the committed summaries and plots are the reporting artifacts.
- Superseded smoke/inner=1 archives and the interrupted partial MNIST two-null run.

The saved GitHub artifacts are sufficient for inspecting the reported summaries and figures. Full raw tables, data files, and trained weights are kept locally and can be regenerated from the scripts when needed.
