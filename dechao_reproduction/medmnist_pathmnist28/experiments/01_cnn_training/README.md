# Phase 2: Shared PathMNIST CNN Training

This folder records the CNN training phase as an experiment record. The reusable model itself is stored only in `../../shared_model/` so later MMMD experiments can share the same frozen checkpoint.

Training command used from the repository root:

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n cv-hw2 python dechao_reproduction/medmnist_pathmnist28/shared_model/python/pathmnist_cnn_pipeline.py --force-retrain
```

Selection rule: best official validation split classification accuracy. Test accuracy is reported only after checkpoint selection.
