# Center-Shift Shared CNN

This folder stores the CNN trained only on `cnn_train_pool` for the PathMNIST center-shift experiment.

Training schedule requested by the user for the initial run:

```text
epochs 1-20: Adam lr=0.001, no scheduler
epochs 21-40: Adam lr=0.0005, no scheduler
```

Logs are written to `logs/pathmnist_train.log`. The best validation checkpoint is `models/pathmnist_cnn_checkpoint.pt`; the latest continuation checkpoint is `models/pathmnist_cnn_latest.pt`.
