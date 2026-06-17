# Shared PathMNIST CNN Model

This directory stores the reusable PathMNIST-28 CNN model and its training artifacts. It is dataset-level infrastructure, not an experiment-specific result folder.

## Rules

- Train only on the official PathMNIST train split.
- Select the checkpoint only by official validation split classification accuracy.
- Report test accuracy only after checkpoint selection.
- Use RGB float pixel values in `[0, 1]`, with no augmentation and no additional normalization for the main line.
- Reuse the checkpoint from this directory in later MMMD experiments.

## Training Command

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n cv-hw2 python dechao_reproduction/medmnist_pathmnist28/shared_model/python/pathmnist_cnn_pipeline.py --force-retrain
```

## Outputs

- `models/pathmnist_cnn_checkpoint.pt`: best validation checkpoint
- `models/train_config.json`: training configuration
- `models/train_history.csv`: epoch-level training history
- `models/training_curves.png`: loss and accuracy curves
- `models/classification_metrics.csv`: selected validation accuracy and post-selection test accuracy
- `models/test_confusion_matrix.csv`: test confusion matrix
- `models/test_confusion_matrix.png`: confusion matrix plot
- `logs/train.log`: training log
