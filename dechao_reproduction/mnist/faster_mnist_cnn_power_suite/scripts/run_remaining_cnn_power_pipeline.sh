#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dechao/kernel_two-sample"
SUITE_DIR="$ROOT/dechao_reproduction/mnist/faster_mnist_cnn_power_suite"
LOG="$SUITE_DIR/Results/run_remaining_cnn_power_pipeline.log"

mkdir -p "$(dirname "$LOG")"

run_exp() {
  local name="$1"
  local script="$2"
  echo "[$(date '+%F %T')] START $name" | tee -a "$LOG"
  conda run -n ktst-mnist Rscript "$script" 2>&1 | tee -a "$LOG"
  echo "[$(date '+%F %T')] FINISH $name" | tee -a "$LOG"
}

run_exp "faster_mnist_cnn_layer2_gaussian5_power" "$ROOT/dechao_reproduction/mnist/faster_mnist_cnn_layer2_gaussian5_power/Code/Kernel_Based_Tests/Body.R"
run_exp "faster_mnist_cnn_multilayer_single_gaussian_power" "$ROOT/dechao_reproduction/mnist/faster_mnist_cnn_multilayer_single_gaussian_power/Code/Kernel_Based_Tests/Body.R"
run_exp "faster_mnist_cnn_multilayer_gaussian15_power" "$ROOT/dechao_reproduction/mnist/faster_mnist_cnn_multilayer_gaussian15_power/Code/Kernel_Based_Tests/Body.R"
