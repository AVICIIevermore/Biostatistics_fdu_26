#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dechao/kernel_two-sample"
SUITE_DIR="$ROOT/dechao_reproduction/mnist/faster_mnist_cnn_power_suite"
LOG="$SUITE_DIR/Results/run_all.log"
mkdir -p "$SUITE_DIR/Results"
: > "$LOG"

log() {
  printf '[%s] %s
' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee -a "$LOG"
}

run_exp() {
  local name="$1"
  local body="$ROOT/dechao_reproduction/$name/Code/Kernel_Based_Tests/Body.R"
  log "starting $name"
  conda run -n ktst-mnist Rscript "$body" | tee -a "$LOG"
  log "finished $name"
}

run_exp "faster_mnist_testset_paired_raw_vs_final_gaussian5_power"
run_exp "faster_mnist_cnn_layer1_gaussian5_power"
run_exp "faster_mnist_cnn_layer2_gaussian5_power"
run_exp "faster_mnist_cnn_multilayer_single_gaussian_power"
run_exp "faster_mnist_cnn_multilayer_gaussian15_power"
log "starting suite summary plotting"
conda run -n ktst-mnist Rscript "$SUITE_DIR/scripts/plot_suite_results.R" | tee -a "$LOG"
log "suite summary plotting finished"
