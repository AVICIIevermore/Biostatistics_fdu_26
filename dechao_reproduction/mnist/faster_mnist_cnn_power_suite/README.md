# Faster MNIST CNN Power Suite

Runs the sequential CNN power experiments in this order:
1. test-set paired raw pixel vs final embedding Gaussian-5
2. layer1 single-layer Gaussian-5
3. layer2 single-layer Gaussian-5
4. multilayer single Gaussian aggregation
5. multilayer Gaussian-15 MMMD

Each experiment writes its own Results directory and auto-generates plots at the end of Body.R.
The suite then aggregates all completed results into suite-level comparison plots under `Results/`.
