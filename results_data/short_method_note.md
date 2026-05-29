# MNIST-CNN Graph-MMMD Supplement

- Status: waiting for embedding CSV; no graph-MMMD run has been executed yet.
- Scope: graph-MMMD only; existing Gaussian5/CNN-MMMD results are referenced, not rerun.
- Dataset target: MNIST-CNN reproduction task from `dechao_reproduction/`.
- Feature input: final FC 128-d embeddings from `results_data/mnist_final_fc128_embeddings.csv`.
- Power task: `set123_vs_set128`, labels `{1,2,3}` vs `{1,2,8}`.
- Null task: `set123_vs_set123`, labels `{1,2,3}` split into two balanced samples.
- Default parameters: sample_sizes=`60,120`; n_reps=`30`; B_boot=`200`; alpha=`0.05`; k_nn=`5`; t_seq=`0.1,0.5,1,2,5`; seed=`20260528`.
- Notes: this is a lightweight supplement and should not be treated as a full-scale estimate.
