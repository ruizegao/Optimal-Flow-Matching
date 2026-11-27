# Eight Gaussians experiments on Compute Canada

These experiments train the same two-dimensional Optimal Flow Matching model on a Gaussian source and an eight-component Gaussian mixture. Each method used seed 42, one CUDA device, and a target of 30,000 training iterations. The jobs requested one NVIDIA A100 5 GB MIG slice and used two CPU threads. Seven methods reached 30,000 iterations. The `cap1000` profile contains 23,767 iterations.

## Setup and training

From the repository root, activate a Python environment with CUDA-enabled PyTorch and install the dependencies.

```bash
python -m pip install -r cc_experiments/requirements.txt
```

The arguments below match the recorded runs. Each invocation trains one method on a single device. The Bash loop runs the methods sequentially.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

for method in plain wolfe retry mlp mlp-retry restricted restricted-wolfe cap1000; do
    python -u cc_experiments/train.py \
        --method "$method" --seed 42 --device cuda --iterations 30000 \
        --batch-size 1024 --width 1024 --hessian-chunk 128 \
        --pretrain-steps 200 --tolerance 0.01 --retry-displacement 25 \
        --t-min 0.01 --t-max 0.85 --output cc_experiments/runs
done
```

| Method | Inverse solve |
| --- | --- |
| `plain` | L-BFGS with up to 10 outer steps and no line search |
| `wolfe` | L-BFGS with Strong Wolfe line search on every batch |
| `retry` | Plain L-BFGS, with a Strong Wolfe retry from the interpolated points if any inverse moves farther than 25 units |
| `mlp` | Plain L-BFGS warm-started by an MLP trained on sufficiently converged inverses |
| `mlp-retry` | MLP warm start with the same Strong Wolfe retry rule |
| `restricted` | Plain L-BFGS with interpolation times sampled between 0.01 and 0.85 |
| `restricted-wolfe` | Strong Wolfe L-BFGS with the same restricted times |
| `cap1000` | Plain L-BFGS with up to 1,000 outer steps |

Each run writes `profile.csv` and `ckpt.pt` to `cc_experiments/runs/<method>/seed_<seed>/`. Training checkpoints and evaluates the held-out sliced Wasserstein-2 metric every 500 iterations by default. Reuse the same settings and output directory to resume, or choose a fresh output directory for a new run. Training saves batches whose loss reaches four times the median of up to 100 preceding batches, once at least 20 preceding losses are available. Spike captures are saved under the run's `spikes/` directory.

## Compare the recorded runs

The archived `results/` directory contains the eight training profiles. The [comparison outputs](../amortized_tests/comparison/) contain the tables and figures generated from the available iterations of each run. Generate the comparison from the archived profiles, or use `--iterations 23767` for tables at the same iteration budget across all eight methods.

```bash
python cc_experiments/compare_results.py cc_experiments/results \
    --output cc_experiments/runs/comparison
```

Give `compare_results.py` the `cc_experiments/runs` directory to compare fresh runs. For a locally captured spike, `diagnose.py spike` takes the capture file and computes a float64 reference inverse on CPU. `diagnose.py evaluate` takes one or more local checkpoint files and evaluates them on a fixed held-out batch.

The seed-42 `plain` run first captured a loss spike at iteration 16,468.
The capture includes the model and Adam state before that batch’s update,
along with the batch itself. [Pre-update plain L-BFGS capture](results/plain/seed_42/spikes/iter_016468.pt)

## FFHQ adult-to-child results

These runs compare inverse solvers on the 512-dimensional FFHQ latent codes
used for the paper’s adult-to-child experiment. All nine profiles use seed 42.
The comparison uses each run’s full logged history. Seven methods reached
30,000 iterations. Nesterov logged 12,320 and `cap1000` logged 5,422.
[Paper, Figure 7](https://arxiv.org/html/2403.13117#S4.F7)

[Recorded profiles](ffhq_results/) · [Comparison figure](ffhq_results/comparison/comparison.png) · [Training curves](ffhq_results/comparison/training_curves.png) · [Run summary](ffhq_results/comparison/run_summary.csv)
