# OFM inverse-solver experiments

These experiments compare OFM inverse solvers on Eight Gaussians and on
FFHQ adult-to-child translation. Each run used seed 42, one NVIDIA A100
5 GB MIG slice, and a target of 30,000 training iterations.

## Setup and training

From the repository root, activate a Python 3.10 or newer environment and
install the dependencies. Profile comparison and model visualization run
on CPU. The recorded training runs used CUDA.

```bash
python -m pip install -r cc_experiments/requirements.txt
```

The arguments below match the Eight Gaussians runs. Each invocation trains
one method on a single device.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

for method in plain wolfe retry mlp mlp-retry restricted restricted-wolfe cap1000 nesterov; do
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
| `nesterov` | Per-sample accelerated updates with a residual-based velocity certificate |

Each run writes `profile.csv` and `ckpt.pt` to `cc_experiments/runs/<method>/seed_<seed>/`. Training checkpoints and evaluates the held-out sliced Wasserstein-2 metric every 500 iterations by default. Reuse the same settings and output directory to resume, or choose a fresh output directory for a new run. Training saves batches whose loss reaches four times the median of up to 100 preceding batches, once at least 20 preceding losses are available. Spike captures are saved under the run's `spikes/` directory.

Nesterov uses a velocity tolerance of 0.1 and allows up to 300 updates per
sample. It estimates local smoothness from sampled Hessians each batch.
The profile records the time spent estimating smoothness and the fraction
of samples whose returned iterate meets the velocity certificate.

## Compare the recorded runs

Each dataset has nine run profiles and nine pretrained models. Each
`profile.csv` contains the full logged history. The accompanying `model.pt`
contains the saved transport weights, warm-start weights where applicable,
run settings, and the number of completed steps at the saved checkpoint.
The FFHQ Nesterov profile reaches 12,320 steps. Its archived model comes
from the regular checkpoint at step 12,000.

The tables and figures use each run’s full history.
[Eight Gaussians comparison](../amortized_tests/comparison/) · [FFHQ comparison](ffhq_results/comparison/)

```bash
python cc_experiments/compare_results.py cc_experiments/results \
    --output analysis/toy
python cc_experiments/compare_results.py cc_experiments/ffhq_results \
    --output analysis/ffhq
```

## Replay the first toy spikes

Each capture contains the exact spike batch, the pre-update model and
optimizer state, and the observed inverse solution. `diagnose.py spike`
computes a float64 reference inverse on CPU.

| Method | First captured training iteration |
| --- | ---: |
| `plain` | [16,468](results/plain/seed_42/spikes/iter_016468.pt) |
| `mlp` | [16,469](results/mlp/seed_42/spikes/iter_016469.pt) |
| `retry` | [28,821](results/retry/seed_42/spikes/iter_028821.pt) |

```bash
python cc_experiments/diagnose.py spike \
    cc_experiments/results/plain/seed_42/spikes/iter_016468.pt \
    --output analysis/plain_spike
```

## Visualize the pretrained models

The toy plot maps common Gaussian inputs with each saved model. The FFHQ
figure compares translations of the same held-out adults. FFHQ rendering
uses the published latent codes, test images, and ALAE decoder cached by
`ffhq.py`. Both commands can run without training.

```bash
export OMP_NUM_THREADS=2
python cc_experiments/visualize_transport.py toy cc_experiments/results \
    --output analysis/toy_transport.png

cache="$HOME/ofm_ffhq_data"
python cc_experiments/ffhq.py "$cache"
python cc_experiments/visualize_transport.py ffhq cc_experiments/ffhq_results \
    --ffhq-data "$cache" --samples 4 --output analysis/ffhq_transport.png
```

## FFHQ adult-to-child results

These runs compare inverse solvers on the 512-dimensional FFHQ latent codes
used for the paper’s adult-to-child experiment. All nine profiles use seed 42.
The comparison uses each run’s full logged history. Seven methods reached
30,000 iterations. Nesterov logged 12,320 and `cap1000` logged 5,422.
[Paper, Figure 7](https://arxiv.org/html/2403.13117#S4.F7)

[Recorded profiles](ffhq_results/) · [Comparison figure](ffhq_results/comparison/comparison.png) · [Training curves](ffhq_results/comparison/training_curves.png) · [Run summary](ffhq_results/comparison/run_summary.csv)

For a fresh FFHQ run, the trainer selects the latent-space model and batch
settings when given the cache directory.

```bash
python cc_experiments/train.py --method plain --seed 42 --device cuda \
    --ffhq-data "$HOME/ofm_ffhq_data" --output cc_experiments/ffhq_runs
```
