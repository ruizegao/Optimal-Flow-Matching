# Optimal Flow Matching

This repository contains **code** for the experiments of NeurIPS 2024 paper [Optimal Flow Matching: Learning Straight Trajectories in Just One Step](https://openreview.net/pdf?id=kqmucDKVcU) by [Nikita Kornilov](https://scholar.google.com/citations?user=zvQZigsAAAAJ&hl=en), [Petr Mokrov](https://scholar.google.com/citations?user=CRsi4IkAAAAJ&hl=en), [Alexander Gasnikov](https://scholar.google.ru/citations?user=AmeE8qkAAAAJ&hl=en) and [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en). 

We propose novel **Optimal Flow Matching** (OFM) method which extends conventional Flow Matching algorithm for specific "optimal" parameterization of transport ODE vector field.

<span style="color:green">**Pros. of our method**: </span>
* OFM recovers <u>straight</u> trajectories between source and target distributions in just <u>one</u> Flow Matching step.
* OFM solves quadratic <u>Optimal Transport</u> problem between source and target distributions.

<span style="color:red">**Cons. of our method**: </span>
* OFM is based on rather complex training procedure (with auxiliary optimization step).
* OFM relies on [Input Convex NNs](https://proceedings.mlr.press/v70/amos17b/amos17b.pdf) which have questionable scalability.

<p align="center"><img src="images/teaser_new.png" width="600" /></p>

## Working with the repository.

The notebooks for Illustrative 2D <u>*Gaussian*->*Eight Gaussians*</u> experiments:

```bash
.\ofm_notebooks\toy8gau_*.ipynb
```

The script for <u>Wasserstein-2 benchmark</u>:

```bash
.\ofm_scripts\bench_script.py
```
Before running this scripts place the checkpoints from the [W2 benchmark](https://openreview.net/pdf?id=CI0T_3l-n1) github repository [folder](https://github.com/iamalexkorotin/Wasserstein2Benchmark/tree/main/benchmarks/Mix3toMix10) in the folder `.\benchmarks\Mix3toMix10`.

The notebooks for <u>ALAE experiment</u>:

```bash
.\ofm_notebooks\alae_*.ipynb
```

Before running ALAE notebooks, have a look at the `./ALAE` folder (e.g. some packages need to be installed).

## Compute Canada experiments

See the [experiment guide](cc_experiments/README.md) for the Eight Gaussians runs, the recorded settings, and the commands to reproduce and compare them.

## Running the `gau_v2cpf` experiments

Notebook: `ofm_notebooks/toy8gau_v2cpf.ipynb` (Gaussian -> Eight Gaussians, `ICNN2CPF` model).

**Setup.** Install the notebook dependencies (a CUDA GPU is assumed by default):

```bash
pip install torch numpy pandas matplotlib tqdm jupyter
```

**Run.** Start Jupyter *from inside `ofm_notebooks/`* (the notebook does `sys.path.append('..')` to import `ofmsrc`):

```bash
cd ofm_notebooks
jupyter notebook toy8gau_v2cpf.ipynb
```

Then run all cells. The first cell holds the parameters:

| Parameter | Meaning |
|---|---|
| `DEVICE` | `'cuda:0'` by default; set `'cpu'` if you have no GPU (slower) |
| `INV_MAX_ITER` | cap on outer L-BFGS steps of the inner solve (the saved run uses `10`) |
| `MAX_ITER` | total number of training iterations (`30000`) |

**Outputs.** The notebook uses Strong Wolfe line search and writes to `ofm_notebooks/gau_v2cpf_cap<INV_MAX_ITER>_wolfe/`. This directory contains `samples.png`, `loss.png`, `profile.png` and `profile.csv` for iteration timings, `conv.png` and `conv.npz` for inverse convergence, model weights (`model*.pth`), and `ckpt.pt` after training.

**Resuming.** Training saves a checkpoint (`ckpt.pt`) every `CKPT_EVERY=500` iterations. Re-running the notebook resumes from it; raise `MAX_ITER` to train longer. Delete the output folder to start from scratch.

**Included results.** `ofm_notebooks/gau_v2cpf_cap10_wolfe/` contains the plots and profile for the Strong Wolfe run with `INV_MAX_ITER=10`. `ofm_notebooks/gau_v2cpf_cap10_plain/` contains an additional plain L-BFGS run. The earlier fork results are under `ofm_notebooks/gau_v2cpf_cap10/` and `ofm_notebooks/gau_v2cpf/`.

## Citation

If you find this repository or the ideas presented in our paper useful, please consider citing our paper.

```
@inproceedings{
    kornilov2024optimal,
    title={Optimal Flow Matching: Learning Straight Trajectories in Just One Step},
    author={Nikita Maksimovich Kornilov and Petr Mokrov and Alexander Gasnikov and Alexander Korotin},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=kqmucDKVcU}
}
```
