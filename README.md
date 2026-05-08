# BIRD: Belief-Encoder Index Restless Diffusion for Partially Observable RMABs

This repository contains the code and saved evaluation arrays needed to
reproduce the figures and tables from the BIRD paper. Checkpoints and raw
clinical data are intentionally omitted; saved `.npz` arrays enable figure
regeneration without retraining.

## Requirements

Create and activate the conda environment:

```bat
conda env create -f environment.yml
conda activate bird-rmab
```

## Quick start

To validate inputs and regenerate paper figures (from provided arrays):

```bat
python plot/verify_paper_inputs.py
run_all.bat
```

## Running experiments

Training and evaluation scripts live in the environment-specific folders.
Common entry points include `diffusion_DPMD_train.py`, `neurwin.py`,
`ppo.py`, and `run_comparison.py` (see each subfolder for options).

Training examples

```bat
# Train DPMD from scratch (defaults: seed=0, N=20, K=5, T=100, epochs=200)
python synthetic-stationary/diffusion_DPMD_train.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --ckpt_dir checkpoints_dpmd

# Evaluate trained checkpoints (example: produce comparison plot)
python synthetic-drifting/run_comparison.py --N 20 --K 5 --T 100 --n_episodes 100 --dpmd_ckpt checkpoints_dpmd/best.pth --out comparison_N20_K5.png --seed 42
```

## Project layout

- `synthetic-stationary/` — stationary synthetic experiments and saved arrays
- `synthetic-drifting/`  — drifting synthetic experiments and saved arrays
- `mimic-icu/`           — MIMIC-derived simulator scripts and saved arrays
- `markov2/`             — two-state RMAB sanity-check experiments
- `plot/`                — plotting and table-generation utilities
- `outputs/`             — generated figures and tables

## Notes

- Saved `.npz` files are provided to enable reproducibility without training.
- Raw MIMIC data must be obtained separately via PhysioNet and is not included.
- For questions or issues, please open an issue on the repository.

