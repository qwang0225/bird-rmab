# BIRD NeurIPS Submission Code

This repository contains the cleaned code and saved evaluation arrays needed to
regenerate the BIRD paper figures and appendix result tables without retraining.

## Environment

Create the conda environment from `environment.yml`:

```bat
conda env create -f environment.yml
conda activate bird-rmab
```

If the environment already exists, use:

```bat
conda activate bird-rmab
```

## Reproduce Paper Figures from Saved Arrays

From the repository root, run:

```bat
python plot\verify_paper_inputs.py
run_all.bat
```

`run_all.bat` regenerates the combined main-comparison figures, actor-ablation
figure, and appendix result tables from the included saved evaluation arrays.
Final outputs are written to `outputs/`.

The saved main-comparison arrays follow the paper protocol:

- training seed: `0` for the trained models used to produce the saved arrays
- evaluation seed: `42`
- evaluation episodes: `100`
- horizon: `T=100`
- problem sizes: `N=20,K=5`, `N=40,K=8`, and `N=100,K=20`

The actor-ablation figure uses the saved seed-0 actor-ablation arrays for
`N=20,K=5`.

## Training from Scratch

The clean release does not include trained checkpoints, but the training scripts
are included. Run commands from the corresponding environment directory.

### BIRD

```bat
cd synthetic-stationary
python diffusion_DPMD_train.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --ckpt_dir checkpoints_dpmd

cd ..\synthetic-drifting
python diffusion_DPMD_train.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --ckpt_dir checkpoints_dpmd

cd ..\mimic-icu
python diffusion_DPMD_train.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --ckpt_dir checkpoints_dpmd
```

### Baselines

Synthetic environments use `neurwin.py` and `ppo.py`:

```bat
cd synthetic-drifting
python neurwin.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --save_dir checkpoints_neurwin
python ppo.py --N 20 --K 5 --T 100 --seed 0 --epochs 400 --ckpt_dir checkpoints_ppo
```

The MIMIC-ICU simulator uses separate NeurWIN and PPO training scripts:

```bat
cd mimic-icu
python train_neurwin.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --save_dir checkpoints_neurwin
python train_ppo.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --save_dir checkpoints_ppo
```

### Actor Ablations

```bat
cd synthetic-drifting
python mlp_actor.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --ckpt_dir checkpoints_mlp_actor
python diffusion_joint_N_ablation.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --save_dir checkpoints_dpmd_joint_N

cd ..\mimic-icu
python mlp_actor.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --ckpt_dir checkpoints_mlp_actor
python diffusion_joint_N_ablation.py --N 20 --K 5 --T 100 --seed 0 --epochs 200 --save_dir checkpoints_dpmd_joint_N
```

### Evaluation after Training

After training checkpoints, run the environment-local comparison script. For
example:

```bat
cd synthetic-drifting
python run_comparison.py --N 20 --K 5 --T 100 --seed 42 --n_episodes 100 --out comparison_N20_K5.png --dpmd_ckpt checkpoints_dpmd\best.pth --nw_ckpt checkpoints_neurwin\best.pth --ppo_ckpt checkpoints_ppo\best.pth
python actor_ablation.py --N 20 --K 5 --T 100 --seed 42 --n_episodes 100
```

Then return to the repository root and run `run_all.bat` to regenerate combined
figures and tables from the saved evaluation arrays.

## Optional Boundary Diagnostic

The `plot/` folder also contains scripts for the boundary-rank diagnostic used
to motivate stochastic diffusion scores:

```bat
python plot\plot_adapt_particle_rank_posterior.py
python plot\plot_adapt_boundary_value_diagnostic.py
python plot\plot_adapt_boundary_combined_diagnostic.py
```

These scripts are not part of the default `run_all.bat` pipeline. The particle
rank diagnostic uses the drifting synthetic simulator. The oracle-value
boundary comparison additionally requires trained BIRD and deterministic MLP
actor checkpoints, which are not included in this anonymized clean release.
Therefore, the boundary diagnostic is provided as analysis code, while the
default reproducibility path regenerates the main paper figures from saved
evaluation arrays.

## Directory Layout

- `synthetic-stationary/`: stationary synthetic partially observable RMAB code and saved evaluation arrays.
- `synthetic-drifting/`: drifting synthetic partially observable RMAB code and saved evaluation arrays.
- `mimic-icu/`: MIMIC-III v1.4 derived simulator code and saved evaluation arrays.
- `markov2/`: fully observable two-state RMAB sanity-check code and saved result array.
- `plot/`: plotting, table-generation, and input-validation scripts.
- `outputs/`: regenerated figures and tables produced by `run_all.bat`.

## Saved Arrays and Data

The `.npz` files inside `synthetic-stationary/`, `synthetic-drifting/`,
`mimic-icu/`, and `markov2/` are intentional. They are saved evaluation arrays
used to reproduce the paper figures without shipping training checkpoints or
rerunning all experiments. Root-level generated `.png` and `.npz` files are
cleaned by `run_all.bat` after final outputs are copied to `outputs/`.

Checkpoints and raw MIMIC-III files are intentionally not included in this clean
submission tree. Raw MIMIC-III data must be obtained through PhysioNet
credentialed access under the MIMIC-III data-use terms.