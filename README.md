# BIRD: Belief-Encoder Index Restless Diffusion

## Setup

```bat
conda activate bayesian_rmab
```

Dependencies are listed in `environment.yml`. Set `CONDA_ROOT` in the batch
files if Anaconda is installed in a different location.

## Run the experiments

From the `BIRD` directory:

```bat
run_all.bat
```

The runner performs these steps:

1. Train missing baseline checkpoints and reuse existing valid checkpoints.
2. Validate saved weights before evaluation.
3. Run the main comparisons and Markov2 sanity check.
4. Retrain and evaluate ablation variants.
5. Evaluate comparisons at N=40 and N=100, then measure transfer inference timing.

Baseline training or checkpoint-validation failures stop the suite before
evaluation. Invalid existing checkpoints are reported without being overwritten.
Experiment-stage failures appear in the final summary and return a nonzero exit code.

Baseline training uses each script's default epochs, seed, and model settings.
The main environments use N=20, K=5; Markov2 uses N=50, K=10.

## Training and checkpoint commands

```bat
rem Preview missing-policy training commands without training
python train_missing.py --dry-run

rem Train missing baseline checkpoints without running experiments
python train_missing.py

rem Validate checkpoints without training
python check_checkpoints.py
```

The baseline stage covers BIRD, NeurWIN, PPO, MLP actor, Gaussian actor, and
learned rollout in the three main environments; BIRD, NeurWIN, and PPO in
Markov2. Checkpoint validation
checks saved weights, not training convergence.

## Run individual experiments

| Script | Experiment | Training behavior |
| --- | --- | --- |
| `run_main_experiments.bat` | Main comparisons and Markov2 | Train missing baselines |
| `run_all_policies_N40_N100.bat` | N=40, K=10 and N=100, K=25 comparisons | Reuse N=20, K=5 baselines; train if missing |
| `run_actor_ablation.bat` | Actor architecture: MLP, joint diffusion, Gaussian, BIRD | Retrain variants |
| `run_aux_ablation_experiments.bat` | Auxiliary prediction loss | Retrain variants |
| `run_critic_ablation_experiments.bat` | Critic architecture | Retrain variants |
| `run_window_l_ablation.bat` | History length | Retrain variants |
| `run_transformer_lstm_mlp_ablation.bat` | Belief encoder | Retrain variants |
| `run_factor_stress_ablation.bat` | Observation and dynamics uncertainty | Retrain variants |
| `run_transfer_timing.bat` | Inference timing | Train missing BIRD checkpoints |

All ablation launchers are included in `run_all.bat` and retrain their variants
by default. Checkpoints and results are saved in the corresponding environment's
variant and ablation directories. Boundary diagnostic scripts are in `plot/`
and require trained checkpoints.

## Figures and tables

After generating the required evaluation results:

```bat
python plot/prepare_results.py
python plot/plot_actor_comparison.py
```

Use `--input-dir` and `--output-dir` to choose different result and output
locations. `prepare_results.py` specifies the required evaluation files,
including separate MIMIC scale and history-window evaluations.

## MIMIC data

Simulator parameters are defined in `mimic-icu/env.py`. To fit parameters from
MIMIC data, provide the CSV path to `mimic-icu/fit_mimic_params_v2.py`.
Clinical data require separate access and are not included.
