# BIRD NeurIPS Submission Code

This cleaned repository contains the code and saved evaluation arrays needed
to regenerate the paper figures and appendix tables for BIRD.

## Reproduce Paper Figures

From the repository root, activate the project environment and run:

```bat
conda activate bayesian_rmab
python plot\verify_paper_inputs.py
run_all.bat
```

`run_all.bat` regenerates the combined main-comparison figures, actor-ablation
figure, and appendix result tables from the included `.npz` evaluation files,
then copies the figures into `outputs/`.

The saved main-comparison arrays use the paper protocol:

- training seed: `0` for the trained models used to produce the saved arrays
- evaluation seed: `42`
- evaluation episodes: `100`
- horizons: `T=100`
- problem sizes: `N=20,K=5`, `N=40,K=8`, and `N=100,K=20`

The actor-ablation figure uses the saved seed-0 actor-ablation arrays for
`N=20,K=5`.

## Directory Layout

- `synthetic-stationary/`: stationary synthetic partially observable RMAB
- `synthetic-drifting/`: drifting synthetic partially observable RMAB
- `mimic-icu/`: MIMIC-III v1.4 based simulator construction and experiments
- `markov2/`: fully observable two-state RMAB sanity-check experiment
- `plot/`: final plotting, table-generation, and input-validation scripts
- `outputs/`: regenerated figure outputs created by `run_all.bat`

Checkpoints and raw MIMIC files are intentionally not included in this clean
submission tree. The paper figures are reproduced from saved evaluation arrays.
