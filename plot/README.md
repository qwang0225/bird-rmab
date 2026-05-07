# Plot Scripts

These scripts regenerate the combined figures and appendix tables used by
`paper/paper.tex` from the saved `.npz` result files.

Run from the repository root:

```bat
python plot\plot_main_comparison_N20_K5.py
python plot\plot_main_comparison_N40_K8.py
python plot\plot_main_comparison_N100_K20.py
python plot\plot_actor_ablation_N20_K5.py
python plot\make_appendix_tables.py
```

The root `run_all.bat` executes the same commands and copies the refreshed
figures into `paper/fig/`.
