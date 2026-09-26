@echo off
setlocal
set "ROOT=%~dp0"
set "CONDA_ROOT=C:\Users\frank\anaconda3"
call "%CONDA_ROOT%\Scripts\activate.bat" bayesian_rmab
if errorlevel 1 exit /b 1
for %%E in (synthetic-stationary synthetic-drifting mimic-icu) do (
    call :run_environment %%E
    if errorlevel 1 exit /b 1
)
exit /b 0

:run_environment
pushd "%ROOT%%~1"
if errorlevel 1 exit /b 1
rem Train every actor variant from scratch in dedicated ablation directories.
python diffusion_DPMD_train.py --N 20 --K 5 --T 100 --epochs 200 --seed 0 --ckpt_dir checkpoints_actor_ablation_dpmd
if errorlevel 1 goto :failed
python mlp_actor.py --N 20 --K 5 --T 100 --epochs 200 --seed 0 --ckpt_dir checkpoints_actor_ablation_mlp
if errorlevel 1 goto :failed
python gaussian_actor.py --N 20 --K 5 --T 100 --epochs 200 --seed 0 --ckpt_dir checkpoints_actor_ablation_gaussian
if errorlevel 1 goto :failed
python diffusion_joint_N_ablation.py --N 20 --K 5 --T 100 --epochs 200 --seed 0 --save_dir checkpoints_actor_ablation_joint
if errorlevel 1 goto :failed
python actor_ablation.py --N 20 --K 5 --T 100 --n_episodes 100 --seed 42 --dpmd_ckpt checkpoints_actor_ablation_dpmd/best.pth --mlp_actor_ckpt checkpoints_actor_ablation_mlp/best.pth --gaussian_actor_ckpt checkpoints_actor_ablation_gaussian/best.pth --joint_n_ckpt checkpoints_actor_ablation_joint/best.pth --out experiment_outputs/actor_ablation/actor_ablation_N20_K5.png
if errorlevel 1 goto :failed
popd
exit /b 0

:failed
echo [ERROR] Actor ablation failed for %~1.
popd
exit /b 1
