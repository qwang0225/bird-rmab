@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"

rem Evaluate existing default checkpoints: no retraining.
rem Preserve the N=20,K=5 activation ratio: 25 percent of arms.
set "K40=10"
set "K100=25"
set "T=100"
set "EVAL_EPISODES=100"
set "SEED=42"
set "ROLLOUT_H=10"
set "CONDA_ROOT=C:\Users\frank\anaconda3"
set "ENV_NAME=bayesian_rmab"
set "HAD_FAILURE=0"

if exist "%CONDA_ROOT%\Scripts\activate.bat" (
    call "%CONDA_ROOT%\Scripts\activate.bat" "%ENV_NAME%"
    if errorlevel 1 goto :activation_failed
)
python -c "import torch, numpy, matplotlib; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
if errorlevel 1 goto :dependencies_failed

python "%ROOT%train_missing.py" --scope scale
if errorlevel 1 exit /b 1

for %%E in (synthetic-stationary synthetic-drifting mimic-icu) do (
    call :run_one "%%E" 40 %K40%
    call :run_one "%%E" 100 %K100%
)
goto :finish

:run_one
pushd "%ROOT%%~1"
if errorlevel 1 (
    set "HAD_FAILURE=1"
    exit /b 1
)
set "PPO_CKPT=checkpoints_ppo\best.pth"
set "OUT_DIR=experiment_outputs\all_policies_transfer"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set "OUT_STEM=%OUT_DIR%\comparison_N%~2_K%~3"
echo.
echo Evaluating %~1: N=%~2 K=%~3, %EVAL_EPISODES% episodes
echo Using existing checkpoints; learned rollout H=%ROLLOUT_H%.
echo Log: %~1\%OUT_STEM%.log
python -u run_comparison.py --N %~2 --K %~3 --T %T% --n_episodes %EVAL_EPISODES% --seed %SEED% --learned_rollout_H %ROLLOUT_H% --ppo_ckpt "%PPO_CKPT%" --out "%OUT_STEM%.png" > "%OUT_STEM%.log" 2>&1
if errorlevel 1 goto :run_failed
rem Some comparison scripts catch policy errors and exit successfully.
rem Require all ten policies and finite episode returns before reporting success.
python -c "import numpy as np, sys; d=np.load(sys.argv[1]); names='random obs_greedy activation_greedy neurwin ppo mlp_actor gaussian_actor learned_rollout dpmd oracle_lookahead'.split(); bad=[n for n in names if n not in d.files or d[n].size != int(sys.argv[2]) or not np.isfinite(d[n]).all()]; print('Missing or invalid policies:', bad); sys.exit(bool(bad))" "%OUT_STEM%.npz" %EVAL_EPISODES%
if errorlevel 1 goto :run_failed
echo [OK] All ten policies saved to %~1\%OUT_STEM%.npz
popd
exit /b 0

:run_failed
echo [FAILED] See %~1\%OUT_STEM%.log
set "HAD_FAILURE=1"
popd
exit /b 1

:activation_failed
echo [ERROR] Could not activate %ENV_NAME%. Check CONDA_ROOT and ENV_NAME in this file.
set "HAD_FAILURE=1"
goto :finish

:dependencies_failed
echo [ERROR] Python needs torch, numpy and matplotlib. Check the environment settings above.
set "HAD_FAILURE=1"

:finish
echo.
if "%HAD_FAILURE%"=="0" (
    echo Done. Each environment has PNG, NPZ and log files in experiment_outputs\all_policies_transfer.
) else (
    echo Evaluation incomplete. Review the errors above and the evaluation logs.
)
pause
exit /b %HAD_FAILURE%
