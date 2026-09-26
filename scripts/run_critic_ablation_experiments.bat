@echo off
setlocal

set ENV_NAME=bayesian_rmab
set "ROOT=%~dp0..\"

echo Activating conda environment: %ENV_NAME%
call "%~dp0activate_environment.bat"
if errorlevel 1 (
    echo [ERROR] Could not activate conda environment "%ENV_NAME%".
    exit /b 1
)

cd /d "%ROOT%"

echo ============================================================
echo  1 of 3 ^| Critic ablation: synthetic stationary
echo ============================================================
pushd synthetic-stationary
python critic_ablation.py --N 20 --K 5 --T 100 --epochs 200 --n_episodes 100 --methods per_arm_twin per_arm_single joint_twin --out_dir experiment_outputs\critic_ablation --force_retrain
if errorlevel 1 ( popd & echo [ERROR] synthetic stationary critic ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  2 of 3 ^| Critic ablation: synthetic drifting
echo ============================================================
pushd synthetic-drifting
python critic_ablation.py --N 20 --K 5 --T 100 --epochs 200 --n_episodes 100 --methods per_arm_twin per_arm_single joint_twin --out_dir experiment_outputs\critic_ablation --force_retrain
if errorlevel 1 ( popd & echo [ERROR] synthetic drifting critic ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  3 of 3 ^| Critic ablation: MIMIC-ICU
echo ============================================================
pushd mimic-icu
python critic_ablation.py --N 20 --K 5 --T 100 --epochs 200 --n_episodes 100 --methods per_arm_twin per_arm_single joint_twin --out_dir experiment_outputs\critic_ablation --force_retrain
if errorlevel 1 ( popd & echo [ERROR] mimic-icu critic ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  DONE: critic ablation outputs written inside each environment folder
echo ============================================================
