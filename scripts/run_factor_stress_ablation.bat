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
echo  1 of 3 ^| Factor stress: synthetic stationary
echo ============================================================
pushd synthetic-stationary
python factor_stress_ablation.py --N 20 --K 5 --T 100 --L 40 --epochs 200 --learned_epochs 50 --n_episodes 100 --out_dir experiment_outputs\factor_stress --force_retrain
if errorlevel 1 ( popd & echo [ERROR] synthetic stationary factor stress ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  2 of 3 ^| Factor stress: synthetic drifting
echo ============================================================
pushd synthetic-drifting
python factor_stress_ablation.py --N 20 --K 5 --T 100 --L 40 --epochs 200 --learned_epochs 50 --n_episodes 100 --out_dir experiment_outputs\factor_stress --force_retrain
if errorlevel 1 ( popd & echo [ERROR] synthetic drifting factor stress ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  3 of 3 ^| Factor stress: MIMIC-ICU
echo ============================================================
pushd mimic-icu
python factor_stress_ablation.py --N 20 --K 5 --T 100 --L 80 --epochs 200 --learned_epochs 50 --n_episodes 100 --out_dir experiment_outputs\factor_stress --force_retrain
if errorlevel 1 ( popd & echo [ERROR] MIMIC-ICU factor stress ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  DONE: factor stress outputs written inside each environment folder
echo ============================================================
