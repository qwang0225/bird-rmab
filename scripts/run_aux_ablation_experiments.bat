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
echo  1 of 3 ^| Aux ablation: synthetic stationary
echo ============================================================
pushd synthetic-stationary
python aux_ablation.py --N 20 --K 5 --T 100 --epochs 200 --n_episodes 100 --methods with_aux without_aux --out_dir experiment_outputs\aux_loss_ablation --force_retrain
if errorlevel 1 ( popd & echo [ERROR] synthetic stationary aux ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  2 of 3 ^| Aux ablation: synthetic drifting
echo ============================================================
pushd synthetic-drifting
python aux_ablation.py --N 20 --K 5 --T 100 --epochs 200 --n_episodes 100 --methods with_aux without_aux --out_dir experiment_outputs\aux_loss_ablation --force_retrain
if errorlevel 1 ( popd & echo [ERROR] synthetic drifting aux ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  3 of 3 ^| Aux ablation: MIMIC-ICU
echo ============================================================
pushd mimic-icu
python aux_ablation.py --N 20 --K 5 --T 100 --epochs 200 --n_episodes 100 --methods with_aux without_aux --out_dir experiment_outputs\aux_loss_ablation --force_retrain
if errorlevel 1 ( popd & echo [ERROR] mimic-icu aux ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  DONE: aux ablation outputs written inside each environment folder
echo ============================================================
