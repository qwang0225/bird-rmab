@echo off
setlocal

set ENV_NAME=bayesian_rmab
set CONDA_ROOT=C:\Users\frank\anaconda3
set "ROOT=%~dp0..\"

echo Activating conda environment: %ENV_NAME%
call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Could not activate conda environment "%ENV_NAME%".
    exit /b 1
)

cd /d "%ROOT%"

python "%ROOT%train_missing.py" --scope main
if errorlevel 1 exit /b 1

if not exist "experiment_outputs" mkdir "experiment_outputs"
if not exist "experiment_outputs\main_comparison" mkdir "experiment_outputs\main_comparison"

echo ============================================================
echo  1 of 4 ^| Synthetic stationary
echo ============================================================
pushd synthetic-stationary
python run_comparison.py --n_episodes 100 --N 20 --K 5 --T 100 --out ..\experiment_outputs\main_comparison\synthetic_stationary_comparison.png
if errorlevel 1 ( popd & echo [ERROR] synthetic stationary failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  2 of 4 ^| Synthetic drifting
echo ============================================================
pushd synthetic-drifting
python run_comparison.py --n_episodes 100 --N 20 --K 5 --T 100 --out ..\experiment_outputs\main_comparison\synthetic_drifting_comparison.png
if errorlevel 1 ( popd & echo [ERROR] synthetic drifting failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  3 of 4 ^| MIMIC-ICU
echo ============================================================
pushd mimic-icu
python run_comparison.py --n_episodes 100 --N 20 --K 5 --T 100 --out ..\experiment_outputs\main_comparison\mimic_icu_comparison.png
if errorlevel 1 ( popd & echo [ERROR] mimic-icu failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  4 of 4 ^| Markov2 sanity check N=50 K=10
echo ============================================================
pushd "%ROOT%markov2"
if errorlevel 1 exit /b 1
python run.py --N 50 --K 10 --T 100 --n_eval 100 --seed 0 --no_score_plot
if errorlevel 1 ( popd & echo [ERROR] Markov2 evaluation failed. & exit /b 1 )
python evaluate_greedy_rollout.py
if errorlevel 1 ( popd & echo [ERROR] Markov2 greedy evaluation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  DONE: paper outputs written to experiment_outputs
echo ============================================================
