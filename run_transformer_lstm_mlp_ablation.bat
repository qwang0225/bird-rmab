@echo off
setlocal

set ENV_NAME=bayesian_rmab
set CONDA_ROOT=C:\Users\frank\anaconda3
set ROOT=%~dp0

echo Activating conda environment: %ENV_NAME%
call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Could not activate conda environment "%ENV_NAME%".
    exit /b 1
)

cd /d "%ROOT%"

echo ============================================================
echo  1 of 3 ^| Encoder ablation: synthetic stationary
echo ============================================================
pushd synthetic-stationary
python encoder_ablation.py --N 20 --K 5 --T 100 --epochs 200 --n_episodes 100 --history_lengths 40 --encoders transformer lstm mlp --out_stem encoder_ablation_transformer_lstm_mlp_N20_K5 --out_dir experiment_outputs\encoder_ablation --force_retrain
if errorlevel 1 ( popd & echo [ERROR] synthetic stationary encoder ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  2 of 3 ^| Encoder ablation: synthetic drifting
echo ============================================================
pushd synthetic-drifting
python encoder_ablation.py --N 20 --K 5 --T 100 --epochs 200 --n_episodes 100 --history_lengths 40 --encoders transformer lstm mlp --out_stem encoder_ablation_transformer_lstm_mlp_N20_K5 --out_dir experiment_outputs\encoder_ablation --force_retrain
if errorlevel 1 ( popd & echo [ERROR] synthetic drifting encoder ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  3 of 3 ^| Encoder ablation: MIMIC-ICU
echo ============================================================
pushd mimic-icu
python encoder_ablation.py --N 20 --K 5 --T 100 --epochs 200 --n_episodes 100 --history_lengths 80 --encoders transformer lstm mlp --out_stem encoder_ablation_transformer_lstm_mlp_N20_K5 --out_dir experiment_outputs\encoder_ablation --force_retrain
if errorlevel 1 ( popd & echo [ERROR] mimic-icu encoder ablation failed. & exit /b 1 )
popd

echo.
echo ============================================================
echo  DONE: Transformer/LSTM/MLP ablation outputs written inside each environment folder
echo ============================================================
