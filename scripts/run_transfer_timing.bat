@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\"
call "%~dp0activate_environment.bat"
if errorlevel 1 exit /b 1

python "%ROOT%train_missing.py" --scope timing
if errorlevel 1 exit /b 1

set OUT_DIR=%ROOT%experiment_outputs\timing_transfer
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set T=100
set EPISODES=100
set WARMUP=10
set HAD_FAILURE=0

echo ============================================================ > "%OUT_DIR%\summary.txt"
echo BIRD transfer inference timing >> "%OUT_DIR%\summary.txt"
echo T=%T% episodes=%EPISODES% warmup_steps=%WARMUP% >> "%OUT_DIR%\summary.txt"
echo ============================================================ >> "%OUT_DIR%\summary.txt"

call :run_env "synthetic-stationary" "checkpoints_dpmd\best.pth"
call :run_env "synthetic-drifting" "checkpoints_dpmd\best.pth"
call :run_env "mimic-icu" "checkpoints_dpmd\best.pth"

echo.
echo Done. Logs saved under "%OUT_DIR%".
echo Summary:
type "%OUT_DIR%\summary.txt"
if "%HAD_FAILURE%"=="0" (
  echo All timing runs completed successfully.
  exit /b 0
) else (
  echo One or more timing runs failed. See logs under "%OUT_DIR%".
  exit /b 1
)

:run_env
set ENV_DIR=%~1
set CKPT=%~2
echo.
echo ============================================================
echo Environment: %ENV_DIR%
echo ============================================================

pushd "%ROOT%%ENV_DIR%" || exit /b 1

call :run_one "%ENV_DIR%" "%CKPT%" 20 5
call :run_one "%ENV_DIR%" "%CKPT%" 50 12
call :run_one "%ENV_DIR%" "%CKPT%" 100 25
call :run_one "%ENV_DIR%" "%CKPT%" 1000 200

popd
exit /b 0

:run_one
set ENV_NAME=%~1
set CKPT_PATH=%~2
set N=%~3
set K=%~4
set LOG=%OUT_DIR%\%ENV_NAME%_N%N%_K%K%.log

echo [%ENV_NAME%] N=%N% K=%K%
echo [%ENV_NAME%] N=%N% K=%K% > "%LOG%"
python eval_dpmd_timing.py --ckpt "%CKPT_PATH%" --N %N% --K %K% --T %T% --n_episodes %EPISODES% --warmup_steps %WARMUP% >> "%LOG%" 2>&1
set STATUS=%ERRORLEVEL%
type "%LOG%"
if not "%STATUS%"=="0" (
  set HAD_FAILURE=1
  echo [%ENV_NAME%] N=%N% K=%K% FAILED with exit code %STATUS% >> "%OUT_DIR%\summary.txt"
) else (
  for /f "tokens=2 delims==" %%A in ('findstr /b "latency_ms_per_decision=" "%LOG%"') do (
    echo [%ENV_NAME%] N=%N% K=%K% latency_ms_per_decision=%%A >> "%OUT_DIR%\summary.txt"
  )
)
exit /b %STATUS%
