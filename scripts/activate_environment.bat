@echo off
rem Run from an Anaconda Prompt or a terminal with conda initialized.
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda was not found. Open an Anaconda Prompt or initialize conda for this terminal.
    exit /b 1
)
call conda activate bayesian_rmab
if errorlevel 1 (
    echo [ERROR] Could not activate bayesian_rmab. Create the environment using environment.yml first.
    exit /b 1
)
exit /b 0
