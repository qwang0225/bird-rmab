@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"
if not exist "outputs" mkdir "outputs"

echo Regenerating combined NeurIPS figures and appendix tables from saved result files.

python plot\verify_paper_inputs.py
if errorlevel 1 ( echo [ERROR] Saved result files do not match the paper protocol. & exit /b 1 )

python plot\plot_main_comparison_N20_K5.py
if errorlevel 1 ( echo [ERROR] Failed to plot N=20,K=5 comparison. & exit /b 1 )

python plot\plot_main_comparison_N40_K8.py
if errorlevel 1 ( echo [ERROR] Failed to plot N=40,K=8 comparison. & exit /b 1 )

python plot\plot_main_comparison_N100_K20.py
if errorlevel 1 ( echo [ERROR] Failed to plot N=100,K=20 comparison. & exit /b 1 )

python plot\plot_actor_ablation_N20_K5.py
if errorlevel 1 ( echo [ERROR] Failed to plot actor ablation. & exit /b 1 )

python plot\make_appendix_tables.py
if errorlevel 1 ( echo [ERROR] Failed to regenerate appendix tables. & exit /b 1 )

copy /Y main_comparison_N20_K5.png outputs\main_comparison_N20_K5.png >nul
copy /Y main_comparison_N40_K8.png outputs\main_comparison_N40_K8.png >nul
copy /Y main_comparison_N100_K20.png outputs\main_comparison_N100_K20.png >nul
copy /Y actor_ablation_combined_N20_K5.png outputs\actor_ablation_combined_N20_K5.png >nul


del /Q "main_comparison_N20_K5.png" "main_comparison_N20_K5.npz" 2>nul
del /Q "main_comparison_N40_K8.png" "main_comparison_N40_K8.npz" 2>nul
del /Q "main_comparison_N100_K20.png" "main_comparison_N100_K20.npz" 2>nul
del /Q "actor_ablation_combined_N20_K5.png" "actor_ablation_combined_N20_K5.npz" 2>nul
rmdir /S /Q "plot\__pycache__" 2>nul
echo Done.
