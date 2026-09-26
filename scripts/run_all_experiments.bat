@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0..\"
cd /d "%ROOT%"

call "%~dp0activate_environment.bat"
if errorlevel 1 exit /b 1

python "%ROOT%train_missing.py" --scope all
if errorlevel 1 exit /b 1

set FAIL_COUNT=0
set STATUS_1=NOT_RUN
set STATUS_2=NOT_RUN
set STATUS_3=NOT_RUN
set STATUS_4=NOT_RUN
set STATUS_5=NOT_RUN
set STATUS_6=NOT_RUN
set STATUS_7=NOT_RUN
set STATUS_8=NOT_RUN
set STATUS_9=NOT_RUN

echo ============================================================
echo  Running all paper experiments
echo ============================================================
echo Root: %ROOT%
echo.

echo ============================================================
echo  1 of 9 ^| Main comparisons + Markov2
echo ============================================================
call "%~dp0run_main_experiments.bat"
if errorlevel 1 (
    set STATUS_1=FAILED
    set /a FAIL_COUNT+=1
) else (
    set STATUS_1=OK
)

echo.
echo ============================================================
echo  2 of 9 ^| Auxiliary loss ablation
echo ============================================================
call "%~dp0run_aux_ablation_experiments.bat"
if errorlevel 1 (
    set STATUS_2=FAILED
    set /a FAIL_COUNT+=1
) else (
    set STATUS_2=OK
)

echo.
echo ============================================================
echo  3 of 9 ^| Critic ablation
echo ============================================================
call "%~dp0run_critic_ablation_experiments.bat"
if errorlevel 1 (
    set STATUS_3=FAILED
    set /a FAIL_COUNT+=1
) else (
    set STATUS_3=OK
)






echo.
echo ============================================================
echo  4 of 9 ^| Window L ablation
echo ============================================================
call "%~dp0run_window_l_ablation.bat"
if errorlevel 1 (
    set STATUS_4=FAILED
    set /a FAIL_COUNT+=1
) else (
    set STATUS_4=OK
)

echo.
echo ============================================================
echo  5 of 9 ^| Transformer/LSTM/MLP encoder ablation
echo ============================================================
call "%~dp0run_transformer_lstm_mlp_ablation.bat"
if errorlevel 1 (
    set STATUS_5=FAILED
    set /a FAIL_COUNT+=1
) else (
    set STATUS_5=OK
)


echo.
echo ============================================================
echo  6 of 9 ^| Factor stress: partial obs known dyn + full obs unknown dyn
echo ============================================================
call "%~dp0run_factor_stress_ablation.bat"
if errorlevel 1 (
    set STATUS_6=FAILED
    set /a FAIL_COUNT+=1
) else (
    set STATUS_6=OK
)


echo.
echo ============================================================
echo  7 of 9 ^| N=40 and N=100 policy comparisons
echo ============================================================
call "%~dp0run_all_policies_N40_N100.bat"
if errorlevel 1 (
    set STATUS_7=FAILED
    set /a FAIL_COUNT+=1
) else (
    set STATUS_7=OK
)

echo.
echo ============================================================
echo  8 of 9 ^| Transfer inference timing
echo ============================================================
call "%~dp0run_transfer_timing.bat"
if errorlevel 1 (
    set STATUS_8=FAILED
    set /a FAIL_COUNT+=1
) else (
    set STATUS_8=OK
)

echo.
echo ============================================================
echo  9 of 9 ^| Actor ablation
echo ============================================================
call "%~dp0run_actor_ablation.bat"
if errorlevel 1 (
    set STATUS_9=FAILED
    set /a FAIL_COUNT+=1
) else (
    set STATUS_9=OK
)

echo.
echo ============================================================
echo  Final summary
echo ============================================================
echo  1 Main comparisons + Markov2:      !STATUS_1!
echo  2 Auxiliary loss ablation:                !STATUS_2!
echo  3 Critic ablation:                        !STATUS_3!
echo  4 Window L ablation:                      !STATUS_4!
echo  5 Transformer/LSTM/MLP encoder ablation:  !STATUS_5!
echo  6 Factor stress ablation:                 !STATUS_6!
echo  7 N=40 and N=100 policy comparisons:      !STATUS_7!
echo  8 Transfer inference timing:              !STATUS_8!
echo 9 Actor ablation:                         !STATUS_9!
echo.

if !FAIL_COUNT! GTR 0 (
    echo DONE WITH FAILURES: !FAIL_COUNT! batch(es) failed.
    exit /b 1
)

echo DONE: all paper experiments completed successfully.
exit /b 0
