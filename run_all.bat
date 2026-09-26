@echo off
setlocal
cd /d "%~dp0"
call run_all_experiments.bat
exit /b %errorlevel%
