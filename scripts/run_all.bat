@echo off
setlocal
cd /d "%~dp0.."
call "%~dp0run_all_experiments.bat"
exit /b %errorlevel%
