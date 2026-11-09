@echo off
rem Lodestar - double-click to start. Uses the Python 3.11 install that has yfinance/fastapi.
cd /d "%~dp0"
set PY=C:\Users\yashs\AppData\Local\Programs\Python\Python311\python.exe
if not exist "%PY%" set PY=python
"%PY%" app.py %*
pause
