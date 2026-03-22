@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM ISA Portfolio Autonomous Orchestrator — Daily Scheduled Task
REM Runs via Windows Task Scheduler. Logs output to orchestrator_output.log
REM Secrets are loaded from .env by config.py (python-dotenv) — NOT set here.
REM ═══════════════════════════════════════════════════════════════════════════

REM --- Paths ---
REM Override PYTHON_EXE if python is not on PATH or you need a specific version.
set PYTHON_EXE=python
REM PROJECT_DIR defaults to the folder containing this .bat file.
set PROJECT_DIR=%~dp0
set LOG_FILE=%PROJECT_DIR%orchestrator_output.log

REM --- Change to project directory ---
cd /d "%PROJECT_DIR%"

REM --- Run with timestamp header appended to log ---
echo. >> "%LOG_FILE%"
echo ═══════════════════════════════════════════════════════════ >> "%LOG_FILE%"
echo Run started: %date% %time% >> "%LOG_FILE%"
echo ═══════════════════════════════════════════════════════════ >> "%LOG_FILE%"

"%PYTHON_EXE%" daily_orchestrator.py >> "%LOG_FILE%" 2>&1

echo Exit code: %ERRORLEVEL% >> "%LOG_FILE%"
echo Run finished: %date% %time% >> "%LOG_FILE%"
