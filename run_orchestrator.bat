@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM ISA Portfolio Autonomous Orchestrator — Daily Scheduled Task
REM
REM Process protection (Windows nohup equivalent):
REM   1. pythonw.exe — no console window, immune to window-CLOSE events
REM   2. START /B — launches detached from .bat parent process
REM   3. Wrapper script — ignores SIGBREAK/SIGINT, PID lockfile prevents
REM      duplicate runs, atexit cleanup
REM   4. Task Scheduler config: "Run whether user is logged on or not" +
REM      "Do not stop if task runs longer than" = unchecked
REM
REM Net effect: discovery can run 2+ hours and survives logoff, console
REM closure, and Task Scheduler session timeout.
REM ═══════════════════════════════════════════════════════════════════════════

REM --- Paths ---
set PROJECT_DIR=%~dp0
set LOG_FILE=%PROJECT_DIR%orchestrator_output.log

REM --- Locate pythonw.exe (preferred) or fall back to python.exe ---
set PYTHONW_EXE=
for /f "tokens=*" %%i in ('python -c "import sys,os; print(os.path.join(os.path.dirname(sys.executable),'pythonw.exe'))" 2^>nul') do set PYTHONW_EXE=%%i

if exist "%PYTHONW_EXE%" (
    set PYTHON_EXE=%PYTHONW_EXE%
) else (
    set PYTHON_EXE=python
)

REM --- Change to project directory ---
cd /d "%PROJECT_DIR%"

REM --- Run with timestamp header appended to log ---
echo. >> "%LOG_FILE%"
echo ═══════════════════════════════════════════════════════════ >> "%LOG_FILE%"
echo Run started: %date% %time% >> "%LOG_FILE%"
echo Using: %PYTHON_EXE% >> "%LOG_FILE%"
echo Protection: nohup-equivalent (detached, signal-protected, PID-locked) >> "%LOG_FILE%"
echo ═══════════════════════════════════════════════════════════ >> "%LOG_FILE%"

REM --- Launch detached (START /B = no new window, process continues after .bat exits) ---
REM The wrapper handles logging, signal protection, and PID lockfile.
START "" /B "%PYTHON_EXE%" "%PROJECT_DIR%run_orchestrator_wrapper.py" %*

REM --- .bat exits immediately; pythonw continues in background ---
REM The wrapper's PID lockfile and atexit handler track lifecycle.
echo Launcher exit (process detached): %date% %time% >> "%LOG_FILE%"
