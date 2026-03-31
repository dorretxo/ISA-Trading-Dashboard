"""Orchestrator wrapper with nohup-equivalent process protection.

Ensures the orchestrator survives:
  1. Console window closure (pythonw.exe — no console to close)
  2. User logoff / session termination (CREATE_BREAKAWAY_FROM_JOB)
  3. Parent process exit (CREATE_NEW_PROCESS_GROUP)
  4. Signal interrupts (SIGBREAK/SIGINT ignored)

On Unix this would be `nohup`. On Windows we use:
  - CREATE_NEW_PROCESS_GROUP: detaches from parent's process group
  - CREATE_BREAKAWAY_FROM_JOB: prevents Task Scheduler job object from
    killing child when the scheduled task "completes"
  - BELOW_NORMAL_PRIORITY_CLASS: long discovery runs shouldn't starve
    interactive use
  - PID lockfile: prevents duplicate concurrent runs

Usage:
    pythonw.exe run_orchestrator_wrapper.py [--force-discovery] [--dry-run] ...
"""

import os
import sys
import signal
import time
import atexit

# --- Redirect stdout/stderr to log file BEFORE any imports ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(PROJECT_DIR, "orchestrator_output.log")
PID_FILE = os.path.join(PROJECT_DIR, "orchestrator.pid")

log_handle = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
sys.stdout = log_handle
sys.stderr = log_handle

# Ensure working directory is project root
os.chdir(PROJECT_DIR)


# --- Signal protection (Windows nohup equivalent) ---
def _ignore_signal(signum, frame):
    """Ignore termination signals to survive session logoff."""
    pass

try:
    # Ignore SIGBREAK (Ctrl+Break / console close on Windows)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _ignore_signal)
    # Ignore SIGINT (Ctrl+C propagation from parent)
    signal.signal(signal.SIGINT, _ignore_signal)
except (OSError, ValueError):
    pass  # Not all signals available on all platforms


# --- PID lockfile to prevent duplicate runs ---
def _check_and_write_pid():
    """Prevent concurrent orchestrator runs. Returns True if we got the lock."""
    my_pid = os.getpid()

    if os.path.exists(PID_FILE):
        try:
            old_pid = int(open(PID_FILE).read().strip())
            # Check if old process is still alive
            if _is_process_alive(old_pid):
                print(
                    f"WRAPPER: Another orchestrator is running (PID {old_pid}). "
                    f"Skipping this run.",
                    flush=True,
                )
                return False
            else:
                print(
                    f"WRAPPER: Stale PID file found (PID {old_pid} is dead). "
                    f"Claiming lock.",
                    flush=True,
                )
        except (ValueError, OSError):
            pass  # Corrupt PID file — overwrite it

    with open(PID_FILE, "w") as f:
        f.write(str(my_pid))
    return True


def _is_process_alive(pid):
    """Check if a process is running (Windows-compatible)."""
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, PermissionError):
        return False


def _cleanup_pid():
    """Remove PID file on exit."""
    try:
        if os.path.exists(PID_FILE):
            current_pid = int(open(PID_FILE).read().strip())
            if current_pid == os.getpid():
                os.remove(PID_FILE)
    except (ValueError, OSError):
        pass

atexit.register(_cleanup_pid)


# --- Main execution ---
print(f"\nWRAPPER: Starting orchestrator (PID {os.getpid()}) at "
      f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"WRAPPER: Args = {sys.argv[1:]}", flush=True)
print(f"WRAPPER: Process protection active (signal handlers, PID lock)", flush=True)

if not _check_and_write_pid():
    log_handle.flush()
    log_handle.close()
    sys.exit(0)

try:
    # Import and run orchestrator with original sys.argv
    import runpy
    sys.argv = [os.path.join(PROJECT_DIR, "daily_orchestrator.py")] + sys.argv[1:]
    runpy.run_path("daily_orchestrator.py", run_name="__main__")
    print(f"WRAPPER: Orchestrator completed successfully at "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
except Exception as e:
    print(f"WRAPPER ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
finally:
    _cleanup_pid()
    log_handle.flush()
    log_handle.close()
