import logging
import sys
import threading
from pathlib import Path

class LockedFileHandler(logging.FileHandler):
    _lock = threading.Lock()

    def emit(self, record):
        with self._lock:
            super().emit(record)
            self.flush()

def setup_logging(run_logfile=None, grad_logfile=None, level=logging.INFO):
    """
    Creates:
      - logger "oep-opt"         -> stdout + run_logfile (clean, objective-focused)
      - logger "oep-opt.grad"    -> grad_logfile only (gradient/Jacobian chatter)

    If grad_logfile is None, "oep-opt.grad" will propagate to "oep-opt".
    """
    run_logger = logging.getLogger("oep-opt")
    run_logger.setLevel(level)

    # Clear handlers if re-called
    if run_logger.hasHandlers():
        run_logger.handlers.clear()
    run_logger.propagate = False  # avoid double logging

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    run_logger.addHandler(sh)

    # Optional run.log (thread-safe)
    if run_logfile:
        run_logfile = Path(run_logfile)
        run_logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = LockedFileHandler(run_logfile, mode="w")
        fh.setFormatter(fmt)
        run_logger.addHandler(fh)

    # Gradient logger
    grad_logger = logging.getLogger("oep-opt.grad")
    grad_logger.setLevel(level)

    # Clear its handlers
    if grad_logger.hasHandlers():
        grad_logger.handlers.clear()

    # By default do not propagate (so grad chatter doesn't pollute run.log/stdout)
    grad_logger.propagate = False

    if grad_logfile:
        grad_logfile = Path(grad_logfile)
        grad_logfile.parent.mkdir(parents=True, exist_ok=True)
        gh = LockedFileHandler(grad_logfile, mode="w")
        gh.setFormatter(fmt)
        grad_logger.addHandler(gh)
    else:
        # If no grad file given, fall back to run_logger outputs
        grad_logger.propagate = True

    return run_logger
