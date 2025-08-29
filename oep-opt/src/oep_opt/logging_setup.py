import logging
import sys

def setup_logging(logfile=None, level=logging.INFO):
    logger = logging.getLogger("oep-opt")
    logger.setLevel(level)

    # clear handlers if re-called
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.flush = sys.stdout.flush  # ensures auto-flush
    logger.addHandler(sh)

    # optional file handler
    if logfile:
        fh = logging.FileHandler(logfile, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
