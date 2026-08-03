import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_RETENTION_DAYS = 30

_configured_for: dict = {"data_dir": None}


def configure_logging(data_dir: str) -> None:
    """Attach a daily-rotating file handler (and console handler) to the "app" logger.

    Rotated files older than LOG_RETENTION_DAYS are deleted automatically by
    TimedRotatingFileHandler's backupCount. Idempotent per data_dir: repeat calls
    with the same data_dir are a no-op so the app can call this from lifespan
    without leaking file handles on reload.
    """
    if _configured_for["data_dir"] == data_dir:
        return

    logger = logging.getLogger("app")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        when="midnight",
        backupCount=LOG_RETENTION_DAYS,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    _configured_for["data_dir"] = data_dir
