import logging
import os
from logging.handlers import TimedRotatingFileHandler

from app.logging_config import LOG_RETENTION_DAYS, configure_logging


def _reset_logging_state():
    logging.getLogger("app").handlers.clear()
    import app.logging_config as logging_config
    logging_config._configured_for["data_dir"] = None


def test_configure_logging_creates_log_directory_and_file(tmp_path):
    _reset_logging_state()

    configure_logging(str(tmp_path))

    log_file = tmp_path / "logs" / "app.log"
    assert log_file.parent.is_dir()

    logging.getLogger("app.test").info("hello")
    for handler in logging.getLogger("app").handlers:
        handler.flush()

    assert log_file.exists()
    assert "hello" in log_file.read_text(encoding="utf-8")


def test_configure_logging_uses_daily_rotation_with_expiry(tmp_path):
    _reset_logging_state()

    configure_logging(str(tmp_path))

    file_handlers = [
        h for h in logging.getLogger("app").handlers
        if isinstance(h, TimedRotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    handler = file_handlers[0]
    assert handler.when.lower() == "midnight"
    assert handler.backupCount == LOG_RETENTION_DAYS


def test_configure_logging_is_idempotent_for_same_data_dir(tmp_path):
    _reset_logging_state()

    configure_logging(str(tmp_path))
    handlers_after_first = list(logging.getLogger("app").handlers)
    configure_logging(str(tmp_path))
    handlers_after_second = list(logging.getLogger("app").handlers)

    assert handlers_after_first == handlers_after_second


def test_configure_logging_switches_data_dir_and_closes_old_handler(tmp_path):
    _reset_logging_state()

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    os.makedirs(first_dir, exist_ok=True)
    os.makedirs(second_dir, exist_ok=True)

    configure_logging(str(first_dir))
    configure_logging(str(second_dir))

    assert (first_dir / "logs" / "app.log").exists()
    assert (second_dir / "logs" / "app.log").parent.is_dir()

    for handler in logging.getLogger("app").handlers:
        if isinstance(handler, TimedRotatingFileHandler):
            assert os.path.dirname(handler.baseFilename) == os.path.join(
                str(second_dir), "logs"
            )
