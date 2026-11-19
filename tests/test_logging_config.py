from __future__ import annotations

import logging.handlers
from pathlib import Path
from uuid import uuid4

import app as router_app
from _run_server import _uvicorn_log_config


def test_router_file_logger_rotates(monkeypatch) -> None:
    log_file = Path(__file__).with_name(f".router-{uuid4().hex}.log")
    monkeypatch.setenv("ROUTER_LOG_ENABLED", "1")
    monkeypatch.setenv("ROUTER_LOG_FILE", str(log_file))
    monkeypatch.setenv("ROUTER_LOG_MAX_BYTES", "1024")
    monkeypatch.setenv("ROUTER_LOG_BACKUP_COUNT", "2")

    handler = router_app._configure_router_logging(force=True)
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.maxBytes == 1024
    assert handler.backupCount == 2
    try:
        router_app.logger.info("x" * 4096)
        router_app.logger.info("y" * 4096)
        handler.flush()
    finally:
        router_app.logger.removeHandler(handler)
        handler.close()

    try:
        assert log_file.exists()
        assert Path(str(log_file) + ".1").exists()
    finally:
        for suffix in ("", ".1", ".2"):
            Path(str(log_file) + suffix).unlink(missing_ok=True)


def test_uvicorn_logging_uses_separate_rotating_files(
    monkeypatch,
) -> None:
    base = Path(__file__).with_name(f".uvicorn-{uuid4().hex}")
    route_log = Path(str(base) + "-router.log")
    access_log = Path(str(base) + "-access.log")
    monkeypatch.setenv("ROUTER_LOG_FILE", str(route_log))
    monkeypatch.setenv("ROUTER_ACCESS_LOG_FILE", str(access_log))
    monkeypatch.setenv("ROUTER_LOG_MAX_BYTES", "4096")
    monkeypatch.setenv("ROUTER_LOG_BACKUP_COUNT", "3")

    config = _uvicorn_log_config()

    assert config["handlers"]["router_file"]["filename"] == str(
        route_log.resolve()
    )
    assert config["handlers"]["access_file"]["filename"] == str(
        access_log.resolve()
    )
    assert (
        config["handlers"]["router_file"]["class"]
        == "logging.handlers.RotatingFileHandler"
    )
    assert (
        config["formatters"]["access"]["()"]
        == "uvicorn.logging.AccessFormatter"
    )
    assert config["handlers"]["router_file"]["maxBytes"] == 4096
    assert config["handlers"]["router_file"]["backupCount"] == 3
