r"""Portable launcher for Squilla API Router.

Adds the repo to sys.path so ``squilla_api_router.*`` imports resolve, then runs
uvicorn against ``app:app``.  It does NOT change the working directory: the
launcher is invoked with cwd = ASCII runtime dir (which holds app.py + the ML
model bundle), because LightGBM cannot open its model files from a non-ASCII
path on Windows.

Usage (from the runtime working directory, typically an ASCII temp dir):
    SET SQUILLA_REPO=E:\桌面\2026专利申请\squilla_api_router
    python C:\...\_run_server.py [--port 8021] [--host 127.0.0.1]

``SQUILLA_REPO`` may be omitted: it then defaults to this file's parent.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _repo_root() -> Path:
    env = os.environ.get("SQUILLA_REPO", "")
    if env:
        return Path(env).resolve()
    # This file lives at <pkg_root>/squilla_api_router/_run_server.py.
    # The package import path must resolve the PARENT of squilla_api_router/
    # (i.e. the directory that CONTAINS that package directory).
    return Path(__file__).resolve().parent.parent


def _uvicorn_log_config() -> dict:
    """Route uvicorn and router logs to size-rotated files."""
    route_log = Path(
        os.environ.get(
            "ROUTER_LOG_FILE",
            str(Path.cwd() / "squilla_router.log"),
        )
    ).resolve()
    access_log = Path(
        os.environ.get(
            "ROUTER_ACCESS_LOG_FILE",
            str(route_log.with_name("squilla_router_access.log")),
        )
    ).resolve()
    route_log.parent.mkdir(parents=True, exist_ok=True)
    access_log.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = max(1024, int(os.environ.get("ROUTER_LOG_MAX_BYTES", "20971520")))
    backup_count = max(1, int(os.environ.get("ROUTER_LOG_BACKUP_COUNT", "5")))
    level_name = os.environ.get("ROUTER_LOG_LEVEL", "INFO").strip().upper()
    if level_name not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        level_name = "INFO"
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": (
                    "timestamp=%(asctime)s level=%(levelname)s "
                    "logger=%(name)s %(message)s"
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": (
                    'timestamp=%(asctime)s level=%(levelname)s '
                    'client=%(client_addr)s "%(request_line)s" '
                    "status=%(status_code)s"
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                "use_colors": False,
            },
        },
        "handlers": {
            "router_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "filename": str(route_log),
                "encoding": "utf-8",
                "maxBytes": max_bytes,
                "backupCount": backup_count,
            },
            "access_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "access",
                "filename": str(access_log),
                "encoding": "utf-8",
                "maxBytes": max_bytes,
                "backupCount": backup_count,
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["router_file"],
                "level": level_name,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["router_file"],
                "level": level_name,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access_file"],
                "level": level_name,
                "propagate": False,
            },
            "squilla.router": {
                "handlers": ["router_file"],
                "level": level_name,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["router_file"],
            "level": "WARNING" if level_name != "DEBUG" else "INFO",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Squilla API Router")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8021)
    args = parser.parse_args()

    # The working directory may hold the runtime app.py + ASCII model bundle
    # (Windows temp copy) OR be the repo itself (Linux/macOS).  Either way the
    # cwd must take priority for "app" resolution.
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.append(str(repo))

    load_dotenv(cwd / ".env", override=False)
    os.environ["ROUTER_EXTERNAL_LOGGING"] = "1"

    import uvicorn

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        log_level="info",
        log_config=_uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
