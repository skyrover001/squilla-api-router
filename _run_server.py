"""Portable launcher for Squilla API Router.

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
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    env = os.environ.get("SQUILLA_REPO", "")
    if env:
        return Path(env).resolve()
    # This file lives at <pkg_root>/squilla_api_router/_run_server.py.
    # The package import path must resolve the PARENT of squilla_api_router/
    # (i.e. the directory that CONTAINS that package directory).
    return Path(__file__).resolve().parent.parent


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

    import uvicorn

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
