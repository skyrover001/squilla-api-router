"""Make the repository importable as ``squilla_api_router`` in CI.

GitHub checkout directories often contain hyphens (``squilla-api-router``),
while Python imports require the package name ``squilla_api_router``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
if "squilla_api_router" not in sys.modules:
    package = ModuleType("squilla_api_router")
    package.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
    sys.modules["squilla_api_router"] = package
