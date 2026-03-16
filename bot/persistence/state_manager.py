"""
StateManager — interface stub.

Full implementation delivered in Phase 3 (03-01-PLAN.md).
CRITICAL: write() must use atomic write (tmp file + os.rename) to prevent
partial-write corruption on crash.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("state.json")


class StateManager:
    """
    Atomic JSON state persistence.

    write(state): serialize state dict to disk atomically (write .tmp, rename)
    read():       deserialize state dict from disk; return {} if missing
    """

    def __init__(self, path: Path | str = _DEFAULT_PATH) -> None:
        self.path = Path(path)

    def write(self, state: dict[str, Any]) -> None:
        """
        Atomically write state to disk.

        Phase 3 implementation: write to path.tmp then os.rename() to path.
        This stub writes directly (non-atomic) — safe for development only.
        """
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, self.path)
            logger.debug("State written to %s", self.path)
        except Exception as exc:
            logger.error("StateManager.write() failed: %s", exc)

    def read(self) -> dict[str, Any]:
        """Read state from disk. Returns empty dict if file is missing or corrupt."""
        try:
            if self.path.exists():
                with open(self.path) as f:
                    return json.load(f)
        except Exception as exc:
            logger.warning("StateManager.read() failed, returning empty state: %s", exc)
        return {}
