"""
StateManager — atomic JSON persistence with versioning, backup fallback, and serialization.

write() injects version metadata and maintains a .bak backup.
read() falls back to backup if primary file is corrupt or missing.
get_age_seconds() returns file modification time age.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import time
from datetime import datetime
from datetime import timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("state.json")
VERSION = 1


class StateManager:
    """
    Atomic JSON state persistence with versioning and backup fallback.

    write(state): serialize state dict to disk atomically with metadata injection
    read():       deserialize state dict from disk; fall back to .bak if needed
    get_age_seconds(): return seconds since file was last modified
    """

    def __init__(self, path: Path | str = _DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.tmp_file = self.path.with_suffix(".tmp")
        self.backup_file = self.path.with_suffix(".bak")

    @staticmethod
    def _json_serialiser(obj):
        """Serialize Enum and dataclass objects to JSON-compatible types."""
        if isinstance(obj, Enum):
            return obj.value
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def write(self, state: dict[str, Any]) -> None:
        """
        Atomically write state to disk with versioned metadata.

        Before overwrite, backs up existing primary file to .bak.
        Metadata includes _version, _written_at (timestamp), _written_at_iso (ISO 8601).
        """
        try:
            now = datetime.now(timezone.utc)
            enriched = {
                **state,
                "_version": VERSION,
                "_written_at": now.timestamp(),
                "_written_at_iso": now.isoformat(),
            }

            with open(self.tmp_file, "w") as f:
                json.dump(enriched, f, indent=2, default=self._json_serialiser)

            if self.path.exists():
                shutil.copy2(self.path, self.backup_file)

            os.replace(self.tmp_file, self.path)
            logger.debug("State written to %s", self.path)
        except Exception as exc:
            logger.error("StateManager.write() failed: %s", exc)

    def read(self) -> dict[str, Any]:
        """
        Read state from disk. Falls back to backup if primary is corrupt/missing.

        Returns empty dict only if both primary and backup are unavailable.
        """
        try:
            if self.path.exists():
                with open(self.path) as f:
                    return json.load(f)
        except Exception as exc:
            logger.warning("StateManager.read() failed on primary file: %s", exc)

        # Try backup file
        try:
            if self.backup_file.exists():
                with open(self.backup_file) as f:
                    data = json.load(f)
                logger.warning("StateManager.read() loaded from backup file")
                return data
        except Exception as exc:
            logger.warning("StateManager.read() also failed on backup file: %s", exc)

        return {}

    def get_age_seconds(self) -> float:
        """
        Return the age of the state file in seconds.

        Returns float('inf') if the file does not exist.
        """
        try:
            if self.path.exists():
                return time.time() - self.path.stat().st_mtime
        except Exception:
            pass
        return float("inf")
