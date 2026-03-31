"""Persistent cache for expensive analysis outputs.

Used for deep-analysis stages that are expensive but reasonably stable within
the same trading session, such as sentiment aggregation and MoE forecasts.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "feature_cache"


class PersistentAnalysisCache:
    """Small JSON-backed TTL cache keyed by namespace and cache key."""

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.path = _CACHE_DIR / f"{namespace}_cache.json"
        self._loaded = False
        self._dirty = False
        self._data: dict[str, dict] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            self._data = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw = payload.get("entries", {})
            self._data = raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Persistent cache load failed for %s: %s", self.namespace, e)
            self._data = {}

    def get(self, key: str, ttl_seconds: float) -> dict | None:
        self._ensure_loaded()
        entry = self._data.get(key)
        if not isinstance(entry, dict):
            return None
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
        except (KeyError, TypeError, ValueError):
            return None

        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age > ttl_seconds:
            return None

        value = entry.get("value")
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: dict) -> None:
        self._ensure_loaded()
        self._data[key] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "value": value,
        }
        self._dirty = True

    def save(self) -> None:
        self._ensure_loaded()
        if not self._dirty:
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "namespace": self.namespace,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "entry_count": len(self._data),
                "entries": self._data,
            }
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self.path)
            self._dirty = False
        except OSError as e:
            logger.warning("Persistent cache save failed for %s: %s", self.namespace, e)
