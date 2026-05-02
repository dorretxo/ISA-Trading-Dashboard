"""Shared atomic persistence helpers for JSON and text payloads.

These helpers centralize the temp-file + fsync + replace flow used across the
project and add Windows/cloud-sync-friendly retry handling around the final
rename step.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRANSIENT_WINERRORS = {5, 13, 32}


def _is_transient_replace_error(exc: OSError) -> bool:
    """Return True when a filesystem error is likely transient."""
    winerror = getattr(exc, "winerror", None)
    if winerror in _TRANSIENT_WINERRORS:
        return True
    return isinstance(exc, PermissionError)


def atomic_write_text(
    path: str | Path,
    payload: str,
    *,
    encoding: str = "utf-8",
    retries: int = 5,
    retry_delay: float = 0.35,
) -> None:
    """Write text atomically with retry handling on the final replace step."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f"{target.suffix}.tmp")

    last_error: OSError | None = None
    for attempt in range(retries):
        try:
            with open(tmp, "w", encoding=encoding) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(target)
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt < retries - 1 and _is_transient_replace_error(exc):
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise

    if last_error is not None:
        raise last_error


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    encoding: str = "utf-8",
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    default=str,
    retries: int = 5,
    retry_delay: float = 0.35,
) -> None:
    """Serialize JSON and persist it atomically."""
    text = json.dumps(
        payload,
        indent=indent,
        separators=separators,
        default=default,
    )
    atomic_write_text(
        path,
        text,
        encoding=encoding,
        retries=retries,
        retry_delay=retry_delay,
    )
