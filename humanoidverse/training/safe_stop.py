"""Lightweight filesystem protocol for checkpoint-before-exit requests."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

SAFE_STOP_REQUEST_FILENAME = ".ufo_safe_stop_request.json"
SAFE_STOP_STATUS_FILENAME = "safe_stop_status.json"


def safe_stop_request_path(work_dir: str | Path) -> Path:
    return Path(work_dir).expanduser().resolve() / SAFE_STOP_REQUEST_FILENAME


def safe_stop_status_path(work_dir: str | Path) -> Path:
    return Path(work_dir).expanduser().resolve() / SAFE_STOP_STATUS_FILENAME


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "request_id": f"malformed-{path.stat().st_mtime_ns}",
            "reason": f"malformed safe-stop request: {exc}",
        }
    return payload if isinstance(payload, dict) else {"reason": "malformed safe-stop request payload"}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def request_safe_stop(work_dir: str | Path, *, reason: str = "user request") -> dict[str, Any]:
    resolved_work_dir = Path(work_dir).expanduser().resolve()
    if not resolved_work_dir.is_dir():
        raise FileNotFoundError(f"Training work directory does not exist: {resolved_work_dir}")
    request_id = f"{time.time_ns()}-{os.getpid()}"
    payload = {
        "request_id": request_id,
        "reason": str(reason),
        "requested_at_unix": time.time(),
        "requesting_pid": os.getpid(),
        "requesting_host": socket.gethostname(),
    }
    atomic_write_json(safe_stop_request_path(resolved_work_dir), payload)
    return payload

