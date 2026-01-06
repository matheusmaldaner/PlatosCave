from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional


def new_run_id() -> str:
    return os.environ.get("PLATOSCAVE_RUN_ID") or uuid.uuid4().hex[:10]


def _perf_ms() -> float:
    return time.perf_counter() * 1000.0


def _fmt_kv(fields: Dict[str, Any]) -> str:
    if not fields:
        return ""
    # keep logs readable; don't dump giant blobs
    parts = []
    for k in sorted(fields.keys()):
        v = fields[k]
        s = str(v)
        if len(s) > 200:
            s = s[:200] + "…"
        parts.append(f"{k}={s}")
    return " | " + " ".join(parts)


@dataclass
class PerfWriter:
    path: str
    enabled: bool = True

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def write(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


@contextmanager
def timed(
    logger,
    name: str,
    *,
    warn_ms: Optional[float] = None,
    perf: Optional[PerfWriter] = None,
    **fields: Any,
):
    t0 = _perf_ms()
    if perf is not None:
        perf.write({"event": f"{name}.start", "t_ms": int(t0), **fields})
    logger.info(f"{name}.start{_fmt_kv(fields)}")
    try:
        yield
    finally:
        t1 = _perf_ms()
        dt = t1 - t0
        out = dict(fields)
        out["dt_ms"] = int(dt)
        if perf is not None:
            perf.write({"event": f"{name}.end", "t_ms": int(t1), **out})

        msg = f"{name}.end{_fmt_kv(out)}"
        if warn_ms is not None and dt >= warn_ms:
            logger.warning(msg)
        else:
            logger.info(msg)

class Counter:
    """
    Simple counter for run statistics. Thread-safe enough for logging purposes.
    """
    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._d: Dict[str, int] = {}

    def inc(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._d[key] = self._d.get(key, 0) + int(n)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._d)
