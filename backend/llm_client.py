"""backend.llm_client

Offline/batch-friendly LLM adapter.

This repo uses browser_use's ChatOpenAI / ChatBrowserUse wrappers. Batch experiments
should not depend on the interactive stdout protocol used by the web UI, and they
should be resilient to transient API failures (rate limits, timeouts, network
hiccups) without silently producing empty outputs.

Key properties:
  - No stdout side effects.
  - Global concurrency limiter across ALL LLM calls.
  - Shared cooldown on rate limits (prevents thundering herd).
  - Exponential backoff + jitter retries on retryable failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple
import asyncio
import inspect
import logging
import random
import re
import time

from ._perf import timed

_LOG = logging.getLogger(__name__)

# Global semaphore: caps concurrent in-flight LLM calls across all papers/nodes.
_LLM_SEM: asyncio.Semaphore = asyncio.Semaphore(8)

# Global cooldown: when we hit a rate limit, pause *all* calls until this time.
_RATE_LIMIT_LOCK = asyncio.Lock()
_RATE_LIMIT_UNTIL_MONO: float = 0.0


def set_global_llm_concurrency(k: int) -> None:
    """Set global max concurrent LLM calls across the whole process."""
    global _LLM_SEM
    _LLM_SEM = asyncio.Semaphore(max(1, int(k)))


def _now_mono() -> float:
    return time.monotonic()


async def _sleep_if_cooldown_active() -> None:
    global _RATE_LIMIT_UNTIL_MONO
    async with _RATE_LIMIT_LOCK:
        until = float(_RATE_LIMIT_UNTIL_MONO or 0.0)
    now = _now_mono()
    if until > now:
        await asyncio.sleep(until - now)


def _is_rate_limit_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True
    # Common message fragments
    if "rate limit" in msg or "too many requests" in msg:
        return True
    # Some SDKs include status text
    if "error code: 429" in msg or (" 429" in msg and "limit" in msg):
        return True
    return False


def _extract_retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Best-effort parse of retry-after hints across SDKs/wrappers."""
    # 1) Common attributes
    for attr in ("retry_after", "retry_after_seconds", "retry_after_s"):
        v = getattr(exc, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)

    # 2) Response headers patterns (varies by library)
    headers = getattr(exc, "headers", None)
    resp = getattr(exc, "response", None)
    if headers is None and resp is not None:
        headers = getattr(resp, "headers", None)
    if isinstance(headers, dict):
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra is not None:
            try:
                return float(ra)
            except Exception:
                pass
        # Some OpenAI-style headers: seconds until reset
        for k in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            v = headers.get(k) or headers.get(k.title())
            if v:
                # Values are often like "1s" or "250ms"
                s = str(v).strip().lower()
                m = re.match(r"^(\d+(?:\.\d+)?)(ms|s)$", s)
                if m:
                    num = float(m.group(1))
                    unit = m.group(2)
                    return num / 1000.0 if unit == "ms" else num

    # 3) Parse from message text
    msg = str(exc)
    # e.g., "Please try again in 20s."
    m = re.search(r"try again in\s*(\d+(?:\.\d+)?)\s*s", msg, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # e.g., "retry after 10"
    m = re.search(r"retry\s*after\s*(\d+(?:\.\d+)?)", msg, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    return None


def _is_retryable_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if _is_rate_limit_error(exc):
        return True

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True

    # Heuristic: network / transport hiccups
    if "timeout" in name or "timed out" in msg:
        return True
    if "connection" in name or "connection" in msg:
        return True
    if "temporarily unavailable" in msg or "service unavailable" in msg:
        return True

    # Some wrappers surface HTTP status in text
    if "error code: 502" in msg or "error code: 503" in msg or "error code: 504" in msg:
        return True

    return False


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai"   # "openai" or "browser_use"
    model: str = "gpt-5-nano"
    temperature: float = 0.0
    timeout_s: Optional[float] = None

    # Retry policy (batch safety defaults)
    max_retries: int = 8
    min_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    # When a rate limit is detected but no retry hint is available, use this cooldown.
    default_rate_limit_cooldown_s: float = 10.0


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        from browser_use.llm.messages import UserMessage
        self._UserMessage = UserMessage

        kwargs: Dict[str, Any] = {"temperature": float(cfg.temperature)}

        if cfg.provider == "browser_use":
            # Uses Browser Use Cloud credits via BROWSER_USE_API_KEY
            from browser_use import ChatBrowserUse
            self._llm = ChatBrowserUse()
            return

        # Default: OpenAI (requires OPENAI_API_KEY)
        from browser_use import ChatOpenAI
        try:
            self._llm = ChatOpenAI(model=cfg.model, **kwargs)
        except TypeError:
            # Some wrappers use model_name instead of model
            self._llm = ChatOpenAI(model_name=cfg.model, **kwargs)

    def _extract_text(self, resp: Any) -> str:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp

        # Common fields across browser_use / OpenAI adapters
        for attr in ("content", "completion", "text", "output_text"):
            v = getattr(resp, attr, None)
            if isinstance(v, str) and v.strip():
                return v

        # Sometimes nested
        msg = getattr(resp, "message", None)
        if isinstance(msg, dict):
            for k in ("content", "text"):
                v = msg.get(k)
                if isinstance(v, str) and v.strip():
                    return v

        # Fallback: repr
        try:
            s = str(resp)
            return s if isinstance(s, str) else ""
        except Exception:
            return ""

    async def _ainvoke_once(self, prompt: str, *, response_format: Optional[Dict[str, Any]] = None) -> Any:
        """One attempt (no retries), with global concurrency + optional timeout."""
        await _sleep_if_cooldown_active()
        async with _LLM_SEM:
            # browser_use expects message objects
            messages = [self._UserMessage(content=prompt)]

            kwargs: Dict[str, Any] = {}
            if response_format is not None:
                kwargs["response_format"] = response_format

            # Some LLM wrappers may not accept kwargs on ainvoke; handle gracefully.
            async def _call():
                if kwargs:
                    try:
                        return await self._llm.ainvoke(messages, **kwargs)
                    except TypeError:
                        return await self._llm.ainvoke(messages)
                return await self._llm.ainvoke(messages)

            if self.cfg.timeout_s and self.cfg.timeout_s > 0:
                return await asyncio.wait_for(_call(), timeout=float(self.cfg.timeout_s))
            return await _call()

    async def _ainvoke_with_retries(
        self,
        prompt: str,
        *,
        response_format: Optional[Dict[str, Any]] = None,
        warn_ms: int = 15_000,
        perf: Optional[Dict[str, Any]] = None,
        run_id: str = "",
    ) -> Any:
        cfg = self.cfg
        attempts = max(1, int(cfg.max_retries) + 1)

        last_exc: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            try:
                with timed(
                    _LOG,
                    "llm.ainvoke",
                    warn_ms=int(warn_ms),
                    perf=perf,
                    run_id=run_id,
                    attempt=int(attempt),
                    max_attempts=int(attempts),
                    prompt_chars=len(prompt or ""),
                    response_format=str(response_format.get("type")) if isinstance(response_format, dict) else "",
                ):
                    return await self._ainvoke_once(prompt, response_format=response_format)

            except BaseException as e:
                last_exc = e

                if attempt >= attempts or not _is_retryable_error(e):
                    raise

                # Decide wait
                wait_s: Optional[float] = None
                if _is_rate_limit_error(e):
                    wait_s = _extract_retry_after_seconds(e)
                    if wait_s is None:
                        wait_s = float(cfg.default_rate_limit_cooldown_s)
                else:
                    # Exponential backoff
                    base = float(cfg.min_backoff_s)
                    wait_s = min(float(cfg.max_backoff_s), base * (2 ** (attempt - 1)))

                # Add jitter (0–25%)
                wait_s = float(wait_s)
                jitter = random.uniform(0.0, 0.25 * wait_s)
                wait_s = min(float(cfg.max_backoff_s), wait_s + jitter)

                # Share cooldown for rate limits (prevents a stampede on retry)
                if _is_rate_limit_error(e):
                    global _RATE_LIMIT_UNTIL_MONO
                    async with _RATE_LIMIT_LOCK:
                        _RATE_LIMIT_UNTIL_MONO = max(_RATE_LIMIT_UNTIL_MONO, _now_mono() + wait_s)

                # Log retry (truncate message to keep logs readable)
                emsg = str(e).replace("\n", " ")
                if len(emsg) > 300:
                    emsg = emsg[:300] + "…"
                _LOG.warning(
                    "LLM call retrying (attempt %s/%s, wait %.2fs): %s: %s",
                    attempt,
                    attempts,
                    wait_s,
                    type(e).__name__,
                    emsg,
                )
                await asyncio.sleep(wait_s)

        # Should never reach here
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM retry loop ended without a result")

    async def complete(self, prompt: str, *, perf: Optional[Dict[str, Any]] = None, run_id: str = "") -> str:
        resp = await self._ainvoke_with_retries(prompt, perf=perf, run_id=run_id, response_format=None)
        return self._extract_text(resp)

    async def complete_json(self, prompt: str, *, perf: Optional[Dict[str, Any]] = None, run_id: str = "") -> str:
        # Ask for JSON object if supported; still parse/repair upstream.
        resp = await self._ainvoke_with_retries(
            prompt,
            perf=perf,
            run_id=run_id,
            response_format={"type": "json_object"},
        )
        return self._extract_text(resp)

    async def aclose(self) -> None:
        # browser_use clients may not require explicit close.
        close = getattr(self._llm, "aclose", None)
        if close and inspect.iscoroutinefunction(close):
            await close()
