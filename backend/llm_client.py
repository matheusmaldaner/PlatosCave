"""backend.llm_client

Offline/batch-friendly LLM adapter.

The interactive backend (backend/main.py + backend/server.py) streams JSON
messages to stdout for the web UI. Batch experiments should not depend on that
I/O protocol. This module provides a thin wrapper around the repo's preferred
LLM client (browser_use.ChatOpenAI) with a stable async API.

Design goals
 - No stdout side effects.
 - Minimal dependency surface.
 - Works without spinning up a Browser/Agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
import inspect

@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai"   # "openai" or "browser_use"
    model: str = "gpt-5-mini"
    temperature: float = 0.0
    timeout_s: Optional[float] = None

class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        from browser_use.llm.messages import UserMessage
        self._UserMessage = UserMessage

        kwargs = {"temperature": float(cfg.temperature)}

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
            self._llm = ChatOpenAI(model_name=cfg.model, **kwargs)

    def _extract_text(self, resp: Any) -> str:
        """Best-effort extraction of the model text from browser_use response objects."""
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

        return str(resp)

    async def complete(self, prompt: str) -> str:
        msg = self._UserMessage(content=prompt)
        resp = await self._llm.ainvoke([msg])
        return self._extract_text(resp)
    
    async def complete_json(self, prompt: str) -> str:
        """
        Request a strict JSON object response (when supported by the underlying client).
        Falls back to regular completion if response_format isn't supported.
        """
        msg = self._UserMessage(content=prompt)
        try:
            resp = await self._llm.ainvoke(
                messages=[msg],
                response_format={"type": "json_object"},
            )
        except (TypeError, AttributeError):
            # Older clients may not accept response_format
            resp = await self._llm.ainvoke([msg])
        return self._extract_text(resp)


    async def aclose(self) -> None:
        """Best-effort cleanup of underlying async HTTP clients."""
        if getattr(self, "_llm", None) is None:
            return

        # Try common close hooks directly on the wrapper
        for name in ("aclose", "close"):
            fn = getattr(self._llm, name, None)
            if callable(fn):
                res = fn()
                if inspect.isawaitable(res):
                    await res
                return

        # Try nested client objects (common in wrappers)
        for attr in ("client", "_client"):
            sub = getattr(self._llm, attr, None)
            if sub is None:
                continue
            for name in ("aclose", "close"):
                fn = getattr(sub, name, None)
                if callable(fn):
                    res = fn()
                    if inspect.isawaitable(res):
                        await res
                    return