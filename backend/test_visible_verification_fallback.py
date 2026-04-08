import importlib
import os
import sys
from types import SimpleNamespace

import pytest


os.environ.setdefault("EXA_API_KEY", "test-exa-key")


if "backend.main" in sys.modules:
    backend_main = sys.modules["backend.main"]
elif "main" in sys.modules:
    backend_main = sys.modules["main"]
else:
    backend_main = importlib.import_module("main")


class DummyLLM:
    async def ainvoke(self, *args, **kwargs):
        return SimpleNamespace(
            completion=(
                '{"credibility": 0.91, "relevance": 0.88, "evidence_strength": 0.79, '
                '"method_rigor": 0.81, "reproducibility": 0.74, "citation_support": 0.77, '
                '"sources_checked": [{"url": "https://example.com", "finding": "supports claim"}], '
                '"verification_summary": "LLM verification succeeded.", "confidence_level": "high"}'
            )
        )


class DummyBrowser:
    async def get_pages(self):
        return []

    async def stop(self):
        return None


@pytest.mark.asyncio
async def test_visible_verification_preserves_llm_result_when_browser_errors(monkeypatch):
    expected = {
        "credibility": 0.91,
        "relevance": 0.88,
        "evidence_strength": 0.79,
        "method_rigor": 0.81,
        "reproducibility": 0.74,
        "citation_support": 0.77,
        "sources_checked": [{"url": "https://example.com", "finding": "supports claim"}],
        "verification_summary": "LLM verification succeeded.",
        "confidence_level": "high",
    }

    async def fake_exa_retrieve(*args, **kwargs):
        return "stub exa context"

    async def fake_attempt_json_repair(*args, **kwargs):
        return None

    class FailingAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self, **kwargs):
            raise RuntimeError("browser agent failed")

    monkeypatch.setattr(backend_main, "exa_retrieve", fake_exa_retrieve)
    monkeypatch.setattr(backend_main, "attempt_json_repair", fake_attempt_json_repair)
    monkeypatch.setattr(backend_main, "send_browser_address", lambda: None)
    monkeypatch.setattr(backend_main, "Agent", FailingAgent)

    result = await backend_main.node_verification(
        idx=1,
        node={"id": 1, "text": "Claim text", "role": "Claim"},
        nodes_to_verify=[{"id": 1, "text": "Claim text", "role": "Claim"}],
        browser_needs_reset=False,
        browser=DummyBrowser(),
        llm=DummyLLM(),
        use_browser_verification=True,
    )

    assert result == expected


@pytest.mark.asyncio
async def test_visible_verification_preserves_llm_result_when_browser_times_out(monkeypatch):
    expected = {
        "credibility": 0.91,
        "relevance": 0.88,
        "evidence_strength": 0.79,
        "method_rigor": 0.81,
        "reproducibility": 0.74,
        "citation_support": 0.77,
        "sources_checked": [{"url": "https://example.com", "finding": "supports claim"}],
        "verification_summary": "LLM verification succeeded.",
        "confidence_level": "high",
    }

    async def fake_exa_retrieve(*args, **kwargs):
        return "stub exa context"

    async def fake_attempt_json_repair(*args, **kwargs):
        return None

    class HangingAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self, **kwargs):
            return SimpleNamespace(final_result=lambda: "")

    async def fake_wait_for(awaitable, *args, **kwargs):
        awaitable.close()
        raise TimeoutError("browser verification timed out")

    monkeypatch.setattr(backend_main, "exa_retrieve", fake_exa_retrieve)
    monkeypatch.setattr(backend_main, "attempt_json_repair", fake_attempt_json_repair)
    monkeypatch.setattr(backend_main, "send_browser_address", lambda: None)
    monkeypatch.setattr(backend_main, "Agent", HangingAgent)
    monkeypatch.setattr(backend_main.asyncio, "wait_for", fake_wait_for)

    result = await backend_main.node_verification(
        idx=1,
        node={"id": 1, "text": "Claim text", "role": "Claim"},
        nodes_to_verify=[{"id": 1, "text": "Claim text", "role": "Claim"}],
        browser_needs_reset=False,
        browser=DummyBrowser(),
        llm=DummyLLM(),
        use_browser_verification=True,
    )

    assert result == expected
