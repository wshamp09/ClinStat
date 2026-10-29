"""Model fallback chain (offline, with a scripted stand-in for the Groq client)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import message

from clinstat.agent import run_agent
from clinstat.llm import GroqChat, ModelsExhausted, friendly_error, is_fallback_error


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


class BadRequestError(Exception):
    status_code = 400


class FakeGroq:
    """``behaviour`` maps model name -> list of outcomes (exception instance or message), consumed in order."""

    def __init__(self, behaviour: dict):
        self.behaviour = {k: list(v) for k, v in behaviour.items()}
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.behaviour[kwargs["model"]].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=outcome)])


CHAIN = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def test_primary_used_when_available():
    fake = FakeGroq({CHAIN[0]: [message("hi")]})
    llm = GroqChat(CHAIN, client=fake)
    assert llm.complete([{"role": "user", "content": "x"}]).content == "hi"
    assert llm.model == CHAIN[0] and not llm.fell_back


def test_falls_back_on_rate_limit_and_stays_there():
    fake = FakeGroq({CHAIN[0]: [RateLimitError("429 rate limit reached")], CHAIN[1]: [message("a"), message("b")]})
    events = []
    llm = GroqChat(CHAIN, client=fake)
    llm.on_fallback = events.append
    assert llm.complete([{"role": "user", "content": "x"}]).content == "a"
    assert llm.complete([{"role": "user", "content": "y"}]).content == "b"  # sticky: primary not retried
    assert [r["model"] for r in fake.requests] == [CHAIN[0], CHAIN[1], CHAIN[1]]
    assert llm.model == CHAIN[1] and llm.fell_back
    assert events and events[0]["from"] == CHAIN[0] and events[0]["to"] == CHAIN[1]


def test_all_models_exhausted():
    fake = FakeGroq({CHAIN[0]: [RateLimitError("429")], CHAIN[1]: [RateLimitError("429")]})
    llm = GroqChat(CHAIN, client=fake)
    with pytest.raises(ModelsExhausted) as exc:
        llm.complete([{"role": "user", "content": "x"}])
    assert "fallback chain" in friendly_error(exc.value)
    assert llm.model == CHAIN[-1]  # still answers after the chain is exhausted


def test_agent_reports_exhausted_chain(retriever):
    fake = FakeGroq({CHAIN[0]: [RateLimitError("429")], CHAIN[1]: [RateLimitError("429")]})
    rec = run_agent("q", retriever, GroqChat(CHAIN, client=fake)).to_record()
    assert "fallback chain is rate-limited" in rec["error"]
    assert [f["from"] for f in rec["fallbacks"]] == CHAIN


def test_non_retryable_errors_do_not_fall_back():
    fake = FakeGroq({CHAIN[0]: [AuthenticationError("401 invalid key")], CHAIN[1]: [message("never")]})
    llm = GroqChat(CHAIN, client=fake)
    with pytest.raises(AuthenticationError):
        llm.complete([{"role": "user", "content": "x"}])
    assert len(fake.requests) == 1


def test_retired_model_name_falls_back():
    gone = BadRequestError("model `openai/gpt-oss-120b` has been decommissioned")
    fake = FakeGroq({CHAIN[0]: [gone], CHAIN[1]: [message("ok")]})
    assert GroqChat(CHAIN, client=fake).complete([{"role": "user", "content": "x"}]).content == "ok"
    assert is_fallback_error(gone)


def test_reasoning_options_per_model_and_think_tags_removed():
    fake = FakeGroq({CHAIN[0]: [RateLimitError("429")], CHAIN[1]: [message("<think>scratch work</think>\nFinal answer.")]})
    llm = GroqChat(CHAIN, client=fake)
    out = llm.complete([{"role": "user", "content": "x"}], reasoning_effort="low")
    assert out.content == "Final answer."
    assert fake.requests[0]["extra_body"] == {"reasoning_effort": "low"}
    assert fake.requests[1]["extra_body"] == {"reasoning_format": "hidden"}


def test_model_rejecting_reasoning_format_is_retried_without_it():
    reject = BadRequestError("'reasoning_format' is not supported with this model")
    fake = FakeGroq({CHAIN[1]: [reject, message("plain")]})
    out = GroqChat([CHAIN[1]], client=fake).complete([{"role": "user", "content": "x"}])
    assert out.content == "plain" and "extra_body" not in fake.requests[1]


def test_agent_records_fallback(retriever):
    from conftest import tool_call

    fake = FakeGroq({
        CHAIN[0]: [message(tool_calls=[tool_call("t1", "search_knowledge_bank", {"query": "washout"})]),
                   RateLimitError("429 tokens per minute")],
        CHAIN[1]: [message("Washout removes carryover [S1].")],
    })
    events = []
    result = run_agent("washout?", retriever, GroqChat(CHAIN, client=fake), on_event=events.append)
    rec = result.to_record()
    assert rec["error"] is None and rec["body"].startswith("Washout")
    assert rec["model"] == f"{CHAIN[0]}, {CHAIN[1]}"
    assert rec["fallbacks"][0]["to"] == CHAIN[1]
    assert any(e["type"] == "fallback" for e in events)
