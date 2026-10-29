"""Groq client wrapper.

The API key is read from Streamlit secrets or the environment (a local,
git-ignored ``.env`` is loaded by ``config``). It is never logged or shown.
"""
from __future__ import annotations

import os
import re
from typing import Any

from .config import settings

REASONING_EFFORTS = ("low", "medium", "high")


class MissingAPIKey(RuntimeError):
    pass


def get_groq_api_key() -> str:
    key = None
    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    key = (key or os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise MissingAPIKey(
            "GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml (local or Streamlit Cloud > Settings > "
            "Secrets) or to a local .env file. Never commit either file."
        )
    return key


def has_groq_api_key() -> bool:
    try:
        get_groq_api_key()
        return True
    except MissingAPIKey:
        return False


class ModelsExhausted(RuntimeError):
    """Every model in the chain failed with a fallback-eligible error."""


def _status(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def is_fallback_error(exc: Exception) -> bool:
    """Errors where trying the next model can help: rate limits, overload, retired model names."""
    status = _status(exc)
    text = str(exc).lower()
    if exc.__class__.__name__ == "RateLimitError" or status == 429 or "rate limit" in text:
        return True
    if status in (500, 502, 503, 504) or "over capacity" in text:
        return True
    if status == 404 or "model_not_found" in text or "decommissioned" in text or "does not exist" in text:
        return True
    return False


_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


class GroqChat:
    """Chat-completions client with tool calling and a model fallback chain.

    Requests go to the first model in ``models``. When a model is rate-limited (or
    overloaded / retired), the same request is retried on the next model, and that
    model is kept for the rest of this conversation turn so an answer is not built
    by alternating models. Create a new instance per question to start again from
    the first model.
    """

    def __init__(self, models: list[str] | tuple[str, ...] | str | None = None, client: Any = None):
        if isinstance(models, str):
            models = [models]
        self.models = list(models or settings.groq_models)
        if not self.models:
            raise ValueError("No Groq models configured (CLINSTAT_GROQ_MODELS).")
        self.active = 0
        self.used: list[str] = []
        self.fallbacks: list[dict] = []
        self.on_fallback = None  # optional callback(dict) so the UI can report a switch as it happens
        if client is None:
            from groq import Groq

            # One quick retry inside the SDK; after that we fall back to the next model
            # instead of waiting out a rate limit.
            client = Groq(api_key=get_groq_api_key(), max_retries=1, timeout=120)
        self.client = client

    # ---- model bookkeeping -------------------------------------------------
    @property
    def model(self) -> str:
        """Model(s) that produced the output so far (primary if nothing was sent yet)."""
        return ", ".join(self.used) if self.used else self.models[min(self.active, len(self.models) - 1)]

    @property
    def fell_back(self) -> bool:
        return bool(self.fallbacks)

    @staticmethod
    def request_kwargs(model: str, messages: list[dict], tools: list[dict] | None, tool_choice: str,
                       temperature: float, reasoning_effort: str | None, json_mode: bool,
                       max_tokens: int | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        if "gpt-oss" in model:
            if reasoning_effort:
                kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        else:
            # Other reasoning models (e.g. Qwen) would otherwise put their thinking in the answer text.
            kwargs["extra_body"] = {"reasoning_format": "hidden"}
        return kwargs

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ):
        temp = settings.temperature if temperature is None else temperature
        last_exc: Exception | None = None
        while self.active < len(self.models):
            model = self.models[self.active]
            kwargs = self.request_kwargs(model, messages, tools, tool_choice, temp, reasoning_effort, json_mode, max_tokens)
            try:
                try:
                    response = self.client.chat.completions.create(**kwargs)
                except Exception as exc:
                    # A model that rejects the reasoning_format option: retry once without it.
                    if _status(exc) == 400 and "reasoning_format" in str(exc) and "extra_body" in kwargs:
                        kwargs.pop("extra_body")
                        response = self.client.chat.completions.create(**kwargs)
                    else:
                        raise
            except Exception as exc:
                if not is_fallback_error(exc):
                    raise
                last_exc = exc
                nxt = self.models[self.active + 1] if self.active + 1 < len(self.models) else None
                event = {"from": model, "to": nxt, "reason": f"{exc.__class__.__name__}: {str(exc)[:160]}"}
                self.fallbacks.append(event)
                if self.on_fallback:
                    try:
                        self.on_fallback(event)
                    except Exception:
                        pass
                self.active += 1
                continue
            if model not in self.used:
                self.used.append(model)
            message = response.choices[0].message
            if getattr(message, "content", None):
                message.content = _THINK.sub("", message.content)
            return message
        raise ModelsExhausted(
            f"All Groq models are rate-limited or unavailable ({', '.join(self.models)}). Last error: {last_exc}"
        ) from last_exc


def friendly_error(exc: Exception) -> str:
    """Turn API errors into a short message for the UI."""
    name = exc.__class__.__name__
    text = str(exc)
    if isinstance(exc, MissingAPIKey):
        return text
    if isinstance(exc, ModelsExhausted):
        return ("Every Groq model in the fallback chain is rate-limited right now. Wait a minute and retry; if it "
                "persists, the daily free quota is used up and resets within 24 hours. Lowering 'Passages per "
                "search' or 'Reasoning effort' uses fewer tokens.")
    if name == "RateLimitError" or "rate limit" in text.lower() or "429" in text:
        return ("Groq rate limit reached (tokens or requests per minute). Wait a minute and retry, lower "
                "'Passages per search', or shorten the pasted text.")
    if name == "AuthenticationError" or "401" in text:
        return "Groq rejected the API key (401). Check GROQ_API_KEY in your secrets."
    if "context" in text.lower() and "length" in text.lower():
        return "The request was too long for the model. Use fewer passages or shorter input."
    return f"{name}: {text[:400]}"
