"""Check each Groq model in the fallback chain: reachable, tool calling works, quota left.

    python check_models.py

Sends one tiny tool-calling request per model (a few hundred tokens in total) and
prints the remaining free-tier quota Groq reports for that model.
"""
from __future__ import annotations

import json
import sys
import time

from clinstat.config import settings
from clinstat.llm import GroqChat, MissingAPIKey, get_groq_api_key

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "inflate_for_dropout",
        "description": "Inflate an evaluable sample size for expected dropout.",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer"}, "dropout_rate": {"type": "number"}},
            "required": ["n", "dropout_rate"],
        },
    },
}
PROBE = [{"role": "user", "content": "Use the tool: inflate n=64 for 15% dropout. Call the tool only."}]


def main() -> int:
    try:
        from groq import Groq

        client = Groq(api_key=get_groq_api_key(), max_retries=0, timeout=60)
    except MissingAPIKey as exc:
        print(exc, file=sys.stderr)
        return 1

    models = settings.groq_models
    print(f"Fallback chain ({len(models)} models): {' -> '.join(models)}\n")
    ok_all = True
    for i, model in enumerate(models, start=1):
        print(f"[{i}/{len(models)}] {model} ...", flush=True)
        kwargs = GroqChat.request_kwargs(model, PROBE, [PROBE_TOOL], "auto", 0.0, "low", False, 300)
        t0 = time.perf_counter()
        try:
            raw = client.chat.completions.with_raw_response.create(**kwargs)
        except Exception as exc:
            ok_all = False
            print(f"    FAILED: {exc.__class__.__name__}: {str(exc)[:300]}\n")
            continue
        resp = raw.parse()
        msg = resp.choices[0].message
        h = raw.headers
        calls = msg.tool_calls or []
        tool_ok = bool(calls) and calls[0].function.name == "inflate_for_dropout"
        args = json.loads(calls[0].function.arguments) if tool_ok else None
        ok_all &= tool_ok
        print(f"    reachable: yes ({time.perf_counter() - t0:.1f} s)")
        print(f"    tool calling: {'OK ' + str(args) if tool_ok else 'NO TOOL CALL - text reply: ' + (msg.content or '')[:120]}")
        print(f"    requests/day left: {h.get('x-ratelimit-remaining-requests')} of {h.get('x-ratelimit-limit-requests')}")
        print(f"    tokens/min left:   {h.get('x-ratelimit-remaining-tokens')} of {h.get('x-ratelimit-limit-tokens')}\n")
    print("All models ready." if ok_all else "Some models failed - fix CLINSTAT_GROQ_MODELS (see Groq's Models page).")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
