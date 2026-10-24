"""
find_models.py - finds which Claude model ids your gateway key accepts, when nobody gave you
the exact ids and the gateway shows no model list.

It sends a tiny request ("Reply OK", at most 5 tokens) with each likely id and prints which
ones answer. Every test costs a fraction of a cent and is logged in cost_log.csv like any call.
Only EU ids are tried, as the platform team asked. Copy a working Sonnet id into CHAT_MODEL.

Usage:
  uv run find_models.py                       # try the usual Sonnet, Opus and Haiku ids
  uv run find_models.py bedrock/eu.anthropic.claude-sonnet-5   # try your own id(s)
"""
from __future__ import annotations

import logging
import sys

import config
import llm

CANDIDATES = [
    # newest first; Bedrock ids with and without the version ending
    "bedrock/eu.anthropic.claude-sonnet-5",
    "bedrock/eu.anthropic.claude-sonnet-5-v1:0",
    "bedrock/eu.anthropic.claude-sonnet-4-8",
    "bedrock/eu.anthropic.claude-sonnet-4-8-v1:0",
    "bedrock/eu.anthropic.claude-sonnet-4-6",
    "bedrock/eu.anthropic.claude-sonnet-4-6-v1:0",
    "bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "bedrock/eu.anthropic.claude-opus-5",
    "bedrock/eu.anthropic.claude-opus-5-v1:0",
    "bedrock/eu.anthropic.claude-opus-4-8",
    "bedrock/eu.anthropic.claude-opus-4-6",
    "bedrock/eu.anthropic.claude-opus-4-6-v1",
    "bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "bedrock/eu.anthropic.claude-haiku-4-5",
]


def try_model(model: str) -> tuple[bool, str]:
    """One tiny request with this model id. Returns (works, short reason)."""
    try:
        reply = llm.chat("Reply with the single word OK.", model=model, max_tokens=5, step="find_models")
        return True, f"answered {reply.strip()[:20]!r} (route: {llm.gateway_route()})"
    except llm.BudgetExceeded:
        raise
    except llm.LLMError as e:
        text = str(e)
        said = text.split("Server said:")[-1] if "Server said:" in text else text.split(": ", 1)[-1]
        code = next((c for c in ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404") if c in text), "error")
        return False, f"{code}: {' '.join(said.split())[:110]}"


def main() -> int:
    config.setup_logging()
    logging.getLogger("llm").setLevel(logging.WARNING)   # one line per model is enough
    if config.CHAT_BACKEND != "bifrost" or not (config.BIFROST_BASE_URL and config.BIFROST_API_KEY):
        print("Set CHAT_BACKEND=bifrost, BIFROST_BASE_URL and BIFROST_API_KEY in .env first.")
        return 1
    candidates = sys.argv[1:] or CANDIDATES
    working = []
    for model in candidates:
        ok, reason = try_model(model)
        print(f"[{'OK  ' if ok else 'no  '}] {model:55} {reason}")
        if ok:
            working.append(model)
    if working:
        best = next((m for m in working if "sonnet" in m), working[0])
        print(f"\nWorking ids: {len(working)}. Put this in .env:\nCHAT_MODEL={best}")
        opus = next((m for m in working if "opus" in m), None)
        if opus:
            print(f"For comparisons use: --model {opus}")
        return 0
    print(f"\nNo id worked (key sent as: {config.BIFROST_AUTH_HEADER}, route: {config.GATEWAY_API}).")
    print("If every line says 401 or 403, try in .env:  BIFROST_AUTH_HEADER=x-bf-vk  (then x-api-key),")
    print("and GATEWAY_API=anthropic. Otherwise ask the platform team for the exact model ids.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
