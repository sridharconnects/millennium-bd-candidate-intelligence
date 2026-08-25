#!/usr/bin/env python3
"""Validate the API configuration with one tiny call before spending real money.

Catches, in order, the four things that actually go wrong: no key, a malformed key,
a wrong/retired model id, and a key that authenticates but has no credit. Each failure
prints the specific fix rather than a stack trace.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from millennium.config import SETTINGS  # noqa: E402


def main() -> int:
    print(f"provider : {SETTINGS.llm.provider}")
    print(f"model    : {SETTINGS.llm.model}")
    print(f"DEMO_MODE: {SETTINGS.flags.demo_mode}  "
          f"({'replay only — no API calls will be made' if SETTINGS.flags.demo_mode else 'live'})")

    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        print("\nFAIL: ANTHROPIC_API_KEY is not set.\n"
              "  cp .env.example .env   then put your key in it.", file=sys.stderr)
        return 2
    if not key.startswith("sk-ant-"):
        print(f"\nFAIL: key does not look like an Anthropic key "
              f"(starts with {key[:7]!r}, expected 'sk-ant-').", file=sys.stderr)
        return 2
    print(f"key      : sk-ant-…{key[-4:]}  ({len(key)} chars)")

    if SETTINGS.flags.demo_mode:
        print("\nDEMO_MODE=1, so no live call was attempted. To parse for real:\n"
              "  sed -i '' 's/^DEMO_MODE=1/DEMO_MODE=0/' .env")
        return 0

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key, timeout=30)
        r = client.messages.create(
            model=SETTINGS.llm.model, max_tokens=16, temperature=0,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}])
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        print(f"\nOK: model replied {text!r} "
              f"({r.usage.input_tokens} in / {r.usage.output_tokens} out)")
        est = 10 * 3 * (4500 / 3.5) / 1e6 * SETTINGS.llm.price_in_per_mtok \
            + 10 * 3 * 900 / 1e6 * SETTINGS.llm.price_out_per_mtok
        print(f"Estimated cost for the full 10-resume run: ~${est:.3f}")
        print("\nReady. Run:  python scripts/run_pipeline.py")
        return 0
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        msg = str(e)
        print(f"\nFAIL ({name}): {msg[:400]}", file=sys.stderr)
        if "not_found" in msg or "model" in msg.lower():
            print("\n  The model id is probably wrong or retired. Set LLM_MODEL in .env "
                  "to a model your account can access.", file=sys.stderr)
        elif "credit" in msg.lower() or "billing" in msg.lower():
            print("\n  The key authenticates but the account has no credit. Add credit "
                  "at console.anthropic.com/settings/billing.", file=sys.stderr)
        elif "authentication" in msg.lower() or "401" in msg:
            print("\n  The key was rejected. Check for a stray space or a truncated "
                  "paste in .env.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
