"""Separate typing model: writes only free text (a field's message or the final answer).

The default backend is the served OpenWebRL actor itself, prompted with its own
full browser context and a forced tool-call prefix, so it skips reasoning and
writes just the argument. Any backend exposing ``__call__(kind, ctx) -> str`` works.
"""
from __future__ import annotations

import json
import time

from openwebrl.decision_actor.projection import parse_tool_call

PREFIX = {
    "write": '<tool_call>\n{"name": "write", "arguments": {"message": "',
    "done": '<tool_call>\n{"name": "done", "arguments": {"response": "',
}


def forced_prefix(input_text: str, kind: str) -> str:
    """Close the visible thinking block (empty) and open the tool call so only the argument is generated."""
    stripped = input_text.rstrip()
    if stripped.endswith("<think>"):
        return stripped + "\n\n</think>\n\n" + PREFIX[kind]
    return input_text + "<think>\n\n</think>\n\n" + PREFIX[kind]


class ActorTypingModel:
    def __init__(self, *, max_new_tokens: int = 160, temperature: float = 0.0):
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    async def __call__(self, kind: str, *, infer, url, input_text, images, timeout=None) -> tuple[str, dict]:
        if kind not in PREFIX:
            raise ValueError(kind)
        prompt = forced_prefix(input_text, kind)
        params = dict(temperature=self.temperature, max_new_tokens=self.max_new_tokens, stop=["</tool_call>"])
        started = time.monotonic()
        output = await infer(url, prompt, params, images, timeout_secs=timeout)
        continuation = output[0]
        tail = prompt[prompt.rindex("<tool_call>"):] + continuation
        if "</tool_call>" not in tail:
            tail += "\n</tool_call>"
        call = parse_tool_call(tail)
        key = "message" if kind == "write" else "response"
        text = (call or {}).get("arguments", {}).get(key) if call else None
        if not isinstance(text, str):
            # Unparsable JSON (e.g. a stray quote): keep the raw argument up to the first closing quote.
            raw = continuation.split('"}', 1)[0]
            try:
                text = json.loads('"' + raw + '"')
            except json.JSONDecodeError:
                text = raw.replace('\\"', '"')
        return text.strip(), dict(seconds=time.monotonic() - started, raw=continuation[:500], parsed=call is not None)
