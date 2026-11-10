"""Conservative GPT-4.1 request liability, without guessing image tokens.

The bound is the model's entire documented context window, not an estimate of
this request's actual usage. A fully metered success releases unused reserved
liability; failed/ambiguous attempts keep it. This intentionally limits paid
judge concurrency when the remaining campaign budget is small. No API call,
credential access or permission is performed here.

Sources checked 2026-10-11:
https://developers.openai.com/api/docs/models/gpt-4.1
https://developers.openai.com/api/docs/guides/images-vision?api-mode=chat
"""
from openwebrl.action_credit import digest


CONTEXT_TOKENS = 1_047_576
MAX_COMPLETION_TOKENS = 32_768
INPUT_MICRODOLLARS_PER_MILLION = 2_000_000
OUTPUT_MICRODOLLARS_PER_MILLION = 8_000_000


def input_bound_protocol():
    return dict(version="gpt41_full_context_liability_v1", model="gpt-4.1",
                context_tokens=CONTEXT_TOKENS, max_completion_tokens=MAX_COMPLETION_TOKENS,
                bound="entire documented context, including text and images",
                single_completion=True, tools=False, streaming=False,
                source="https://developers.openai.com/api/docs/models/gpt-4.1",
                reviewed_utc_date="2026-10-11")


def input_bound_protocol_sha256():
    return digest(input_bound_protocol())


def gpt41_context_upper_bound(payload):
    """Accept only this study's canonical chat shape and one bounded output.

Provider/model identity and standard pricing are separately pinned by the
controller. Other endpoints, tools, multiple choices, service tiers, audio,
and future model aliases require their own reviewed protocol. Using the full
context avoids an unproved local estimate for chat framing or image detail.
    """
    required = {"model", "messages", "max_completion_tokens"}
    if (not isinstance(payload, dict) or not required <= set(payload)
            or set(payload) - required - {"seed"}
            or payload["model"] != "gpt-4.1"):
        raise ValueError("only the pinned single-completion GPT-4.1 judge is bounded")
    output = payload["max_completion_tokens"]
    if type(output) is not int or not 0 < output <= MAX_COMPLETION_TOKENS:
        raise ValueError("explicit supported completion token ceiling required")
    messages = payload["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("explicit canonical judge messages required")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user"}):
            raise ValueError("only canonical system/user messages are supported")
        content = message["content"]
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise ValueError("explicit text/image content required")
        for part in content:
            if (isinstance(part, dict) and set(part) == {"type", "text"}
                    and part["type"] == "text" and isinstance(part["text"], str)):
                continue
            if not isinstance(part, dict) or set(part) != {"type", "image_url"} or part["type"] != "image_url":
                raise ValueError("unsupported billable input modality")
            image = part["image_url"]
            if (not isinstance(image, dict) or "url" not in image or set(image) - {"url", "detail"}
                    or not isinstance(image["url"], str) or not image["url"]
                    or image.get("detail", "auto") not in {"low", "high", "auto"}):
                raise ValueError("unsupported GPT-4.1 image input")
    return CONTEXT_TOKENS


def maximum_request_microdollars(output_tokens):
    if type(output_tokens) is not int or not 0 < output_tokens <= MAX_COMPLETION_TOKENS:
        raise ValueError("invalid completion ceiling")
    total = CONTEXT_TOKENS * INPUT_MICRODOLLARS_PER_MILLION + output_tokens * OUTPUT_MICRODOLLARS_PER_MILLION
    return (total + 999_999) // 1_000_000
