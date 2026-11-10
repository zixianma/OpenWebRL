"""Action-only likelihood-ratio scoring and autoregressive guidance arithmetic."""
from __future__ import annotations

import math
from typing import Sequence


def action_score(positive_token_logps: Sequence[float],
                 negative_token_logps: Sequence[float]) -> float:
    """Sum log p+ - log p- over the *same* action tokens, including termination."""
    if not positive_token_logps or len(positive_token_logps) != len(negative_token_logps):
        raise ValueError("Positive and negative branches need aligned action tokens")
    if not all(math.isfinite(x) for x in (*positive_token_logps, *negative_token_logps)):
        raise ValueError("Nonfinite token log probability")
    return math.fsum(p - n for p, n in zip(positive_token_logps, negative_token_logps))


def guided_token_logprobs(base: Sequence[float], positive: Sequence[float],
                          negative: Sequence[float], strength: float) -> list[float]:
    """Normalize local log p0 + strength * (log p+ - log p-)."""
    if not base or len(base) != len(positive) or len(base) != len(negative):
        raise ValueError("All branches need the same nonempty legal token support")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Strength must be finite and nonnegative")
    if any(math.isnan(x) or x == math.inf for branch in (base, positive, negative)
           for x in branch):
        raise ValueError("Invalid token log probability")
    logits = []
    for b, p, n in zip(base, positive, negative):
        if n == -math.inf and p > -math.inf:
            raise ValueError("Positive policy lacks negative-policy support")
        logits.append(b if strength == 0 else
                      (-math.inf if b == -math.inf or p == -math.inf else b + strength * (p - n)))
    maximum = max(logits)
    if maximum == -math.inf:
        raise ValueError("Guidance eliminated every legal token")
    normalizer = maximum + math.log(math.fsum(math.exp(x - maximum) for x in logits))
    return [x - normalizer for x in logits]
