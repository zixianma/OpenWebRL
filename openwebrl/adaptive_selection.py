"""Sequential stopping for a frozen pointwise action-selection head.

The caller generates and scores one complete actor proposal at a time. Scores
must be comparable across proposal counts; a listwise head is not suitable.
The policy changes only how many proposals are requested, never their seeds or
the actor distribution used for each proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class AdaptiveScorePolicy:
    threshold: float
    min_candidates: int = 1
    max_candidates: int = 5

    def __post_init__(self):
        if not math.isfinite(self.threshold):
            raise ValueError('Adaptive threshold must be finite')
        if not 1 <= self.min_candidates <= self.max_candidates:
            raise ValueError('Invalid adaptive candidate bounds')

    def should_stop(self, scores):
        if not scores or len(scores) > self.max_candidates or not all(math.isfinite(x) for x in scores):
            raise ValueError('Invalid adaptive score prefix')
        return len(scores) >= self.max_candidates or (
            len(scores) >= self.min_candidates and max(scores) >= self.threshold)

    def selected_index(self, scores):
        if not scores or not all(math.isfinite(x) for x in scores):
            raise ValueError('No finite candidate scores')
        return max(range(len(scores)), key=lambda i: (scores[i], -i))

    def replay(self, scores):
        """Return (number requested, selected index) using only seen prefixes."""
        if len(scores) < self.max_candidates:
            raise ValueError('Replay lacks the full maximum-candidate panel')
        for n in range(1, self.max_candidates + 1):
            if self.should_stop(scores[:n]):
                return n, self.selected_index(scores[:n])
        raise AssertionError('Maximum candidate count did not stop')

    @classmethod
    def from_file(cls, path, head_path):
        """Load the frozen threshold only for its exact head checkpoint."""
        settings = json.loads(Path(path).read_text())
        head_hash = hashlib.sha256(Path(head_path).read_bytes()).hexdigest()
        if (settings.get('revision') != 'adaptive-selection-score-threshold-v1' or
                settings.get('head_sha256') != head_hash):
            raise ValueError('Adaptive policy has the wrong head identity')
        return cls(settings['threshold'], settings['min_candidates'], settings['max_candidates'])


async def select_adaptively(propose, score, policy: AdaptiveScorePolicy):
    """Ask for one proposal at a time; return only proposals actually generated.

    `propose(index)` and `score(index, proposal)` are async callbacks. The score
    callback must evaluate the new proposal with the same frozen pointwise head.
    """
    proposals, scores = [], []
    for index in range(policy.max_candidates):
        proposal = await propose(index)
        value = float(await score(index, proposal))
        if not math.isfinite(value):
            raise ValueError('Adaptive scorer returned a nonfinite value')
        proposals.append(proposal)
        scores.append(value)
        if policy.should_stop(scores):
            break
    return {'proposals': proposals, 'scores': scores,
            'selected_index': policy.selected_index(scores),
            'generated_candidates': len(proposals)}
