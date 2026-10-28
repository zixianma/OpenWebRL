"""Label-blind text adaptation of WEBPRMBench pairs for an actor-feature head.

The deployed head was trained on five screenshot-conditioned candidates. Here it
compares the two released text responses, with no image or continuation evidence.
This is a transfer evaluation, not the head's original input distribution.
"""
from __future__ import annotations

import hashlib


def parse_pair(prompt):
    """Extract exactly the released state text, thoughts and actions; no labels."""
    markers = ['#### Intent ####', '#### Assistant Responses ####',
               '[The Begin of Response 1]', '[The End of Response 1]',
               '[The Begin of Response 2]', '[The End of Response 2]']
    if any(prompt.count(m) != 1 for m in markers):
        raise ValueError('Ambiguous released prompt delimiters')
    positions = [prompt.index(m) for m in markers]
    if positions != sorted(positions):
        raise ValueError('Released prompt delimiters are out of order')
    context = prompt[positions[0]:positions[1]].strip()
    responses = []
    for i in (1, 2):
        body = prompt.split(f'[The Begin of Response {i}]', 1)[1].split(
            f'[The End of Response {i}]', 1)[0].strip()
        if not body.startswith('THOUGHT:') or body.count('\nACTION:') != 1:
            raise ValueError('Expected one released thought/action response')
        thought, action = body[len('THOUGHT:'):].split('\nACTION:', 1)
        if not thought.strip() or not action.strip():
            raise ValueError('Empty released thought or action')
        # The actor prefix ends at <think>; preserve content inside its native
        # thought/action boundary rather than treating all text as action tokens.
        responses.append(thought.strip() + '\n</think>\n' + action.strip())
    return context, tuple(responses)


def actor_prefix(tokenizer, context):
    prefix = tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': context}], tokenize=False,
        add_generation_prompt=True)
    return prefix if prefix.endswith('<think>\n') else prefix + '<think>\n'


def choose(scores):
    """Ties are unavailable predictions and count as errors in released metrics."""
    import math
    if len(scores) != 2 or not all(math.isfinite(v) for v in scores):
        raise ValueError('Two finite selection scores required')
    if scores[0] == scores[1]:
        return None
    return 1 if scores[0] > scores[1] else 2


def prompt_key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def inference_record(row):
    """Keep source-local identity; identical text in two rows is still two pairs."""
    return {key: row[key] for key in ('source_name', 'state_idx', 'pair_idx', 'prompt')}


def pair_identity(row):
    return row['source_name'], row['state_idx'], row['pair_idx']
