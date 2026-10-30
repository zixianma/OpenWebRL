"""Input transport projection for preserved native Responses actor history.

The provider returns reasoning ``status`` but rejects it when that same item is
sent as input (HTTP 400, input[2].status). SDK output serialization is therefore
not an input transport contract. Keep the original output receipts/history
untouched; project only the separate request copy. Never decode opaque state.
"""
from copy import deepcopy


def actor_transport_input(items):
    """Remove response lifecycle metadata, retaining native execution identity.

    Reasoning input fields follow ResponseReasoningItemParam in the official
    OpenAI SDK, with its response-populated status omitted for the live endpoint.
    Its optional content is omitted only when null. Native message phases and
    all call IDs, arguments, summaries and encrypted reasoning remain unchanged.
    Unknown fields fail closed rather than silently deleting future context.
    """
    projected = deepcopy(items)
    allowed_reasoning = {'type', 'id', 'summary', 'content', 'encrypted_content', 'status'}
    for item in projected:
        if item.get('type') == 'reasoning':
            if set(item) - allowed_reasoning:
                raise ValueError('Unknown native reasoning fields in transport input')
            item.pop('status', None)
            if item.get('content') is None:
                item.pop('content', None)
        elif item.get('type') == 'function_call':
            # Status is response lifecycle metadata, not part of the action.
            item.pop('status', None)
    return projected
