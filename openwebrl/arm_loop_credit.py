"""Loop-focused action credit for mixed-only relaxed-B ARM runs (idea 4, 2026-11-04).

Both arms share one collection protocol and differ only in the per-turn credit u (advantage A + 0.5*u):

* Collection: the frozen executed-turn label hook (K=5, q=0.2, min2 gate, response-index credit), plus every
  turn whose executed action repeats an earlier action of the same trajectory is labelled (sampled with
  probability 1). Each executed turn records whether it repeats, and labelled turns record whether their five
  candidates mix repeated and new actions.
* ``loop_bonus`` (A): the original ARM unit bonus u = 1[ARM picks the executed response] - 0.2, kept only on
  loop-relevant turns (executed repeat, or mixed candidates); 0 elsewhere.
* ``repeat_penalty`` (B, no ARM in the reward): u = -0.2 on every executed repeat (the value the ARM bonus gives
  a rejected action), 0 elsewhere. ARM labels are still collected, identically, for telemetry only.

A repeat is the first tool call (name and arguments; wait/done ignored) already issued earlier in the
trajectory's visible history, exactly as in scripts/analyze_arm_label_repeats.py. The visible history window can
miss very old repeats; this only undercounts repeats.
"""
import json
import re

TOOL_CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)
SKIP = ('wait', 'done')
LOOP_MODES = ('loop_bonus', 'repeat_penalty')
REPEAT_UNIT = -.2
_REGISTRY = {}
_LOOP_CLASS = None


def norm(call):
    if not isinstance(call, dict) or call.get('name') in SKIP or not call.get('name'):
        return None
    return json.dumps(dict(name=call['name'], arguments=call.get('arguments') or {}), sort_keys=True)


def first_call(text):
    m = TOOL_CALL.search(str(text))
    if not m:
        return None
    try:
        return norm(json.loads(m.group(1)))
    except json.JSONDecodeError:
        return None


def first_action(action):
    """First call of a canonical action list (the label's saved ``actions`` entries)."""
    try:
        calls = json.loads(action) if isinstance(action, str) else action
    except json.JSONDecodeError:
        return None
    return norm(calls[0]) if isinstance(calls, list) and calls else None


def prior_calls(history):
    return {x for x in (first_call(h) for h in history or []) if x}


def loop_flags(record):
    """executed_repeat for every executed turn; mixed_candidates for labelled turns."""
    prior = set(record.get('prior_calls') or ())
    flags = dict(executed_repeat=bool(record.get('executed_repeat')))
    actions = record.get('actions')
    if actions:
        rep = [a is not None and a in prior for a in map(first_action, actions)]
        flags['mixed_candidates'] = any(rep) and not all(rep)
    else:
        flags['mixed_candidates'] = False
    flags['loop_relevant'] = flags['executed_repeat'] or flags['mixed_candidates']
    return flags


def selector_class():
    from openwebrl.arm_turn_bonus import ShadowSelector

    class LoopShadowSelector(ShadowSelector):
        """Frozen label hook; an executed repeat is always sampled (subject to the same queue bound)."""
        def __init__(self, config, trajectory_id, **kw):
            super().__init__(config, trajectory_id, **kw)
            self.config = dict(self.config)  # per-trajectory copy; scored_fraction is switched per turn
            self.base_fraction = config['scored_fraction']
            _REGISTRY[self.trajectory_id] = self

        async def __call__(self, *, infer, url, input_text, sampling_params, images,
                           observation, history, task, task_id, turn, timeout):
            prior = prior_calls(history)
            seen = dict(first=True, repeat=False)

            async def first_infer(*a, **kw):
                out = await infer(*a, **kw)
                if seen['first']:
                    seen['first'] = False
                    seen['repeat'] = (call := first_call(out[0])) is not None and call in prior
                    # The parent decides sampling right after the executed inference.
                    self.config['scored_fraction'] = 1. if seen['repeat'] else self.base_fraction
                return out
            try:
                result = await super().__call__(infer=first_infer, url=url, input_text=input_text,
                    sampling_params=sampling_params, images=images, observation=observation, history=history,
                    task=task, task_id=task_id, turn=turn, timeout=timeout)
            finally:
                self.config['scored_fraction'] = self.base_fraction
            self.records[turn].update(executed_repeat=seen['repeat'], prior_calls=sorted(prior))
            return result
    return LoopShadowSelector


async def generate(args, sample, sampling_params, evaluation=False):
    """Frozen label hook with the loop selector; attaches loop flags to every executed turn."""
    global _LOOP_CLASS
    from openwebrl import arm_turn_bonus as parent
    if _LOOP_CLASS is None:
        _LOOP_CLASS = selector_class()
    # Installed once for the whole process (only loop runs import this hook); restoring it around each await
    # would race with concurrently running trajectories.
    parent.ShadowSelector = _LOOP_CLASS
    turns = await parent.generate(args, sample, sampling_params, evaluation=evaluation)
    selector = _REGISTRY.pop(str(sample.index), None)
    if selector is None:
        return turns  # evaluation-overlap task: the parent ran it without a selector
    for t in turns:
        meta = t.metadata or {}
        record = selector.records.get(meta.get('turn_index'))
        if record is not None:
            t.metadata = dict(meta, arm_loop=loop_flags(record))
    return turns


def loop_unit(sample, mode, arm_unit):
    """Per-turn u for the loop modes; arm_unit is the validated original ARM unit bonus of the turn."""
    if mode not in LOOP_MODES:
        raise ValueError(mode)
    flags = (sample.metadata or {}).get('arm_loop')
    if flags is None:
        return 0.
    if mode == 'repeat_penalty':
        return REPEAT_UNIT if flags['executed_repeat'] else 0.
    return arm_unit if flags['loop_relevant'] else 0.
