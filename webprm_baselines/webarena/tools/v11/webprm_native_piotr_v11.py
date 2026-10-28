"""v11 (2026-10-27): Piotr's released SelectionARM over the released WebArbiter coordinator's candidates, 1280x720 actor.

User (2026-10-27): "ah I see. yes do A. so then we can use gpt4o + our ARM right? no need for openwebrl-sft", then chose
the 1280x720 setup: the actor viewport is 1280x720 for both no_search (rerun) and piotr_selectionarm, and the ARM sees
exactly the actor's screenshot. (Copy of v10-tools/webprm_native_piotr_v10.py; v10 cropped the top 1024x576 of a
1024x2490 viewport.)
OM2W protocol on the native WebArena harness: gpt-4o-2024-08-06 proposes with the released WebArbiter policy prompt
(5 BLOCKS in one call, released default decoding T=1.0 / top_p=0.9); the released sampler filters and deduplicates; the
released Piotr SelectionARM (PTeterwak/OpenWebRL-4B-SelectionARM @ 81b452d) picks one from the pre-action screenshot,
grounded candidate descriptors and its 5-step history (representation native_operation_descriptor_preaction_status_v2,
frozen in webprm_webarena_piotr.py, used unchanged).
  frame       the native 1280x720 observation screenshot PNG, byte for byte (the actor's own capture; the only further
              resizing is the ARM processor view v3, max 262,144 pixels, inside the model server). AX union bounds are
              viewport CSS pixels = screenshot pixels, unscaled.
  pool size   1..5 distinct surviving candidates; schema maximum = pool size; one candidate executes without an ARM
              call; more than five survivors: the first five in released rerank order. (OM2W's viewer harness required
              exactly five admissible raw slots and invalidated the cell otherwise.)
  grounding   an action the frozen grounder cannot describe is shown as a canonical descriptor with target_status
              'ungroundable' instead of invalidating the cell; counts are recorded.
  history     every executed action is grounded from its own pre-action observation, with the selected thought.
No model is loaded here; the caller supplies the synchronous broker transport.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .webprm_arbiter_harness import load_arbiter_harness
from .webprm_webarena import EvaluationNotReady
from .webprm_webarena_piotr import (MODEL, REVISION, REPRESENTATION_V2, validate_piotr_selectionarm, load_prompt,
                                    ground_native_action, _history, _screenshot, canonical, digest)

NATIVE_VIEWPORT = (1280, 720)  # v11: actor viewport == ARM frame
VIEWPORT = {'width': 1280, 'height': 720}
MAX_CANDIDATES = 5
RECEIPT = dict(version='v11_native_piotr_selectionarm_1280x720_v1', model=MODEL, revision=REVISION, representation=REPRESENTATION_V2,
    policy='released WebArbiter 5-BLOCK policy prompt, one call, released default decoding (temperature 1.0, top_p 0.9)',
    candidates='released filter_candidates + deduplication + rerank; first 5; k=1 executes without an ARM call',
    frame=dict(native_viewport=list(NATIVE_VIEWPORT), arm_viewport=dict(VIEWPORT), image='native observation PNG bytes unchanged',
               crop=None, rescale='none (ARM processor view v3 only)'),
    selection='Piotr SelectionARM, temperature 0, max_tokens 128, JSON schema selection 1..k',
    terminal='released terminal detector short-circuits selection (as no_search and OM2W)',
    ungroundable='canonical descriptor target_status=ungroundable', history='last 5 executed actions, own pre-action grounding')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def frame_bounds(row):
    """AX union bounds (viewport CSS px == screenshot px) of a native observation capture, unscaled."""
    nodes = row.get('obs_nodes_info')
    if not isinstance(nodes, dict):
        raise EvaluationNotReady('Native observation lacks its AX node map')
    bounds = {}
    for key, node in nodes.items():
        value = node.get('union_bound') if isinstance(node, dict) else None
        if (isinstance(value, (list, tuple)) and len(value) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in value)):
            bounds[str(key)] = list(value)
    return bounds


def frame_png(row):
    """The captured native 1280x720 viewport screenshot PNG, byte for byte."""
    image = row.get('image')
    if not isinstance(image, dict) or set(image) != {'path', 'sha256'}:
        raise EvaluationNotReady('Native observation lacks its screenshot reference')
    path = Path(image['path'])
    data = path.read_bytes()
    if not path.is_absolute() or _sha(data) != image['sha256']:
        raise EvaluationNotReady('Native observation screenshot changed')
    _screenshot(data, VIEWPORT)  # frozen check: PNG, exactly 1280x720
    return data


def selected_parts(decision):
    selected = decision.get('selected_action') if isinstance(decision, dict) else None
    get = (lambda k: selected.get(k)) if isinstance(selected, dict) else (lambda k: getattr(selected, k, None))
    action, thought = get('action'), get('thought')
    return (action if isinstance(action, str) else None), (thought if isinstance(thought, str) else '')


class NativePiotrSelector:
    """Selection over the released coordinator's pool plus the executed-action history Piotr's prompt reads."""

    def __init__(self, config, journal, *, request):
        self.config, self.journal, self.request = deepcopy(config), journal, request
        manifest = validate_piotr_selectionarm(config)
        if config.get('piotr_manifest') != manifest or manifest['representation'] != REPRESENTATION_V2:
            raise EvaluationNotReady('Pinned v2-representation Piotr manifest required')
        self.builder = load_prompt(config['piotr_source'])
        self.harness = load_arbiter_harness(config['arbiter_source'])
        self.history, self.pending, self.ungroundable = [], None, 0

    def ground(self, action, bounds):
        try:
            grounded = ground_native_action(action, parse_action=self.harness.parse_action, bounds=bounds,
                viewport=dict(VIEWPORT), key_mapping=self.harness.key_mapping, representation=REPRESENTATION_V2)
            return grounded['action'], 'grounded'
        except Exception as error:
            descriptor = dict(native_action=action, operation='unrecognized', target_status='ungroundable',
                              error_type=type(error).__name__)
            return canonical(descriptor).replace('(', '\\u0028').replace(')', '\\u0029'), 'ungroundable'

    def note_decision(self, decision):
        action, thought = selected_parts(decision)
        self.pending = dict(action=action, thought=thought) if action else None

    def note_execution(self, action, row):
        """Call with the selected action string and the pre-action observation, before the browser executes it."""
        display, status = self.ground(action, frame_bounds(row))
        self.ungroundable += status == 'ungroundable'
        thought = self.pending['thought'] if self.pending and self.pending['action'] == action else ''
        self.history.append(dict(action=action, thought=thought, selectionarm_action=display))
        self.pending = None

    def select(self, pool, context, row, turn):
        if not callable(self.request) or inspect.iscoroutinefunction(self.request):
            raise EvaluationNotReady('Explicit synchronous broker transport required for Piotr selector')
        if row.get('text') != context.get('observation_text'):
            raise EvaluationNotReady('Piotr frame is not the observation the policy saw')
        if any(not isinstance(context.get(k), str) for k in ('intent', 'current_url')):
            raise EvaluationNotReady('Actor-visible task and current URL required')
        pool = deepcopy(list(pool))
        considered = pool[:MAX_CANDIDATES]
        if not considered or any(not isinstance(c.get('action'), str) or not c['action'] for c in considered):
            raise EvaluationNotReady('Piotr pool requires candidate action strings')
        bounds = frame_bounds(row)
        grounded = [self.ground(c['action'], bounds) for c in considered]
        history = _history(self.history)
        trace = dict(selector='piotr_selectionarm', protocol=RECEIPT, turn=turn, pool_size=len(pool),
            considered=len(considered), truncated=len(pool) - len(considered),
            grounded_candidates=[g[0] for g in grounded], grounding_status=[g[1] for g in grounded],
            observation_image_sha256=row['image']['sha256'], bounds_sha256=digest(bounds), history_sha256=digest(history),
            history_length=len(self.history), ungroundable_history_actions=self.ungroundable)
        if len(considered) == 1:
            trace.update(selected_index=0, arm_called=False)
            return 0, trace
        png = frame_png(row)
        clusters = [dict(rep=SimpleNamespace(molmo_action=g[0], thought=c.get('thought') if isinstance(c.get('thought'), str) else ''),
                         vote_count=1, cluster_key=str(i)) for i, (c, g) in enumerate(zip(considered, grounded))]
        messages = self.builder(context['intent'], history, context['current_url'], clusters, png)
        payload = dict(model=MODEL, messages=messages, temperature=0, max_tokens=128,
            response_format={'type': 'json_schema', 'json_schema': {'name': 'vision_selection_nocot', 'strict': True,
                'schema': {'type': 'object', 'properties': {'selection': {'type': 'integer', 'minimum': 1,
                                                                          'maximum': len(considered)}},
                           'required': ['selection']}}})
        response = self.request(self.config, deepcopy(payload), self.journal, local=True,
                                endpoint=self.config.get('piotr_endpoint'), purpose='selector_piotr')
        if inspect.isawaitable(response):
            if inspect.iscoroutine(response):
                response.close()
            raise EvaluationNotReady('Piotr selector transport must be synchronous')
        if not isinstance(response, dict) or response.get('model') != MODEL:
            raise EvaluationNotReady('Piotr response model differs from frozen checkpoint route')
        choices = response.get('choices')
        if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                or choices[0].get('finish_reason') != 'stop' or not isinstance(choices[0].get('message'), dict)):
            raise EvaluationNotReady('Piotr selector response incomplete')
        raw = choices[0]['message'].get('content')
        try:
            verdict = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise EvaluationNotReady('Piotr selector returned malformed JSON; no first-candidate fallback') from error
        if (not isinstance(verdict, dict) or set(verdict) != {'selection'} or type(verdict['selection']) is not int
                or not 1 <= verdict['selection'] <= len(considered)):
            raise EvaluationNotReady('Piotr selection does not identify one considered candidate')
        index = verdict['selection'] - 1
        trace.update(selected_index=index, arm_called=True, raw_response=raw, frame_png_sha256=_sha(png),
                     prompt_sha256=digest(messages))
        return index, trace
