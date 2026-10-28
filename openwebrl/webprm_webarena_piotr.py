"""Piotr's released SelectionARM on unchanged GPT/WebArena candidate slots.

The released screenshot prompt cannot ground native AX IDs. This adapter gives
it an explicit, pre-action descriptor of each native operation, including its
normalized target point. Descriptors are scorer inputs only; the caller executes
the original selected action. This is an action-representation adaptation and
an additional visual selector, not an exact Table 5 method or a new trained head.
No model is loaded and no transport is supplied by default.
"""
from __future__ import annotations

import __future__
import ast
from copy import deepcopy
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import re
from types import MappingProxyType, SimpleNamespace

from .webprm_arbiter_harness import load_arbiter_harness
from .webprm_webarena import EvaluationNotReady

MODEL = 'PTeterwak/OpenWebRL-4B-SelectionARM'
REVISION = '81b452d800d9f859687074f82680dd5257e02d89'
CHECKPOINT = Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection') / REVISION
SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source')
SOURCE_REVISION = '02276b0ff3b9048d34e6a2afdcb9042dd9018c8c'
PROMPT_SHA = '043ab7e98127f07ccd76b0eb736837f8a089c1328fa211c4ca4c20d771a7751c'
INFER_SHA = 'a8710c9865542bbf1fd4ed93ab53a08c4735696e9700f2f21bae4b6eb975ec97'
METADATA = {
    'config.json': '0a7a20dc3671c5904310c84ddce467a1fe0da866cfb6d70590d6ee7f8339648a',
    'generation_config.json': 'f86380bf2ba7ef4d84074388aebd955e89f0a2dd142d938af3d201329f50b459',
    'processor_config.json': '01562eddd6f9e9ec4bc31656a3b7055284cafbf889acc6c4348dca431ae31f68',
    'tokenizer.json': 'be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506',
    'tokenizer_config.json': 'c55596e013e69474cf05276bda8c2f240d46b2c954fb06945ee86164af2188e6',
    'chat_template.jinja': '36e042fe45641f067b1f2381fcc8955d10d956a3ed333ecdf7f7eb0916f68956',
    'model.safetensors.index.json': '80b079b82833257f10da556e5d1a0351248ff6d11e4579debea9a0f489e77450',
}
WEIGHTS = {
    'model-00001-of-00002.safetensors': (4990822120, '469e219bd9e7031f9d7c04065d8677367f4645db78e61b6844f597aa2020e77f'),
    'model-00002-of-00002.safetensors': (3884897208, '91565365c5b4080211c895c0e555a39030c4a4828d711c2211a8227f1e4ce6b4'),
}
FLAGS = MappingProxyType({
    'CATTS_VISION_PROMPT_V2': '1', 'CATTS_VISION_COLORED': '1',
    'VISION_NO_SOM': '1', 'VISION_ABLATE_DOM': '1', 'VISION_ABLATE_VOTES': '1',
    'NORMALIZE_COORDS': '1', 'CLUSTER_NO_DOM': '1', 'VISION_NO_COT': '1',
    'VISION_NO_COT_THINK': '0', 'VISION_ABLATE_IMAGE': '0',
    'VISION_ABLATE_CAND_THOUGHTS': '0', 'VISION_ABLATE_HISTORY': '0',
    'VISION_ABLATE_TRAJ_THOUGHTS': '0', 'VISION_FULL_HISTORY': '0',
})
REPRESENTATION = 'native_operation_descriptor_normalized_point_v1'
REPRESENTATION_V2 = 'native_operation_descriptor_preaction_status_v2'


def representation(config):
    value = config.get('piotr_representation', REPRESENTATION)
    if value not in (REPRESENTATION, REPRESENTATION_V2):
        raise EvaluationNotReady('Unknown Piotr native action representation')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_piotr_selectionarm(config):
    """Cheap pinned metadata/source validation; does not read/load tensor bodies.

    LFS hashes below are cached download identities, not a fresh weight digest.
    The owned serving preflight must separately verify tensor content and the
    actual engine/processor/grammar contract before any benchmark execution.
    """
    if config.get('piotr_model') != MODEL or config.get('piotr_revision') != REVISION:
        raise EvaluationNotReady('Exact released Piotr SelectionARM model/revision required')
    checkpoint = Path(config.get('piotr_checkpoint', ''))
    if checkpoint != CHECKPOINT or checkpoint.resolve() != CHECKPOINT:
        raise EvaluationNotReady('Pinned local Piotr checkpoint path required')
    source = Path(config.get('piotr_source', ''))
    if not source.is_absolute() or source.resolve() != source:
        raise EvaluationNotReady('Canonical local Piotr source path required')
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('revision') != SOURCE_REVISION:
        raise EvaluationNotReady('Piotr source revision changed')
    for name, expected in [('inference/selection_prompt.py', PROMPT_SHA), ('inference/selection_infer.py', INFER_SHA)]:
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise EvaluationNotReady('Piotr source file changed: ' + name)
    for name, expected in METADATA.items():
        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != expected:
            raise EvaluationNotReady('Piotr checkpoint metadata changed: ' + name)
    weights = []
    for name, (size, etag) in WEIGHTS.items():
        path = checkpoint / name
        cached = (checkpoint / '.cache/huggingface/download' / (name + '.metadata')).read_text().splitlines()
        if path.stat().st_size != size or cached[:2] != [REVISION, etag]:
            raise EvaluationNotReady('Piotr cached weight identity changed: ' + name)
        weights.append(dict(file=name, bytes=size, cached_lfs_sha256=etag))
    return dict(model=MODEL, revision=REVISION, checkpoint=str(checkpoint), metadata_sha256=METADATA,
        weights=weights, weights_fresh_sha256_verified=False,
        source=str(source), source_revision=SOURCE_REVISION, prompt_sha256=PROMPT_SHA,
        inference_source_sha256=INFER_SHA, canonical_flags=dict(FLAGS),
        representation=representation(config), action_representation_adapted=True,
        additional_selector_modality='pre-action screenshot', history_window=5,
        selector_temperature=0, selector_max_tokens=128, context_limit=32768,
        processor_max_pixels=262144, runtime_model_loading_verified=False)


def load_prompt(source):
    """Compile unchanged public builder with private immutable environment flags."""
    path = Path(source) / 'inference/selection_prompt.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PROMPT_SHA:
        raise EvaluationNotReady('Piotr prompt source changed')
    names = {'_parse_click_xy', 'build_catts_vision_prompt_v2'}
    constants = {'CATTS_VISION_V2_SYSTEM_PROMPT_NOCOT', 'CATTS_VISION_V2_SYSTEM_PROMPT_NOCOT_THINK'}
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets):
            nodes.append(node)
    namespace = dict(re=re, os=SimpleNamespace(environ=FLAGS))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace['build_catts_vision_prompt_v2']


def _viewport(value):
    if value != {'width': 1280, 'height': 720} or any(type(value.get(k)) is not int for k in ('width', 'height')):
        raise EvaluationNotReady('Piotr pre-action viewport must be exactly 1280 by 720')
    return value


def _screenshot(value, viewport):
    if not isinstance(value, bytes) or len(value) > 16 * 1024 * 1024:
        raise EvaluationNotReady('Explicit bounded pre-action PNG bytes required')
    from PIL import Image
    try:
        with Image.open(io.BytesIO(value)) as image:
            if image.format != 'PNG' or image.size != (viewport['width'], viewport['height']):
                raise EvaluationNotReady('Pre-action PNG and native viewport dimensions differ')
            image.verify()
    except (OSError, ValueError) as error:
        raise EvaluationNotReady('Invalid pre-action screenshot') from error
    return hashlib.sha256(value).hexdigest()


def ground_native_action(raw, *, parse_action, bounds, viewport, key_mapping, representation=REPRESENTATION):
    """Describe exact native effects; never produce an executable replacement."""
    if representation not in (REPRESENTATION, REPRESENTATION_V2):
        raise EvaluationNotReady('Unknown Piotr native action representation')
    v2 = representation == REPRESENTATION_V2
    parsed = parse_action(raw)
    kind, args, kw = parsed['kind'], parsed['args'], parsed['kwargs']
    descriptor = dict(native_action=raw, operation=kind)
    grounding = None
    if kind in ('click', 'hover', 'clear', 'type'):
        eid = kw.get('element_id')
        missing = not eid or str(eid) not in bounds
        rectangle = bounds.get(str(eid)) if not missing else None
        if v2 and missing:
            descriptor.update(target_element_id=str(eid), target_status='unavailable', point_2d=None)
            grounding = dict(element_id=str(eid), target_status='unavailable',
                             bounds_css_pixels=None, native_center_css_pixels=None,
                             normalized_point_2d=None, viewport=dict(viewport))
        else:
            if (not isinstance(rectangle, (list, tuple)) or len(rectangle) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in rectangle)):
                raise EvaluationNotReady('Native candidate lacks frozen numeric target bounds')
            x, y, width, height = rectangle
            center = [x + width / 2, y + height / 2]
            in_view = 0 <= center[0] < viewport['width'] and 0 <= center[1] < viewport['height']
            if width <= 0 or height <= 0:
                raise EvaluationNotReady('Native target geometry must have positive dimensions' if v2
                                         else 'Native target center is outside the captured viewport')
            if not v2 and not in_view:
                raise EvaluationNotReady('Native target center is outside the captured viewport')
            point = [1000 * center[0] / viewport['width'], 1000 * center[1] / viewport['height']] if in_view else None
            grounding = dict(element_id=str(eid), bounds_css_pixels=list(rectangle),
                             native_center_css_pixels=center, normalized_point_2d=point, viewport=dict(viewport))
            descriptor.update(point_2d=point, coordinate_space='normalized_0_1000')
            if v2:
                status = 'in_viewport' if in_view else 'center_outside_viewport'
                descriptor.update(target_element_id=str(eid), target_status=status)
                grounding['target_status'] = status
                if not in_view:
                    descriptor.pop('coordinate_space')
                    descriptor.update(observed_bounds_css_pixels=list(rectangle),
                                      observed_center_css_pixels=center)
        if kind in ('clear', 'type'):
            descriptor.update(focus='click_target', clear_existing=True,
                              clear_keys='Meta+A (Control+A on non-Mac), then Backspace')
        if kind == 'type':
            if not kw['text'].endswith('\n'):
                raise EvaluationNotReady('Native type must retain mandatory Enter')
            descriptor.update(text=kw['text'][:-1], press_enter=True)
    elif kind == 'scroll':
        descriptor.update(direction=kw['direction'], target='document.scrollingElement || document.body',
                          amount='one window.innerHeight', mechanism='assign scrollTop')
    elif kind == 'key_press':
        descriptor.update(keys='+'.join(key_mapping.get(k.lower(), k) for k in kw['key_comb'].split('+')),
                          platform_rule='Meta becomes Control on non-Mac')
    elif kind == 'goto_url':
        descriptor['url'] = kw['url']
    elif kind == 'go_back' or v2 and kind == 'go_forward':
        descriptor['mechanism'] = 'browser page.' + kind
        if v2:
            descriptor['destination_state'] = 'not_observed_pre_action'
    elif v2 and kind == 'new_tab':
        descriptor.update(mechanism='browser context.new_page', activation='new_page_bring_to_front',
                          destination_state='not_observed_pre_action')
    elif v2 and kind == 'page_focus':
        descriptor.update(mechanism='context.pages[index].bring_to_front', tab_index=args[0],
                          index_semantics='native_python_context_pages_index',
                          destination_state='not_observed_pre_action')
    elif v2 and kind == 'page_close':
        descriptor.update(mechanism='close_current_page',
                          next_page_rule='last_remaining_context_page_else_new_page',
                          destination_state='not_observed_pre_action')
    elif v2 and kind == 'send_message_to_user':
        descriptor.update(mechanism='send_message_to_user', message=args[0])
    else:
        raise EvaluationNotReady('Piotr native action representation unsupported: ' + kind)
    # The public builder rescales any textual click(x,y), even inside strings.
    # JSON escapes preserve arbitrary typed text without that unintended rewrite.
    display = canonical(descriptor).replace('(', '\\u0028').replace(')', '\\u0029')
    return dict(action=display, descriptor=descriptor, grounding=grounding,
                raw_action_sha256=hashlib.sha256(raw.encode()).hexdigest())


def _history(rows):
    result = []
    for row in rows[-5:]:
        if not isinstance(row, dict) or not isinstance(row.get('thought'), str):
            raise EvaluationNotReady('Explicit prior thought/action history required')
        action = row.get('selectionarm_action')
        try:
            decoded = json.loads(action)
        except (TypeError, ValueError) as error:
            raise EvaluationNotReady('Prior action lacks its own saved pre-action grounding') from error
        if not isinstance(decoded, dict) or decoded.get('native_action') != row.get('action'):
            raise EvaluationNotReady('Saved historical grounding belongs to a different action')
        result.append(dict(action=action, thought=row['thought']))
    return result


class PiotrSelectionARMSelector:
    """One canonical visual selector call over exactly five unchanged raw slots."""
    def __init__(self, config, journal, *, request=None):
        self.config, self.journal, self.request = deepcopy(config), journal, request
        self.manifest = validate_piotr_selectionarm(config)
        self.builder = load_prompt(config['piotr_source'])
        self.harness = load_arbiter_harness(config['arbiter_source'])

    def historical_action(self, raw, *, viewport, element_bounds):
        """Grounded history display for an executed action that bypassed selection.

        Terminal-detector fallbacks execute a native action without a Piotr call;
        later Piotr prompts still need that step in the frozen history format.
        Uses the same pre-action viewport/bounds grounding as candidates.
        """
        if not isinstance(element_bounds, dict):
            raise EvaluationNotReady('Malformed causal grounding for historical action')
        return ground_native_action(raw, parse_action=self.harness.parse_action, bounds=element_bounds,
            viewport=_viewport(viewport), key_mapping=self.harness.key_mapping,
            representation=self.manifest['representation'])['action']

    def select(self, proposal, turn, *, evidence):
        if not callable(self.request) or inspect.iscoroutinefunction(self.request):
            raise EvaluationNotReady('Explicit synchronous broker transport required for Piotr selector')
        pool = proposal.get('candidates')
        if (proposal.get('terminal') or proposal.get('exact_five_raw_slots') is not True
                or proposal.get('raw_slot_count') != 5 or not isinstance(pool, list) or len(pool) != 5
                or [row.get('raw_slot') for row in pool] != list(range(5))):
            raise EvaluationNotReady('Piotr requires the original five valid raw candidate slots; no padding or resampling')
        if set(evidence) != {'screenshot', 'viewport', 'element_bounds', 'history'}:
            raise EvaluationNotReady('Only explicit pre-action screenshot, viewport, bounds and history may enter Piotr evidence')
        viewport = _viewport(evidence['viewport'])
        screenshot_sha = _screenshot(evidence['screenshot'], viewport)
        bounds = evidence['element_bounds']
        if not isinstance(bounds, dict) or not isinstance(evidence['history'], list):
            raise EvaluationNotReady('Malformed causal grounding/history')
        history = _history(evidence['history'])
        prepared = []
        for row in pool:
            if any(not isinstance(row.get(name), str) for name in ('action', 'thought')):
                raise EvaluationNotReady('Original candidate thought and action must be strings')
            prepared.append(ground_native_action(row['action'], parse_action=self.harness.parse_action,
                bounds=bounds, viewport=viewport, key_mapping=self.harness.key_mapping,
                representation=self.manifest['representation']))
        context = proposal['context']
        if any(not isinstance(context.get(name), str) for name in ('intent', 'current_url')):
            raise EvaluationNotReady('Actor-visible task and current URL required')
        clusters = [dict(rep=SimpleNamespace(molmo_action=grounded['action'], thought=row['thought']),
                         vote_count=1, cluster_key=str(row['raw_slot'])) for row, grounded in zip(pool, prepared)]
        messages = self.builder(context['intent'], history, context['current_url'], clusters, evidence['screenshot'])
        payload = dict(model=MODEL, messages=messages, temperature=0, max_tokens=128,
            response_format={'type': 'json_schema', 'json_schema': {'name': 'vision_selection_nocot', 'strict': True,
                'schema': {'type': 'object', 'properties': {'selection': {'type': 'integer', 'minimum': 1, 'maximum': 5}},
                           'required': ['selection']}}})
        before = digest(pool)
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
        raw = choices[0].get('message', {}).get('content')
        try:
            verdict = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise EvaluationNotReady('Piotr selector returned malformed JSON; no first-candidate fallback') from error
        if (not isinstance(verdict, dict) or set(verdict) != {'selection'}
                or type(verdict['selection']) is not int or not 1 <= verdict['selection'] <= 5
                or digest(pool) != before):
            raise EvaluationNotReady('Piotr selection does not identify one original candidate')
        index = verdict['selection'] - 1
        trace = dict(selector='piotr_selectionarm', model=MODEL, revision=REVISION, turn=turn,
            representation=self.manifest['representation'], selected_index=index, selected_raw_slot=pool[index]['raw_slot'],
            selected_grounded_action=prepared[index]['action'], candidates_sha256=before,
            screenshot_sha256=screenshot_sha, viewport=dict(viewport), history_sha256=digest(history),
            prompt_sha256=digest(messages), native_bounds_sha256=digest(bounds),
            grounded_candidates=prepared, raw_response=raw,
            source_prompt_sha256=PROMPT_SHA, action_execution='original native candidate unchanged')
        return index, trace
