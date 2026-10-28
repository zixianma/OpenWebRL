"""Pinned native Arbiter viewer policy; BrowserGym execution is an adapter.

The native action parser is executed unchanged with recording constructors.
Those records preserve parser grammar, not WebArena action arrays or execution.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from weakref import WeakKeyDictionary

from .webprm_source import ARBITER_REVISION, _extract, source_manifest


class ActionParsingError(ValueError):
    pass


def load_arbiter_harness(root, *, expected_revision=ARBITER_REVISION):
    root = Path(root)
    prefix = 'viewer/src/'
    files = [prefix + p for p in ['prompts/sample_p.py',
             'server/core/tools/policy.py', 'browser_env/actions.py',
             'browser_env/constants.py', 'prompts/observation_p.py',
             'prompts/terminal_p.py', 'prompts/prompt_builder.py',
             'server/core/tools/state_diff.py', 'server/core/tools/terminal.py',
             'server/engine.py']]
    manifest = source_manifest(root, files, expected_revision)
    ns = {'__name__': __name__}
    _extract(root / files[0], ns, names={'render_prompt'}, constants=True)
    policy_ns = {'json': json, 'logger': logging.getLogger(__name__),
                 'BlockInfo': SimpleNamespace, 'PolicyGenerationResult': SimpleNamespace}
    tree = ast.parse((root / files[1]).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PolicyGenerator')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_parse_response')
    pure_cls = ast.ClassDef(name='PurePolicy', bases=[], keywords=[], body=[method], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pure_cls], type_ignores=[])),
                 str(root / files[1]), 'exec'), policy_ns)
    action_ns = {'re': re, 'Action': dict, 'ActionParsingError': ActionParsingError}
    names = ('click', 'clear', 'hover', 'type', 'key_press', 'scroll', 'goto_url',
             'new_tab', 'go_back', 'go_forward', 'page_focus', 'page_close', 'send_message_to_user')
    for name in names:
        def record(*args, _kind=name, **kwargs):
            return {'kind': _kind, 'args': args, 'kwargs': kwargs}
        action_ns['create_' + name + '_action'] = record
    action_ns['Page'] = object
    _extract(root / files[2], action_ns, names={'create_id_based_action', 'execute_scroll', 'execute_key_press'})
    constants_tree = ast.parse((root / files[3]).read_text())
    keys = next(n for n in constants_tree.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == 'SPECIAL_KEY_MAPPINGS' for t in n.targets))
    key_mapping = ast.literal_eval(keys.value)
    # The released viewer's sampler intentionally cannot stop. Its post-action
    # path computes a state difference and asks a separate terminal detector.
    # Keep these prompts and the tolerant terminal parser exactly upstream.
    diff_ns = {}
    _extract(root / files[4], diff_ns, names={'render_diff_user_prompt'}, constants=True)
    terminal_ns = {'json': json, 're': re, 'BlockInfo': SimpleNamespace,
                   'TerminalDetectionResult': lambda **kw: SimpleNamespace(
                       terminal_action=kw.get('terminal_action'))}
    _extract(root / files[5], terminal_ns, names={'render_terminal_prompt'}, constants=True)
    terminal_ns['terminal_system_text'] = terminal_ns['system_text']
    _extract(root / files[6], terminal_ns, names={'TerminalPromptBuilder'})
    terminal_tree = ast.parse((root / files[8]).read_text())
    terminal_cls = next(n for n in terminal_tree.body
                        if isinstance(n, ast.ClassDef) and n.name == 'TerminalDetector')
    terminal_method = next(n for n in terminal_cls.body
                           if isinstance(n, ast.FunctionDef) and n.name == '_parse_detection_result')
    parser_cls = ast.ClassDef(name='PureTerminal', bases=[], keywords=[],
                             body=[terminal_method], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[parser_cls], type_ignores=[])),
                 str(root / files[8]), 'exec'), terminal_ns)
    return SimpleNamespace(manifest=manifest, policy_system_prompt=ns['intro'],
                           render_policy_prompt=ns['render_prompt'],
                           parse_policy_response=policy_ns['PurePolicy']()._parse_response,
                           parse_action=action_ns['create_id_based_action'],
                           key_mapping=key_mapping,
                           execute_scroll=action_ns['execute_scroll'],
                           execute_key_press=action_ns['execute_key_press'],
                           state_diff_system_prompt=diff_ns['system_text_diff'],
                           render_state_diff_prompt=diff_ns['render_diff_user_prompt'],
                           build_terminal_prompt=terminal_ns['TerminalPromptBuilder'].build_terminal_prompt,
                           parse_terminal_response=terminal_ns['PureTerminal']()._parse_detection_result)


def to_browsergym_actions(parsed, id_map=None, *, key_mapping=None):
    """Translate native parser records; preserve type's mandatory Enter.

    AXTree observation IDs must map to BrowserGym bids. Scroll is a special
    native dispatch marker; callers must execute it using native_scroll_code.
    """
    kind, args, kw = parsed['kind'], parsed['args'], parsed['kwargs']
    def bid():
        element_id = kw.get('element_id')
        if not element_id:
            raise ActionParsingError('Accessible-name click requires explicit element resolution')
        if id_map is None:
            return str(element_id)
        if str(element_id) not in id_map:
            raise ActionParsingError('Unknown observation element ID')
        return str(id_map[str(element_id)])
    if kind in {'click', 'hover'}:
        return [f'{kind}({bid()!r})']
    if kind == 'clear':
        return [f'fill({bid()!r}, "")']
    if kind == 'type':
        text = kw['text']
        if not text.endswith('\n'):
            raise ActionParsingError('Native type parser did not append Enter')
        return [f'fill({bid()!r}, {text[:-1]!r})', f'press({bid()!r}, "Enter")']
    if kind == 'key_press':
        combo = '+'.join((key_mapping or {}).get(k.lower(), k) for k in kw['key_comb'].split('+'))
        return [f'keyboard_press({combo!r})']
    if kind == 'scroll':
        return [f'native_scroll({kw["direction"]!r})']
    if kind == 'goto_url':
        return [f'goto({kw["url"]!r})']
    if kind == 'page_focus':
        return [f'tab_focus({args[0]!r})']
    if kind == 'page_close':
        return ['tab_close()']
    if kind == 'send_message_to_user':
        return [f'send_msg_to_user({args[0]!r})']
    if kind in {'new_tab', 'go_back', 'go_forward'}:
        return [f'{kind}()']
    raise ActionParsingError(f'Unsupported native action: {kind}')


def native_scroll_code(root, direction):
    """Return exact upstream scroll body for BrowserGym's page execution scope.

    This bypasses BrowserGym's mouse-wheel scroll implementation. The caller
    must pin/check the checkout via load_arbiter_harness before executing it.
    """
    if direction not in {'up', 'down'}:
        raise ActionParsingError('Invalid scroll direction')
    path = Path(root) / 'viewer/src/browser_env/actions.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_scroll')
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    return ast.unparse(node) + '\nexecute_scroll(' + repr(direction) + ', page)\n'


def capture_element_bounds(page, id_map):
    """Capture viewport bounds before actor inference; no locator auto-scroll.

    Unlike WebArena's backend-node IDs, this resolves BrowserGym bid attributes.
    Missing/ambiguous targets remain unavailable rather than guessed.
    """
    result = {}
    for element_id, bid in id_map.items():
        selector = '[bid=' + json.dumps(str(bid)) + ']'
        matches = [frame.locator(selector) for frame in page.frames]
        matches = [locator for locator in matches if locator.count()]
        if len(matches) != 1 or matches[0].count() != 1:
            continue
        box = matches[0].bounding_box()
        if box:
            result[str(element_id)] = [box['x'], box['y'], box['width'], box['height']]
    return result


def native_execution_code(root, parsed, element_bounds, *, key_mapping=None):
    """Generate native coordinate/keyboard dispatch in BrowserGym's page scope.

    Pure upstream execution helper bodies are extracted unchanged. Bounds are
    frozen pre-inference viewport rectangles. Pageshow bridges native tab focus
    into BrowserGym's active-page callback; observation extraction stays adapted.
    """
    path = Path(root) / 'viewer/src/browser_env/actions.py'
    tree = ast.parse(path.read_text())
    helpers = {'execute_scroll', 'execute_key_press', 'execute_mouse_click',
               'execute_mouse_hover', 'execute_keyboard_type'}
    parts = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in helpers:
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            parts.append(ast.unparse(node))
    parts.append('_native_page_before = page\n_native_viewport_before = dict(page.viewport_size)\n_native_context = page.context\n_native_tabs_before = len(_native_context.pages)')
    kind, args, kw = parsed['kind'], parsed['args'], parsed['kwargs']
    if kind in {'click', 'hover', 'clear', 'type'}:
        eid = kw.get('element_id')
        if not eid or str(eid) not in element_bounds:
            raise ActionParsingError('Native coordinate target lacks captured viewport bounds')
        x, y, width, height = element_bounds[str(eid)]
        center = f'({x + width / 2!r}) / page.viewport_size["width"], ({y + height / 2!r}) / page.viewport_size["height"]'
        parts.append(f'execute_mouse_{"hover" if kind == "hover" else "click"}({center}, page)')
        if kind in {'clear', 'type'}:
            parts.extend(['execute_key_press("Meta+A", page)', 'execute_key_press("Backspace", page)'])
        if kind == 'type':
            text = kw['text']
            if not text.endswith('\n'):
                raise ActionParsingError('Native type parser did not append Enter')
            if text[:-1]:
                parts.append(f'execute_keyboard_type({text[:-1]!r}, page)')
            parts.append('execute_key_press("Enter", page)')
    elif kind == 'scroll':
        parts.append(f'execute_scroll({kw["direction"]!r}, page)')
    elif kind == 'key_press':
        combo = '+'.join((key_mapping or {}).get(k.lower(), k) for k in kw['key_comb'].split('+'))
        parts.append(f'execute_key_press({combo!r}, page)')
    elif kind == 'goto_url':
        parts.append(f'page.goto({kw["url"]!r})')
    elif kind in {'go_back', 'go_forward'}:
        parts.append(f'page.{kind}()')
    elif kind == 'new_tab':
        parts.append('page = _native_context.new_page()')
    elif kind == 'page_focus':
        parts.extend([f'page = _native_context.pages[{args[0]!r}]', 'page.bring_to_front()'])
    elif kind == 'page_close':
        parts.extend(['page.close()', 'page = _native_context.pages[-1] if _native_context.pages else _native_context.new_page()'])
    elif kind == 'send_message_to_user':
        parts.append(f'send_message_to_user({args[0]!r})')
    else:
        raise ActionParsingError('Unsupported native dispatch action')
    parts.extend(['page.wait_for_timeout(0)',
                  'if len(_native_context.pages) > _native_tabs_before:\n    page = _native_context.pages[-1]\n    page.bring_to_front()'])
    # BrowserGym observes real navigation/input events itself. Only a different
    # page object needs this bridge: action execution keeps its local `page`.
    # Evaluating JS after every click races navigation and turns successful
    # native actions into spurious "Execution context was destroyed" errors.
    # Sites can open explicitly sized popup windows. Keep newly selected pages
    # under the same fixed viewport contract as their originating observation;
    # leave same-page mismatches for the observation guard to reject.
    parts.append('if page is not _native_page_before:\n'
                 '    if page.viewport_size != _native_viewport_before:\n'
                 '        page.set_viewport_size(_native_viewport_before)\n'
                 '    for _activation_attempt in range(3):\n'
                 '        try:\n'
                 '            page.wait_for_load_state("domcontentloaded")\n'
                 '            page.evaluate("window.dispatchEvent(new Event(\'pageshow\', {bubbles:true,cancelable:false}));")\n'
                 '            break\n'
                 '        except Exception as _activation_error:\n'
                 '            if "Execution context was destroyed" not in str(_activation_error) or _activation_attempt == 2:\n'
                 '                raise\n')
    return '\n'.join(parts) + '\n'


def load_native_observation_processor(root, viewport_size, *, current_viewport_only=True):
    """Run upstream AXTree/CDP observation extraction against a live page.

    IDs and viewport rectangles are native, eliminating bid/tree translation.
    BrowserGym screenshots reset CDP screen dimensions to headless defaults;
    restore them to the actual viewport before native screen-based filtering.
    BrowserGym still owns page lifecycle, task reset and terminal validation.
    """
    root = Path(root)
    files = ['viewer/src/browser_env/processors.py', 'viewer/src/browser_env/constants.py']
    manifest = source_manifest(root, files, ARBITER_REVISION)
    constants = ast.parse((root / files[1]).read_text())
    ignored = next(n for n in constants.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'IGNORED_ACTREE_PROPERTIES' for t in n.targets))
    tree = ast.parse((root / files[0]).read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
                n.name in {'ObservationProcessor', 'create_empty_metadata', 'TextObervationProcessor'}]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    ns = {'re': re, 'defaultdict': defaultdict, 'IN_VIEWPORT_RATIO_THRESHOLD': .6,
          'IGNORED_ACTREE_PROPERTIES': ast.literal_eval(ignored.value)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])),
                 str(root / files[0]), 'exec'), ns)
    processor = ns['TextObervationProcessor']('accessibility_tree', current_viewport_only, viewport_size)
    last_geometry = {}
    clients = WeakKeyDictionary()
    def observe(page):
        client = clients.get(page)
        if client is None:
            client = page.context.new_cdp_session(page)
            clients[page] = client
        try:
            if page.viewport_size != viewport_size:
                raise ValueError('Native observation viewport differs from BrowserGym page viewport')
            # Supplying width/height alone resets screen to 800x600 in this
            # headless Chromium. The native processor filters by screen while
            # the rendered viewport is 1280x720, hiding real right/bottom UI.
            # BrowserGym screenshot extraction repeats that reset every step,
            # so restoration must happen at each native observation.
            # Chromium caches the last override per CDP session. Clear that
            # cache first: replaying identical parameters after a different
            # screenshot session reset them otherwise becomes a no-op.
            client.send('Emulation.clearDeviceMetricsOverride')
            client.send('Emulation.setDeviceMetricsOverride', dict(
                width=viewport_size['width'], height=viewport_size['height'],
                screenWidth=viewport_size['width'], screenHeight=viewport_size['height'],
                deviceScaleFactor=1, mobile=False))
            dimensions = page.evaluate('({screenWidth:screen.width,screenHeight:screen.height,'
                                       'innerWidth:innerWidth,innerHeight:innerHeight,'
                                       'devicePixelRatio:devicePixelRatio})')
            if (dimensions['screenWidth'] != viewport_size['width']
                    or dimensions['screenHeight'] != viewport_size['height']
                    or dimensions['innerWidth'] != viewport_size['width']
                    or dimensions['innerHeight'] != viewport_size['height']
                    or dimensions['devicePixelRatio'] != 1):
                raise ValueError('Native observation screen and viewport could not be aligned')
            last_geometry.clear()
            last_geometry.update(dimensions)
            text = processor.process(page, client)
            bounds = {eid: info['union_bound'] for eid, info in processor.obs_nodes_info.items()
                      if info.get('union_bound') is not None}
            return text, bounds
        except Exception:
            clients.pop(page, None)
            try:
                client.detach()
            except Exception:
                pass
            raise
    def close():
        for client in list(clients.values()):
            try:
                client.detach()
            except Exception:
                pass
        clients.clear()
    # CDP emulation is reverted on detach. Keep each page's session alive so
    # aligned geometry also survives inference and native action execution.
    # Browser/context teardown owns the sessions; close() permits early release.
    return SimpleNamespace(observe=observe, processor=processor, manifest=manifest,
                           last_geometry=last_geometry, close=close)
