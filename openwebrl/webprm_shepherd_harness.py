"""Capture exact author demo BrowserGym actor messages without an API call.

This public demo supplies the vanilla actor prompt, not the paper search loop.
Its file hash and BrowserGym action description belong in the run manifest.
"""
from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace


SHEPHERD_ACTION_SUBSETS = ('chat', 'tab', 'nav', 'bid', 'infeas')


def parse_shepherd_response(raw_text, action_set):
    """Canonicalize one response using the author's declared actor action set.

    The demo declares ``HighLevelActionSet(strict=False, multiaction=False)``
    for its actor prompt. Its standalone environment omits this action mapping
    and inherits BrowserGym's multi-action default; this adapter deliberately
    enforces the declared single-action contract. Code fences do not take
    precedence over action-looking prose: two parsed calls remain invalid.
    Parsing here does not execute generated code or repair actor responses.
    """
    from browsergym.core.action.highlevel import highlevel_action_parser

    if action_set.strict or action_set.multiaction:
        raise ValueError('Shepherd requires non-strict, single-action BrowserGym parsing')
    if not isinstance(raw_text, str):
        raise ValueError('Shepherd response must be text')
    # Validate the whole response with the exact execution parser first. This
    # retains native errors for missing, multiple and unsupported action calls.
    action_set.to_python_code(raw_text)
    calls = sum(highlevel_action_parser.search_string(raw_text).as_list(), [])
    name, arguments = calls[0]
    action = name + '(' + ', '.join(repr(argument) for argument in arguments) + ')'
    return dict(thought=raw_text, raw_response=raw_text, action=action)


def classify_shepherd_choices(choices, action_set):
    """Classify every fixed-draw sample independently, preserving rejections.

    This is an explicit candidate-handling policy for a new protocol, not a
    reproduction of an unpublished author search loop. It neither resamples nor
    guarantees five distinct candidates. A caller must handle an empty accepted
    set explicitly. Original response indices remain available for audit even
    though accepted candidates form a compact list for frequency ranking.
    """
    from pyparsing import ParseBaseException

    accepted, rejected = [], []
    for sample_index, choice in enumerate(choices):
        raw_text = choice.get('message', {}).get('content')
        metadata = dict(sample_index=sample_index, choice_index=choice.get('index', sample_index))
        try:
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Incomplete actor response')
            parsed = parse_shepherd_response(raw_text, action_set)
        except (ValueError, NameError, SyntaxError, TypeError, ParseBaseException) as error:
            rejected.append(dict(metadata, raw_response=raw_text,
                                 finish_reason=choice.get('finish_reason'),
                                 error_type=type(error).__name__, error=str(error)))
        else:
            accepted.append(dict(parsed, **metadata))
    return accepted, rejected


def load_shepherd_actor(path, action_set, *, expected_sha256=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Unexpected author demo source hash')
    tree = ast.parse(raw.decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BrowserAgent')
    cls.bases = [ast.Name(id='object', ctx=ast.Load())]
    ns = {'logger': logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(path), 'exec'), ns)
    agent = ns['BrowserAgent'].__new__(ns['BrowserAgent'])
    agent.model_name = 'gpt-4o-mini'
    agent.use_html, agent.use_axtree, agent.use_screenshot = False, True, False
    agent.action_set, agent.action_history = action_set, []

    def messages(obs, history=(), model='gpt-4o-mini'):
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=''))])
        agent.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=capture)))
        agent.model_name, agent.action_history = model, list(history)
        agent.get_action(obs)
        return captured['messages']

    def preprocess(obs):
        # Use the same library functions as the author demo, not reconstructed text.
        from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
        ns.update(flatten_axtree_to_str=flatten_axtree_to_str,
                  flatten_dom_to_str=flatten_dom_to_str, prune_html=prune_html)
        return agent.obs_preprocessor(obs)

    return SimpleNamespace(messages=messages, preprocess=preprocess,
                           manifest={'source_sha256': digest, 'source_file': str(path),
                                     'action_description': action_set.describe(with_long_description=False, with_examples=True)})
