"""Load pinned upstream pure inference code without heavyweight imports.

AST extraction executes upstream Python: accept only a reviewed checkout and pin
its revision and file hashes in an experiment manifest. No model or API calls.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from abc import ABC, abstractmethod
from types import SimpleNamespace

SHEPHERD_REVISION = 'b87898f614e43d7a5bdb738eaf6b01d416be24c2'
ARBITER_REVISION = '764ec2670750c50388bac56efb1a3989fd5c1b54'


def source_manifest(root, paths, expected_revision=None):
    root = Path(root).resolve()
    revision = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(f'Unexpected upstream revision: {revision}')
    return {'revision': revision, 'files': {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in paths}}


def _extract(path, namespace, *, names=None, constants=False):
    tree = ast.parse(Path(path).read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if names is None or node.name in names:
                nodes.append(node)
        elif constants and isinstance(node, (ast.Assign, ast.AnnAssign)):
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def load_shepherd(root, *, expected_revision=SHEPHERD_REVISION):
    """Return native get_messages, judge probability parser and score reducer."""
    root = Path(root)
    prefix = 'src/webprm/prompts/'
    constants = ['action.py', 'eval_type.py', 'input_information.py',
                 'judge_prompt.py', 'checklist_prompt.py']
    paths = [prefix + name for name in constants] + [
        prefix + 'construct_messages.py', 'src/webprm/models.py',
        'src/utils/eval_utils.py']
    manifest = source_manifest(root, paths, expected_revision)
    ns = {'ABC': ABC, 'abstractmethod': abstractmethod, 'math': math, 're': re}
    for name in constants:
        _extract(root / prefix / name, ns, names=set(), constants=True)
    _extract(root / prefix / 'construct_messages.py', ns)
    model_tree = ast.parse((root / 'src/webprm/models.py').read_text())
    cls = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == 'WebPRM')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and
               n.name in {'add_to_ori_logprob', 'get_judge_probs'}]
    pure_cls = ast.ClassDef(name='PureWebPRM', bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pure_cls], type_ignores=[])),
                 str(root / 'src/webprm/models.py'), 'exec'), ns)
    _extract(root / 'src/utils/eval_utils.py', ns, names={'get_score'})
    parser = ns['PureWebPRM']()
    return SimpleNamespace(manifest=manifest, get_messages=ns['get_messages'],
                           get_judge_probs=parser.get_judge_probs, get_score=ns['get_score'])


def load_arbiter(root, *, expected_revision=ARBITER_REVISION):
    """Load native verdict parser. Benchmark prompts are supplied by dataset.

    The released evaluator has no paper prompt builder; it consumes rendered
    JSONL prompt fields. The viewer's modified prompts are deliberately excluded.
    """
    root = Path(root)
    name = 'eval/WebPRMBench/script/evaluate_webprm_vllm.py'
    manifest = source_manifest(root, [name], expected_revision)
    ns = {'re': re}
    tree = ast.parse((root / name).read_text())
    nodes = [n for n in tree.body if
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ANSWER_RE' for t in n.targets)) or
             (isinstance(n, ast.FunctionDef) and n.name == 'parse_answer')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / name), 'exec'), ns)
    return SimpleNamespace(manifest=manifest, parse_answer=ns['parse_answer'])


def load_arbiter_benchmark_prompts(path):
    """Read exact paper benchmark strings and bind them to the dataset hash."""
    path = Path(path)
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    for record in records:
        if not isinstance(record.get('prompt'), str) or not record['prompt'].strip():
            raise ValueError('Benchmark record lacks a rendered prompt')
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'records': records}

ARBITER_BENCHMARK_SHA256 = '9cc5a462fcf9b6c5b45cbe76425111dd47105b5cabae240feceba9b6571e8cc3'
_ARBITER_FIELDS = ('intent', 'text_observation', 'trajectory', 'start_url',
                   'current_url', 'thought1', 'action1', 'thought2', 'action2')


def _arbiter_split_prompt(prompt):
    """Recover only variable spans; all surrounding upstream text is retained."""
    patterns = [
        r'(?<=#### Intent ####\n)(.*?)(?=\n\n#### AXTREE ####)',
        r'(?<=Always use bid to refer to elements in your actions.\n)(.*?)(?=\n\n#### Trajectory ####)',
        r"(?<=the concrete operation it performed \(`action`\).\n)(.*?)(?=\n\n#### start url ####)",
        r'(?<=#### start url ####\n)(.*?)(?=\n\n#### current url ####)',
        r'(?<=cart, checkout\).\n)(.*?)(?=\n\n#### Assistant Responses ####)',
        r'(?<=\[The Begin of Response 1\]\n\nTHOUGHT:\n)(.*?)(?=\n\nACTION:\n)',
        r'(?<=\n\nACTION:\n)(.*?)(?=\n\n\[The End of Response 1\])',
        r'(?<=\[The Begin of Response 2\]\n\nTHOUGHT:\n)(.*?)(?=\n\nACTION:\n)',
    ]
    spans = []
    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL)
        if match is None:
            raise ValueError('Unsupported official benchmark prompt layout')
        spans.append(match.span(1))
    tail_start = spans[-1][1]
    match = re.search(r'\n\nACTION:\n(.*?)\n\n\[The End of Response 2\]',
                      prompt[tail_start:], re.DOTALL)
    if match is None:
        raise ValueError('Missing second candidate action')
    spans.append((tail_start + match.start(1), tail_start + match.end(1)))
    parts, values, cursor = [], {}, 0
    for name, (start, end) in zip(_ARBITER_FIELDS, spans):
        if start < cursor:
            raise ValueError('Overlapping benchmark fields')
        parts.append(prompt[cursor:start])
        values[name] = prompt[start:end]
        cursor = end
    parts.append(prompt[cursor:])
    return parts, values


def load_arbiter_causal_renderer(path, *, expected_sha256=ARBITER_BENCHMARK_SHA256,
                                 template_source='webarena'):
    """Recover paper benchmark template from pinned rendered dataset.

    Only causal fields are accepted. Dataset preference labels, source metadata
    and candidate outcomes never enter rendered live prompts.
    """
    dataset = load_arbiter_benchmark_prompts(path)
    if expected_sha256 is not None and dataset['sha256'] != expected_sha256:
        raise ValueError('Unexpected Arbiter benchmark dataset hash')
    selected = next((record for record in dataset['records']
                     if record.get('source_name') == template_source), None)
    if selected is None:
        raise ValueError('Requested benchmark template source is absent')
    parts, _ = _arbiter_split_prompt(selected['prompt'])
    checked = 0
    variants = set()
    matched = 0
    for record in dataset['records']:
        other, values = _arbiter_split_prompt(record['prompt'])
        variants.add(hashlib.sha256(json.dumps(other, ensure_ascii=False).encode()).hexdigest())
        matched += int(other == parts)
        rebuilt = ''.join(part + values[name] for part, name in zip(other, _ARBITER_FIELDS)) + other[-1]
        if rebuilt != record['prompt']:
            raise ValueError('Benchmark template round-trip failed')
        checked += 1

    def render(**fields):
        if set(fields) != set(_ARBITER_FIELDS):
            raise ValueError(f'Expected only causal fields: {_ARBITER_FIELDS}')
        if any(not isinstance(value, str) for value in fields.values()):
            raise TypeError('All causal fields must be serialized strings')
        return ''.join(part + fields[name] for part, name in zip(parts, _ARBITER_FIELDS)) + parts[-1]

    return SimpleNamespace(render=render, manifest={
        'dataset_sha256': dataset['sha256'], 'records_validated': checked, 'records_matching_template': matched,
        'static_template_variants': len(variants), 'template_source': template_source,
        'template_sha256': hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest(),
        'prompt_origin': 'official_WEBPRMBENCH_rendered_dataset',
        'causal_fields': list(_ARBITER_FIELDS)})
