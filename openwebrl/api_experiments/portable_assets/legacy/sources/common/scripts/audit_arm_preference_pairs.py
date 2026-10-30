#!/usr/bin/env python3
"""Bounded, stdlib-only structural audit of C2 candidates; no model/image loads."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import time
from urllib.parse import urlsplit


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def validate(value, schema):
    kind = schema.get('type')
    types = {'object': dict, 'array': list, 'string': str, 'boolean': bool}
    if kind in types and not isinstance(value, types[kind]):
        raise ValueError('wrong_type_' + kind)
    if kind in ('number', 'integer'):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('invalid_number')
        if kind == 'integer' and int(value) != value:
            raise ValueError('not_integer')
        for bound, comparator in [('minimum', lambda a, b: a < b), ('maximum', lambda a, b: a > b)]:
            if bound in schema and comparator(value, schema[bound]):
                raise ValueError(bound)
        value = int(value) if int(value) == value else value
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('enum')
    if kind == 'object':
        props = schema.get('properties', {})
        if set(schema.get('required', [])) - value.keys():
            raise ValueError('missing_required')
        if value.keys() - props.keys():
            raise ValueError('unknown_argument')
        value = {k: validate(v, props[k]) for k, v in value.items()}
        for k, prop in props.items():
            if k not in value and 'default' in prop:
                value[k] = validate(prop['default'], prop)
    if kind == 'array':
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', math.inf):
            raise ValueError('array_length')
        value = [validate(v, schema.get('items', {})) for v in value]
    return value


def schemas(prompt):
    blocks = [m for m in re.finditer(r'<tools>\s*(.*?)\s*</tools>', prompt, re.S) if m[1].strip()]
    if not blocks:
        raise ValueError('missing_tool_schema')
    block = blocks[0]
    result = {}
    for line in block[1].splitlines():
        if line.strip():
            f = json.loads(line)['function']
            result[f['name']] = f['parameters']
    return result


def parse_action(response, tools):
    # Mirrors ToolParser._parse_regex's XML/JSON format, with stricter checks
    # for malformed extra blocks and schemas. Does not import transformers.
    spans = list(re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.S))
    if not spans or len(spans) != response.count('<tool_call>') or len(spans) != response.count('</tool_call>'):
        raise ValueError('malformed_or_missing_tool_call')
    calls = []
    for match in spans:
        call = json.loads(match[1])
        if not isinstance(call, dict) or call.get('name') not in tools:
            raise ValueError('unknown_tool')
        args = call.get('arguments', call.get('parameters', {}))
        calls.append({'name': call['name'], 'arguments': validate(args, tools[call['name']])})
    return calls, [[s.start(), s.end()] for s in spans]


def coordinate_distance(left, right):
    """Max Euclidean coordinate distance if ALL other call semantics match."""
    if len(left) != len(right):
        return None
    distances = []
    for a, b in zip(left, right):
        if a['name'] != b['name'] or a['arguments'].keys() != b['arguments'].keys():
            return None
        for key, value in a['arguments'].items():
            other = b['arguments'][key]
            if key in ('point_2d', 'start_point_2d', 'end_point_2d'):
                distances.append(math.dist(value, other))
            elif value != other:
                return None
    return max(distances, default=0.)


def key(value):
    return digest(('arm-pref-v2-audit:42:' + value).encode())


def stratified(rows, count):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['host'], r['winner_type'])].append(r)
    for group in groups.values():
        group.sort(key=lambda r: key(r['id']))
    names = sorted(groups, key=lambda x: key(str(x)))
    selected = []
    for offset in range(max(map(len, groups.values()), default=0)):
        for name in names:
            if offset < len(groups[name]):
                selected.append(groups[name][offset])
                if len(selected) == count:
                    return selected
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sleep-ms', type=float, default=10.)
    args = p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 185))
    begin = time.monotonic()
    audit = json.loads((args.source_root / 'dataset-audit.json').read_text())
    dataset = Path(audit['dataset'])
    args.output.mkdir(parents=True, exist_ok=False)
    counts, reasons, thresholds = Counter(), Counter(), Counter()
    hosts, action_types, schema_hashes = Counter(), Counter(), set()
    rows = []
    failures = []
    seen, image_refs = set(), {}
    execution_path, execution, outcome = None, None, None
    stream_hash = hashlib.sha256()
    source_bytes = execution_bytes = 0
    manifest = (args.output / 'eligible-states.jsonl').open('w')
    try:
        with dataset.open('rb') as stream:
            for lineno, line in enumerate(stream, 1):
                stream_hash.update(line)
                r = json.loads(line)
                counts['source_turns'] += 1
                try:
                    ident = (r['task_id'], r['turn'])
                    if ident in seen:
                        raise ValueError('duplicate_state')
                    seen.add(ident)
                    raw = Path(r['source']).read_bytes(); source_bytes += len(raw)
                    if digest(raw) != r['source_sha256']:
                        raise ValueError('source_hash')
                    c = json.loads(raw)
                    if c['task_id'] != r['task_id'] or c['turn'] != r['turn'] or c['selected_index'] != r['selected_index']:
                        raise ValueError('source_join')
                    if digest(c['prompt'].encode()) != c['prompt_sha256']:
                        raise ValueError('prompt_hash')
                    ep = Path(c['attempt']) / 'execution.json'
                    if execution_path != ep:
                        eraw = ep.read_bytes(); execution_bytes += len(eraw)
                        e = json.loads(eraw)
                        if e['task_id'] != r['task_id']:
                            raise ValueError('execution_task_join')
                        execution = {t['turn']: t for t in e['turns']}
                        outcome = json.loads((ep.parent / 'outcome.json').read_text())
                        execution_path = ep
                    t = execution[r['turn']]
                    if not outcome.get('valid') or outcome.get('reward') != 1 or outcome['task_id'] != r['task_id']:
                        raise ValueError('terminal_outcome')
                    if not t['executed'] or t['source_sha256'] != r['source_sha256'] or t['source'] != r['source']:
                        raise ValueError('execution_source_join')
                    if t['prompt_token_ids'] != r['prompt_token_ids'] or t['image_grid_thw'] != r['image_grid_thw']:
                        raise ValueError('execution_tokens_or_grid')
                    if c.get('fallback') or c['mode'] != 'selection':
                        raise ValueError('selector_fallback_or_mode')
                    for img in c['images']:
                        if img['path'] in image_refs and image_refs[img['path']] != img['sha256']:
                            raise ValueError('image_reference_conflict')
                        image_refs[img['path']] = img['sha256']
                    tool_schemas = schemas(c['prompt'])
                    schema_hashes.add(digest(canonical(tool_schemas).encode()))
                    wi = r['selected_index']; w = c['raw_candidates'][wi]
                    if w['finish_type'] != 'stop' or not w['response_token_ids']:
                        raise ValueError('winner_unusable')
                    wa, ws = parse_action(w['response'], tool_schemas)
                    groups = {}
                    for i, x in enumerate(c['raw_candidates']):
                        if i == wi:
                            continue
                        counts['candidate_losers'] += 1
                        if x.get('finish_type') != 'stop' or not x.get('response_token_ids'):
                            reasons['truncated_or_empty_loser'] += 1; continue
                        if len(r['prompt_token_ids']) + len(x['response_token_ids']) > 32768:
                            reasons['oversize_loser'] += 1; continue
                        try:
                            la, ls = parse_action(x['response'], tool_schemas)
                        except (ValueError, TypeError, KeyError) as err:
                            reasons['loser_parse_schema_' + str(err)[:60]] += 1; continue
                        distance = coordinate_distance(wa, la)
                        if distance == 0:
                            reasons['exact_action_equivalent_loser'] += 1; continue
                        lk = canonical(la)
                        if lk in groups:
                            reasons['duplicate_distinct_loser_action'] += 1; continue
                        groups[lk] = {'index': i, 'action': la, 'action_char_spans': ls,
                                      'response_tokens': len(x['response_token_ids']), 'coordinate_distance': distance}
                    losers = list(groups.values())
                    for threshold in (0, 2, 5, 10):
                        if any(x['coordinate_distance'] is None or x['coordinate_distance'] > threshold for x in losers):
                            thresholds[str(threshold)] += 1
                    if not losers:
                        reasons['state_without_distinct_schema_valid_loser'] += 1; continue
                    host = (urlsplit(c['url']).hostname or '').lower()
                    row = {'id': f'{r["task_id"]}:{r["turn"]}', 'task_id': r['task_id'], 'turn': r['turn'],
                           'source': r['source'], 'source_sha256': r['source_sha256'], 'source_line': lineno,
                           'host': host, 'winner_index': wi, 'winner_type': '+'.join(a['name'] for a in wa),
                           'winner_action': wa, 'winner_action_char_spans': ws, 'winner_response_tokens': len(w['response_token_ids']),
                           'prompt_tokens': len(r['prompt_token_ids']), 'images': c['images'], 'losers': losers}
                    manifest.write(json.dumps(row, ensure_ascii=False) + '\n')
                    rows.append(row); hosts[host] += 1; action_types[row['winner_type']] += 1
                except (ValueError, KeyError, OSError, TypeError) as err:
                    failures.append({'line': lineno, 'task_id': r['task_id'], 'error': str(err)[:160]})
                if lineno % 1000 == 0:
                    print(f'audited {lineno} turns, {len(rows)} structurally eligible, {len(failures)} source/winner failures', flush=True)
                time.sleep(args.sleep_ms / 1000)
    finally:
        manifest.close()
    if stream_hash.hexdigest() != audit['dataset_sha256']:
        failures.append({'error': 'dataset_hash'})
    missing = []
    for i, path in enumerate(image_refs):
        if not Path(path).is_file(): missing.append(path)
        if i % 64 == 0: time.sleep(.01)

    # Provisional split only: drop near-coordinate alternatives <=5 units pending
    # screenshot review; no GPU-ranked loser selection is performed here.
    conservative = []
    for r in rows:
        ls = [x for x in r['losers'] if x['coordinate_distance'] is None or x['coordinate_distance'] > 5]
        if ls: conservative.append(dict(r, losers=ls))
    tasks = sorted({r['task_id'] for r in conservative}, key=key)
    validation_tasks = set(tasks[:math.ceil(.2 * len(tasks))])
    pool = [r for r in conservative if r['task_id'] not in validation_tasks]
    validation_pool = [r for r in conservative if r['task_id'] in validation_tasks]
    train = stratified(pool, 2048); validation = stratified(validation_pool, 256)
    calibration = stratified(train, 128)
    review = stratified(conservative, 64)
    for label, selected in [('provisional-train', train), ('provisional-validation', validation), ('provisional-calibration', calibration), ('review-64', review)]:
        with (args.output / (label + '.jsonl')).open('w') as f:
            for r in selected: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    assert not {r['task_id'] for r in train} & {r['task_id'] for r in validation}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {'source_dataset': str(dataset), 'source_dataset_sha256': stream_hash.hexdigest(),
              'counts': dict(counts), 'structurally_eligible_states': len(rows), 'exclusions': dict(reasons),
              'eligible_states_by_coordinate_exclusion_radius': dict(thresholds), 'source_or_winner_failures': failures,
              'unique_image_references': len(image_refs), 'missing_images': missing, 'unique_tool_schema_versions': len(schema_hashes),
              'hosts': dict(hosts.most_common()), 'winner_types': dict(action_types.most_common()),
              'provisional_split': {'radius': 5, 'eligible': len(conservative), 'eligible_tasks': len(tasks),
                                    'train_pool': len(pool), 'validation_pool': len(validation_pool),
                                    'train': len(train), 'validation': len(validation), 'calibration': len(calibration),
                                    'train_tasks': len({r['task_id'] for r in train}), 'validation_tasks': len({r['task_id'] for r in validation})},
              'deferred': ['Image byte hashing and screenshot semantic review', 'Native SGLang parser parity (stdlib mirror used)',
                           'Tokenizer token-span verification and processor-expanded prefix check',
                           'Frozen-reference scoring and final hard-negative choice', 'Calibration and training'],
              'resources': {'wall_seconds': time.monotonic() - begin, 'cpu_seconds': usage.ru_utime + usage.ru_stime,
                            'peak_rss_mib': usage.ru_maxrss / 1024, 'source_json_bytes_read': source_bytes,
                            'execution_json_bytes_read': execution_bytes, 'dataset_bytes': dataset.stat().st_size,
                            'throttle_sleep_ms_per_state': args.sleep_ms},
              'artifact_sha256': {p.name: digest(p.read_bytes()) for p in args.output.glob('*.jsonl')}}
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    compact = {k: report[k] for k in ('counts', 'structurally_eligible_states', 'exclusions',
          'eligible_states_by_coordinate_exclusion_radius', 'provisional_split', 'resources')}
    compact['source_or_winner_failure_count'] = len(failures)
    compact['failure_examples'] = failures[:3]
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == '__main__':
    main()
