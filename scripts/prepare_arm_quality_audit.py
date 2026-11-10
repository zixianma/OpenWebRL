#!/usr/bin/env python3
"""Bounded offline ARM audit: read archive metadata, never tensor storage/models.

Produces a late-actor candidate audit and an outcome-blind fixed-state panel for
comparing SFT / baseline-20 / baseline-90. No GPU or API calls are made.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

from audit_arm_preference_pairs import canonical, parse_action, schemas
from replay_arm_confidence import archive_metadata, current_url, image_bytes, sha
from openwebrl.arm_inference import split_response


def stable_key(*parts):
    return sha(json.dumps(['arm-quality-20260921', *parts]).encode())


def choose_attempt(records):
    """Choose before reading outcome/validity; never replace an invalid draw."""
    return min(records, key=lambda r: stable_key(r['task_id'], r['attempt']))


def inspect_candidates(candidates, prompt, selected_index, finish_types):
    tools = schemas(prompt)
    classes, errors, sequences = [], [], []
    for candidate in candidates:
        try:
            action, _ = parse_action(candidate['action'], tools)
            classes.append(canonical(action))
            errors.append(None)
            sequences.append(tuple(call['name'] for call in action))
        except (ValueError, TypeError, KeyError) as exc:
            classes.append(None)
            errors.append(str(exc))
            sequences.append(None)
    valid = [c for c in classes if c is not None]
    selected = classes[selected_index]
    return dict(valid_candidates=len(valid), unique_valid_actions=len(set(valid)),
                action_classes=classes, parse_errors=errors,
                unique_valid_tool_sequences=len({s for s in sequences if s is not None}),
                all_five_are_done=all(s == ('done',) for s in sequences),
                selected_action_valid=selected is not None,
                selected_action_multiplicity=classes.count(selected) if selected is not None else None,
                stopped_candidates=sum(f == 'stop' for f in finish_types),
                finish_types=finish_types)


def load_turns(record):
    root = Path(record['artifact'])
    archives = list(root.glob('*.pt'))
    if len(archives) != 1:
        raise ValueError('Expected exactly one trajectory archive')
    archive = archive_metadata(archives[0])
    if archive['task_id'] != record['task_id']:
        raise ValueError('Task identity mismatch')
    turns = archive['turns']
    indices = [s['metadata']['turn_index'] for s in turns]
    if indices != list(range(len(turns))):
        raise ValueError('Non-contiguous archived history')
    return archives[0], turns


def context(record, archive, turns, turn, image_dir, screenshot_hash=None):
    sample = turns[turn]
    if screenshot_hash is None:
        # The current screenshot is the last image in the actor input. Prior
        # observations remain in the exact actor prompt/input archive.
        import base64
        images = (sample.get('multimodal_inputs') or {}).get('images', [])
        if not images or not isinstance(images[-1], str) or not images[-1].startswith('data:image/'):
            raise ValueError('Missing current image')
        raw = base64.b64decode(images[-1].split(',', 1)[1], validate=True)
        screenshot_hash = sha(raw)
    else:
        raw = image_bytes(sample, screenshot_hash)
    path = image_dir / (screenshot_hash + '.png')
    path.write_bytes(raw)
    # Retain ALL input images, not only the image shown to the selector.
    actor_images = []
    import base64
    for encoded in (sample.get('multimodal_inputs') or {}).get('images', []):
        if not isinstance(encoded, str) or not encoded.startswith('data:image/'):
            raise ValueError('Unrecognized archived actor image')
        body = base64.b64decode(encoded.split(',', 1)[1], validate=True)
        digest = sha(body)
        target = image_dir / (digest + '.png')
        target.write_bytes(body)
        actor_images.append(dict(path=str(target), sha256=digest))
    return dict(task_id=record['task_id'], task=sample['metadata']['intent'],
                archive=str(archive), attempt=record['attempt'], turn=turn,
                actor_prompt=sample['prompt'], prompt_sha256=sha(sample['prompt'].encode()),
                actor_images=actor_images, url=current_url(sample),
                history=[split_response(s['response']) for s in turns[:turn]],
                screenshot=str(path), screenshot_sha256=screenshot_hash,
                state_source_policy=record['policy_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 95))
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / 'images'
    images.mkdir()
    retries = [json.loads(p.read_text()) for p in sorted((args.pilot / 'retry_records').glob('*.json'))]
    audited, unavailable, counts = [], [], Counter()
    for record in retries:
        if record['mode'] != 'arm':
            continue
        counts['selection_mode_attempts'] += record['selector_calls']
        archive, turns = load_turns(record)
        selections = list((Path(record['artifact']) / 'selections').glob('*.jsonl'))
        if len(selections) != 1:
            raise ValueError('Ambiguous selection log')
        trace_raw = selections[0].read_bytes()
        for line, raw in enumerate(trace_raw.splitlines(), 1):
            r = json.loads(raw)
            counts['logged_selections'] += 1
            if r['task_id'] != record['task_id'] or len(r['candidates']) != 5:
                raise ValueError('Candidate identity/count mismatch')
            index, turn = r['selected_index'], r['turn']
            if turn >= len(turns):
                unavailable.append(dict(task_id=record['task_id'], turn=turn,
                                        reason='selection_without_saved_executed_turn'))
                continue
            s = turns[turn]
            if sha(s['prompt'].encode()) != r['prompt_sha256']:
                raise ValueError('Actor prompt hash mismatch')
            if split_response(s['response']) != r['candidates'][index]:
                raise ValueError('Executed response differs from selected candidate')
            c = context(record, archive, turns, turn, images, r['screenshot_sha256'])
            if c['url'] != r['url']:
                raise ValueError('Current URL mismatch')
            metrics = inspect_candidates(r['candidates'], s['prompt'], index, r['finish_types'])
            c.update(id=stable_key(record['task_id'], 'rescue', turn),
                     candidates=r['candidates'], selected_index=index,
                     selector_fallback=r['fallback'], trace=str(selections[0]),
                     trace_sha256=sha(trace_raw), trace_line=line, metrics=metrics,
                     trajectory_valid=record['valid'], trajectory_reward=record['reward'])
            audited.append(c)
            counts['joined_states'] += 1
            counts['candidates'] += len(r['candidates'])
            counts['valid_candidates'] += metrics['valid_candidates']
            counts['all_candidates_valid_states'] += metrics['valid_candidates'] == 5
            counts['all_candidates_stopped_states'] += metrics['stopped_candidates'] == 5
            counts['selected_action_valid_states'] += metrics['selected_action_valid']
            counts['fallback_states'] += bool(r['fallback'])
            counts['all_five_done_states'] += metrics['all_five_are_done']
            counts['all_valid_at_least_two_distinct_states'] += metrics['valid_candidates'] == 5 and metrics['unique_valid_actions'] >= 2
        time.sleep(.01)
    by_task = {}
    for p in sorted((args.pilot / 'screen_records').glob('*.json')):
        record = json.loads(p.read_text())
        by_task.setdefault(record['task_id'], []).append(record)
    panel, missing, assignments = [], [], []
    for task, records in sorted(by_task.items()):
        record = choose_attempt(records)
        archive, turns = load_turns(record)
        # Hash chooses one of first/middle/last thirds, independently of outcomes.
        fraction = (int(stable_key(task, 'turn'), 16) % 3 + .5) / 3
        assignments.append(dict(task_id=task, attempt=record['attempt'], quantile=fraction,
                                archive=str(archive)))
        if not turns:
            missing.append(dict(task_id=task, attempt=record['attempt'], reason='no_saved_turns'))
            continue
        turn = min(int(len(turns) * fraction), len(turns) - 1)
        row = context(record, archive, turns, turn, images)
        row.update(id=stable_key(task, 'shared-state', record['attempt'], turn),
                   turn_quantile=fraction, archive_turns=len(turns))
        panel.append(row)
        time.sleep(.01)
    for name, rows in [('late-rescue-candidates.jsonl', audited), ('shared-states.jsonl', panel)]:
        path = args.output / name
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), pilot=str(args.pilot),
                  counts=dict(counts), unique_valid_action_histogram=dict(Counter(r['metrics']['unique_valid_actions'] for r in audited)),
                  unique_valid_tool_sequence_histogram=dict(Counter(r['metrics']['unique_valid_tool_sequences'] for r in audited)),
                  call_accounting_note='MeteredSelector increments before candidate generation/selector completion: attempts are not confirmed successful ARM RPCs.',
                  parse_errors=dict(Counter(e for r in audited for e in r['metrics']['parse_errors'] if e)),
                  selection_join_exclusions=unavailable, panel_tasks_total=len(by_task),
                  panel_states=len(panel), panel_missing=missing, panel_assignments=assignments,
                  output=str(args.output), inputs_sha256={p.name: sha(p.read_bytes()) for p in args.output.glob('*.jsonl')},
                  limitation='Structural validity is not action correctness. Shared states come from baseline-90; no early-state visitation or causal drift claim.',
                  elapsed_seconds=time.monotonic()-started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'panel_assignments'}, indent=2))


if __name__ == '__main__':
    main()
