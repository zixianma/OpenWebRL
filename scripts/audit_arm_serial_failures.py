#!/usr/bin/env python3
"""Bounded CPU audit of saved serial responses; never executes or repairs actions."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import resource

from audit_arm_preference_pairs import parse_action, schemas
from openwebrl.serial_actions import canonical, instructions, parse_serial


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def diagnose(text, variant, tools):
    result = dict(protocol_valid=False, alternative_openings=text.count('<alternative'),
                  alternative_closings=text.count('</alternative>'),
                  think_closings=text.count('</think>'),
                  selection_fields=re.findall(r'Selected alternative: (\d+)', text),
                  final_schema_valid=False, final_matches_selected=None)
    try:
        parse_serial(text, variant)
        result['protocol_valid'] = True
    except (ValueError, TypeError, KeyError) as exc:
        result['protocol_error'] = str(exc)
    # Independent diagnostics only. This does NOT change the evaluation parser.
    tail = text.rsplit('</think>', 1)[-1]
    try:
        calls, _ = parse_action(tail, tools)
        result.update(final_schema_valid=True, final_calls=calls)
    except (ValueError, TypeError, KeyError) as exc:
        result['final_schema_error'] = str(exc)
    candidates = {}
    pattern = r'<alternative id="(\d+)">\s*Reasoning: (.*?)\nProposed action: (.*?)\s*</alternative>\s*'
    for m in re.finditer(pattern, text, re.S):
        try: candidates[m[1]] = json.loads(html.unescape(m[3]))
        except ValueError: pass
    result['complete_candidate_ids'] = list(candidates)
    if len(result['selection_fields']) == 1:
        selected = result['selection_fields'][0]
        if selected in candidates:
            try:
                raw_calls = [json.loads(m[1]) for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tail, re.S)]
                result['final_matches_selected'] = bool(raw_calls) and canonical(raw_calls) == canonical(candidates[selected])
            except ValueError: pass
    result['serial_suffix_valid_for_diagnosis_only'] = False
    if '<alternative' in text:
        try:
            parse_serial(text[text.index('<alternative'):], variant)
            result['serial_suffix_valid_for_diagnosis_only'] = True
        except (ValueError, TypeError, KeyError): pass
    return result


def audit_training(data):
    result = {}
    for variant in ('all5', 'diverse3'):
        for split in ('train', 'validation'):
            path = data / f'{variant}.{split}.jsonl'
            counts = Counter(); failures = []; first_weights = []
            for line in path.open():
                row = json.loads(line); counts['rows'] += 1
                try:
                    target = row['chosen']; final, _, _ = parse_serial(target['text'], variant)
                    parse_action(final, schemas(row['prompt']))
                    assert row['prompt'].endswith('<think>\n')
                    assert row['prompt'].split('<|im_end|>', 1)[0].count(instructions(variant)) == 1
                    assert target['text'].startswith('<alternative id="1">')
                    assert target['token_ids'][0] == 27  # '<' in the frozen actor tokenizer.
                    assert len(target['token_ids']) == len(target['segments']) == target['tokens']
                    assert target['segments'][0] == 0 and set(target['segments']) == {0, 1, 2}
                    first_weights.append(.5 / target['segments'].count(0))
                    counts['valid_targets_and_opening_alignment'] += 1
                except (ValueError, TypeError, KeyError, AssertionError) as exc:
                    failures.append(dict(id=row['id'], error=repr(exc)))
            result[f'{variant}/{split}'] = dict(counts=counts, failures=failures,
                first_token_loss_weight=dict(min=min(first_weights), max=max(first_weights),
                    mean=sum(first_weights)/len(first_weights)), source_sha256=sha(path))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path)
    args = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (90, 95))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    panel = {r['id']: r for r in map(json.loads, args.panel.open())}
    saved = json.loads(args.generation.read_text())['examples']; examples = []
    for row in saved:
        source = panel[row['id']]
        record = diagnose(row['text'], source['variant'], schemas(source['prompt']))
        record.update(id=row['id'], temperature=row['temperature'], raw_text=row['text'])
        examples.append(record)
    summary = dict(responses=len(examples),
        protocol_valid=sum(r['protocol_valid'] for r in examples),
        responses_with_alternatives=sum(r['alternative_openings'] > 0 for r in examples),
        responses_without_alternatives=sum(r['alternative_openings'] == 0 for r in examples),
        protocol_errors=Counter(r.get('protocol_error') for r in examples),
        final_schema_valid=sum(r['final_schema_valid'] for r in examples),
        final_schema_invalid=sum(not r['final_schema_valid'] for r in examples),
        selected_final_match=sum(r['final_matches_selected'] is True for r in examples),
        selected_final_mismatch=sum(r['final_matches_selected'] is False for r in examples),
        selected_final_not_comparable=sum(r['final_matches_selected'] is None for r in examples))
    report = dict(summary=summary, examples=examples,
        note='Diagnostic only: no output repairs, execution, or gate promotion. Structural validity does not establish action quality.',
        sources_sha256={str(x): sha(x) for x in (args.generation, args.panel, Path(__file__))})
    if args.data: report['training_data'] = audit_training(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(summary=summary, training_data=report.get('training_data')), indent=2))


if __name__ == '__main__': main()
