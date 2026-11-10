#!/usr/bin/env python3
"""Bounded CPU feasibility audit of five-alternative SFT, not a training builder."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import resource

from audit_arm_preference_pairs import canonical, parse_action, schemas
from report_arm_candidate_lengths import summarize

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction')
DATA = ROOT / 'joint-data-v2-20260910'
OUT = Path('openwebrl/docs/arm_results/serial_sft_feasibility.json')


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT/tokenizer.json')
    length = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    counts = defaultdict(Counter)
    lengths = defaultdict(lambda: defaultdict(list))
    failures = []
    hashes = {}
    hfpath = ROOT / 'joint-data-inputs-0d83b48/openwebrl_actor/candidates_merged.jsonl'
    with hfpath.open('rb') as hf:
        for split in ['train', 'validation']:
            path = DATA / f'joint_pairs.{split}.jsonl'
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    group = row['source'] + '/' + split
                    counts[group]['input_states'] += 1
                    try:
                        if row['source'] == 'C2':
                            raw = Path(row['source_path']).read_bytes()
                            if hashlib.sha256(raw).hexdigest() != row['source_sha256']:
                                raise ValueError('source_hash')
                            source = json.loads(raw)
                            texts = [x['response'] for x in source['raw_candidates']]
                            finishes = [x['finish_type'] for x in source['raw_candidates']]
                            if source['selected_index'] != row['chosen_index']:
                                raise ValueError('winner_join')
                        else:
                            hf.seek(row['candidate_file_offset']); raw = hf.readline()
                            if hashlib.sha256(raw).hexdigest() != row['candidate_line_sha256']:
                                raise ValueError('source_hash')
                            source = json.loads(raw)
                            if 'Piotr:' + source['state_id'] != row['draw_id']:
                                raise ValueError('draw_join')
                            texts = source['candidates']; finishes = [None] * len(texts)
                        if len(texts) != 5:
                            raise ValueError('not_five_candidates')
                        tools = schemas(row['prompt'])
                        blocks = []; calls_all = []
                        for i, (text, finish) in enumerate(zip(texts, finishes)):
                            calls, spans = parse_action(text, tools)
                            if row['source'] == 'C2' and finish != 'stop':
                                raise ValueError('candidate_not_stop')
                            if row['source'] == 'Piotr' and length(text) >= 1024:
                                raise ValueError('candidate_cap_suspected')
                            if text.count('</think>') != 1 or '</think>' not in text[:spans[0][0]]:
                                raise ValueError('candidate_reasoning_boundary')
                            thought = text.split('</think>')[0].removeprefix('<think>').strip()
                            if not thought:
                                raise ValueError('candidate_empty_reasoning')
                            if text[spans[-1][1]:].replace('<|im_end|>', '').strip():
                                raise ValueError('candidate_trailing_text')
                            calls_all.append(calls)
                            # Alternative actions are inert JSON, never tool-call blocks.
                            blocks.append(f'<alternative id="{i+1}">\nReasoning: {html.escape(thought, quote=False)}\nProposed action: {html.escape(canonical(calls), quote=False)}\n</alternative>')
                        chosen = row['chosen_index']
                        if calls_all[chosen] != row['chosen']['actions']:
                            raise ValueError('chosen_action_join')
                        distinct = len({canonical(calls) for calls in calls_all})
                        decision = f'Selected alternative: {chosen+1}\n</think>\n'
                        final = '\n'.join('<tool_call>' + canonical(c) + '</tool_call>' for c in calls_all[chosen]) + '<|im_end|>\n'
                        alternatives = '<think>\n' + '\n'.join(blocks) + '\n'
                        target = alternatives + decision + final
                        parsed, _ = parse_action(target, tools)
                        if parsed != row['chosen']['actions']:
                            raise ValueError('serialization_execution_mismatch')
                        nt = length(target)
                        counts[group]['all_five_structurally_usable'] += 1
                        counts[group][f'distinct_actions_{distinct}'] += 1
                        counts[group]['fits_32768_existing_prompt'] += row['prompt_tokens'] + nt <= 32768
                        counts[group]['fits_16384_existing_prompt'] += row['prompt_tokens'] + nt <= 16384
                        counts[group]['target_over_4096'] += nt > 4096
                        counts[group]['target_over_8192'] += nt > 8192
                        for key, value in dict(prompt=row['prompt_tokens'],target=nt,
                                total=row['prompt_tokens']+nt,alternatives=length(alternatives),
                                final_action=length(final),decision=length(decision),
                                previous_winner=row['chosen']['tokens']).items():
                            lengths[group][key].append(value)
                    except (ValueError, KeyError, OSError) as exc:
                        counts[group]['excluded/' + str(exc)] += 1
                        failures.append(dict(id=row['id'], reason=str(exc)))
    report = dict(time_utc=datetime.now(timezone.utc).isoformat(), data=str(DATA), input_hashes=hashes,
                  counts=dict(counts), lengths={g:{k:summarize(v) for k,v in fields.items()} for g,fields in lengths.items()},
                  exclusions=failures,
                  limitations=['Existing image-expanded prompt counts reused, not reprocessed.',
                    'Prompt instruction changes and compressed history not included in lengths.',
                    'Image bytes and complete evaluation decontamination not reaudited.',
                    'Full candidate structural audit; no semantic validation of reasoning.',
                    'Piotr completion cap is conservative heuristic; finish metadata absent.',
                    'One deterministic serialization; training permutations may change token counts slightly.',
                    'No GPU memory, throughput, or learning feasibility measured.'])
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k!='exclusions'}, indent=2))


if __name__ == '__main__':
    main()
