#!/usr/bin/env python3
"""Recompute difficulty-stratified rates from the nine saved stealth90 cohorts."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
RESULTS = REPO / 'openwebrl/docs/arm_results/rl_integration'
METHODS = ['baseline', 'additive', 'gate-b']
BANDS = ['easy', 'medium', 'hard']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(root):
    records = {}
    for path in (root / 'rollouts').glob('*.json'):
        r = read(path)
        task_id = str(r['task_id'])
        assert task_id not in records, (root, task_id)
        records[task_id] = (r, path)
    return records


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    source = RUNTIME / 'reference-arm-stealth90-o4-t06-20260929/online_mind2web_monitor.jsonl'
    tasks = [json.loads(line)['metadata'] for line in source.open()]
    metadata = {str(t['task_id']): t for t in tasks}
    assert len(metadata) == len(tasks) == 300
    length_mismatches = []
    for t in tasks:
        length = int(t['reference_length'])
        assert length >= 1
        assert t['difficulty'] in BANDS
        derived = 'easy' if length <= 5 else 'medium' if length <= 10 else 'hard'
        if t['difficulty'] != derived:
            length_mismatches.append(dict(task_id=t['task_id'], saved_difficulty=t['difficulty'],
                                          reference_length=length, length_derived_difficulty=derived))
        t['saved_difficulty'] = t['difficulty']
        t['difficulty'] = derived
    official = read(RESULTS / 'stealth-o4-t06-iteration90-three-repeat-summary.json')
    rows = []
    per_repeat = []
    sources = []
    for method in METHODS:
        for cohort in official['methods'][method]['per_repeat']:
            rep = cohort['repeat']
            audit_path = RESULTS / cohort['source']
            assert sha(audit_path) == cohort['sha256']
            audit = read(audit_path)
            if rep == 1:
                original = load_records(Path(audit['original_root']))
                root = Path(audit['recovery_root'])
                recovery = load_records(root)
                manifest = read(root / 'evaluation_manifest.json')
                expected_retry = set(manifest['provider_recovery']['expected_task_ids'])
                assert set(original) == set(metadata)
                assert set(recovery) == expected_retry
                assert all(original[k][0]['metrics']['valid_trajectories'] == 0 for k in expected_retry)
                merged = dict(original, **recovery)
            else:
                root = RUNTIME / audit['root']
                merged = load_records(root)
            assert set(merged) == set(metadata)
            manifest = read(root / 'evaluation_manifest.json')
            assert manifest['completed_iteration'] == 90
            assert manifest['actor_temperature'] == .6
            for ident, (r, path) in sorted(merged.items()):
                m = r['metrics']
                assert r['judge_model'] == 'o4-mini' and r['judge_prompt_variant'] == 'agenttrek'
                n, v, s = m['trajectories'], m['valid_trajectories'], m['successes']
                assert n == 1 and v in (0, 1) and s in (0, 1) and s <= v
                assert m['invalid_trajectories'] == 1 - v
                if v:
                    assert r['reward_metadata']['judge_text'].strip()
                archive = path.parent / r['rollout_file']
                assert archive.is_file() and archive.stat().st_size > 0
                rows.append(dict(method=method, repeat=rep, task_id=ident,
                                 difficulty=metadata[ident]['difficulty'],
                                 saved_difficulty=metadata[ident]['saved_difficulty'],
                                 reference_length=metadata[ident]['reference_length'],
                                 success=int(s), valid=int(v), invalid=1-int(v),
                                 record_sha256=sha(path)))
            batch = [r for r in rows if r['method'] == method and r['repeat'] == rep]
            assert sum(r['success'] for r in batch) == cohort['successes']
            assert sum(r['valid'] for r in batch) == cohort['valid']
            for band in BANDS:
                group = [r for r in batch if r['difficulty'] == band]
                n = len(group); s = sum(r['success'] for r in group); v = sum(r['valid'] for r in group)
                per_repeat.append(dict(method=method, repeat=rep, difficulty=band,
                                       tasks=n, successes=s, valid=v, invalid=n-v,
                                       overall=s/n, valid_only=s/v if v else None))
            sources.append(dict(method=method, repeat=rep, audit_file=audit_path.name,
                                audit_sha256=sha(audit_path), root=str(root)))
    assert len(rows) == 2700
    summary = []
    for method in METHODS:
        for band in BANDS:
            groups = [r for r in per_repeat if r['method'] == method and r['difficulty'] == band]
            assert len(groups) == 3 and all(r['valid'] > 0 for r in groups)
            summary.append(dict(method=method, difficulty=band, tasks_per_repeat=groups[0]['tasks'],
                                overall_mean=statistics.mean(r['overall'] for r in groups),
                                overall_sample_sd=statistics.stdev(r['overall'] for r in groups),
                                valid_only_mean=statistics.mean(r['valid_only'] for r in groups),
                                valid_only_sample_sd=statistics.stdev(r['valid_only'] for r in groups),
                                successes_by_repeat=[r['successes'] for r in groups],
                                valid_by_repeat=[r['valid'] for r in groups],
                                invalid_by_repeat=[r['invalid'] for r in groups]))
    for entry in summary:
        baseline = next(s for s in summary if s['method'] == 'baseline' and s['difficulty'] == entry['difficulty'])
        entry['overall_delta_baseline_pp'] = 100*(entry['overall_mean']-baseline['overall_mean'])
    for method in METHODS:
        flat = sum(s['overall_mean']*s['tasks_per_repeat'] for s in summary if s['method'] == method)/300
        assert abs(flat-official['methods'][method]['overall_mean']) < 1e-12
    table = output / 'per-task-verdicts.csv'
    with table.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result = dict(iteration=90, repeats=3, tasks=300, protocol=official['protocol'],
                  difficulty_definition={'easy': 'human reference 1–5 steps', 'medium': '6–10 steps', 'hard': '11+ steps'},
                  difficulty_source=str(source), difficulty_source_sha256=sha(source),
                  difficulty_counts=dict(Counter(t['difficulty'] for t in tasks)),
                  label_policy='Apply published human-reference-length bins to the frozen task metadata; retain original labels separately; no evaluation artifacts modified',
                  reference_length_label_mismatches=length_mismatches,
                  saved_difficulty_counts=dict(Counter(t['saved_difficulty'] for t in tasks)),
                  difficulty_definition_url='https://huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard/blob/2dba94f7112bf2e47c701a49c5af971fbf0723d4/content.py',
                  summary=summary, per_repeat=per_repeat, sources=sources,
                  per_task_verdicts_sha256=sha(table), implementation_sha256=sha(Path(__file__)),
                  verified_complete=True, new_model_calls=0, new_browser_calls=0,
                  checks=['Exact same 300 IDs in each cohort; all difficulty labels present, with reference-length mismatches recorded.',
                          'Only declared credit-blocked records replaced in repeat1; no valid records replaced.',
                          'All 2700 retained records have the expected judge, binary metrics and existing rollout archives.',
                          'All nine flat success/valid totals reproduce the previously verified audits.',
                          'Difficulty-weighted overall means reproduce the published full300 aggregate.'],
                  caveats=['Three evaluation repeats of one checkpoint per method, not training-seed replication.',
                           'Valid-only means average per-repeat ratios, not pooled counts.',
                           'Repeat1 contains previously audited credit-outage recovery.',
                           'This is descriptive subgroup analysis; no significance claim.',
                           'Two stale medium labels have reference lengths 11/12; primary tables classify them as hard under the published rule.',
                           'OM2W human-step difficulty differs from WebGym rubric-fact counts.'])
    (output / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for s in summary:
        print(s['method'], s['difficulty'], 'overall', round(100*s['overall_mean'], 2),
              'valid-only', round(100*s['valid_only_mean'], 2), 'valid', s['valid_by_repeat'],
              'delta', round(s['overall_delta_baseline_pp'], 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
