#!/usr/bin/env python3
"""Render all approved weighted tasks for review without browsing or model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band(d):
    return 'hard' if d >= 7 else 'medium' if d >= 4 else 'easy'


def main(pool, cohort, output):
    cohort_summary = json.loads((cohort/'summary.json').read_text())
    difficulty_first = cohort_summary.get('selection_mode') == 'difficulty_first'
    minimum_difficulty = cohort_summary.get('minimum_rubric_difficulty')
    manifest = {(r['source'], str(r['task_id'])): r for r in rows(cohort/'selection-manifest.jsonl')}
    tasks = []
    for row in rows(cohort/'weighted-tasks.jsonl'):
        key = row['benchmark_name'], str(row['task_id'])
        m = manifest[key]
        facts = sum(len(r.get('facts', [])) for r in row['evaluator_reference'])
        assert facts == row['difficulty']
        assert row['task_name'] == m['instruction'] and row['website'] == m['start_url']
        tasks.append(dict(key=':'.join(key), task_id=key[1], source=key[0], site=m['site'],
                          url=row['website'], instruction=row['task_name'],
                          difficulty=row['difficulty'], band=band(row['difficulty']),
                          rubric=row['evaluator_reference'], cluster=m['cluster_id'],
                          workflow=m['primary_workflow'], selection_rank=m['within_site_rank'],
                          actor_difficulty='Not measured', domain=row.get('domain'),
                          subdomain=row.get('subdomain')))
    assert len(tasks) == len(manifest) == 2000
    eligible = {(r['source'], str(r['task_id'])) for r in rows(pool/'benchmark-sites-20260930/verified/available-benchmark-tasks.jsonl')}
    pool_bands = Counter()
    for r in rows(pool/'candidates.jsonl'):
        if (r['benchmark_name'], str(r['task_id'])) in eligible:
            pool_bands[band(r['difficulty'])] += 1
    assert sum(pool_bands.values()) == 59115
    digest = sha(cohort/'weighted-tasks.jsonl')
    selected_counts = Counter(t['band'] for t in tasks)
    payload = dict(tasks=tasks, cohort_sha256=digest,
                   selected_bands={b:selected_counts[b] for b in ['easy','medium','hard']},
                   pool_bands=dict(pool_bands), sites=len({t['site'] for t in tasks}),
                   label=('Difficulty first, weighted coverage at the cutoff' if difficulty_first
                          else f'Rubric score ≥{minimum_difficulty}, then weighted coverage' if minimum_difficulty
                          else 'Weighted semantic coverage'),
                   selection_note=('Retain all 1,426 tasks scoring 6+; select 574 score-5 tasks by additional semantic coverage. Actor difficulty remains unmeasured.'
                                   if difficulty_first else
                                   f'Diagnostic alternative: require rubric score ≥{minimum_difficulty}, then optimize semantic coverage across all eligible scores. This does not replace the approved cohort.' if minimum_difficulty else
                                   'This selection optimizes semantic coverage without a difficulty target.'))
    template = (REPO/'scripts/templates/arm_selected_tasks.html').read_text()
    output.write_text(template.replace('__TASK_DATA__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')))
    summary = dict(tasks=len(tasks), sites=payload['sites'],
                   selected_bands=payload['selected_bands'], eligible_bands=payload['pool_bands'],
                   cohort_sha256=digest, html_sha256=sha(output),
                   selection_label=payload['label'],
                   template_sha256=sha(REPO/'scripts/templates/arm_selected_tasks.html'),
                   implementation_sha256=sha(Path(__file__)),
                   facts_equal_source_difficulty_for_all_selected=True,
                   difficulty_definition='WebGym rubric facts: easy1–3, medium4–6, hard7+; not measured actor difficulty',
                   manifest_changed=False, api_calls=0, browser_calls=0)
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();main(a.pool, a.cohort, a.output)
