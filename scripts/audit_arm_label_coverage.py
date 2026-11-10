#!/usr/bin/env python3
"""Bounded CPU audit of saved ARM labels; never runs models or loads tensors."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import resource
import statistics
import time

from audit_arm_preference_pairs import canonical, coordinate_distance, parse_action, schemas


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def disposition(r):
    if r.get('eligible'):
        return 'usable'
    reason = r.get('reason', 'missing_reason')
    error = r.get('error', '')
    if reason != 'label_unavailable':
        return reason
    if error == 'duplicate_action':
        return 'duplicate_action'
    if error == 'truncated_or_empty':
        return 'truncated_or_empty'
    if r.get('selector_requests', 0):
        if r.get('error_type') in ('ValueError', 'JSONDecodeError'):
            return 'selector_parse_or_value_error'
        return 'selector_transport_or_timeout'
    if r.get('candidates'):
        return 'candidate_parse_or_schema_error'
    return 'candidate_generation_transport_or_timeout'


def diversity(r):
    cs = r.get('candidates', [])
    if len(cs) != 5:
        return {'stratum': 'no_complete_candidate_set'}
    actions, reasoning, errors = [], [], Counter()
    try:
        tools = schemas(r['prompt'])
    except (ValueError, KeyError):
        return {'stratum': 'missing_tool_schema'}
    for c in cs:
        response = c.get('response', '')
        try:
            a, spans = parse_action(response, tools)
            actions.append(canonical(a))
        except (ValueError, TypeError, KeyError) as e:
            actions.append(None)
            errors[type(e).__name__ if type(e) is not ValueError else str(e)] += 1
        text = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.S)
        reasoning.append(set(re.findall(r'\w+', text.lower())))
    lengths = [len(c.get('response_token_ids', [])) for c in cs]
    pairs = [(a, b) for i, a in enumerate(reasoning) for b in reasoning[i+1:]]
    out = {'parse_errors': dict(errors), 'valid_candidates': sum(a is not None for a in actions),
           'stopped_candidates': sum(c.get('finish_type') == 'stop' for c in cs),
           'response_tokens_mean': statistics.mean(lengths),
           'response_tokens_std': statistics.pstdev(lengths),
           'reasoning_jaccard_distance': statistics.mean(1-len(a & b)/len(a | b) if a | b else 0 for a, b in pairs)}
    if errors:
        out['stratum'] = 'invalid_candidates'
        return out
    classes = list(dict.fromkeys(actions))
    counts = Counter(actions)
    n = len(classes)
    out.update(unique_actions=n, multiplicities=sorted(counts.values(), reverse=True),
               action_class_ids=[classes.index(a) for a in actions],
               action_frequency_entropy=-sum((v/5)*math.log(v/5) for v in counts.values())/math.log(5),
               unique_tool_sequences=len({tuple(x['name'] for x in json.loads(a)) for a in actions}),
               stratum='all_same_action' if n == 1 else ('all_distinct_actions' if n == 5 else 'partially_duplicated_actions'))
    # Complete-link coordinate clusters; all non-coordinate arguments must match.
    # These are geometric proxies, not verified equivalence of clicked DOM elements.
    parsed = [json.loads(a) for a in sorted(classes)]
    for tolerance in (10, 25):
        clusters = []
        for a in parsed:
            for cluster in clusters:
                distances = [coordinate_distance(a, b) for b in cluster]
                if all(d is not None and d <= tolerance for d in distances):
                    cluster.append(a)
                    break
            else:
                clusters.append([a])
        out[f'coordinate_clusters_{tolerance}'] = len(clusters)
    return out


def mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (float, int))]
    return statistics.mean(values) if values else None


def audit(roots, output, per_stratum=30):
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'runs': {},
              'confidence_available': False, 'gpu_used': False,
              'limitations': ['No scores/logits were archived; correlations with ARM confidence require new inference.',
                 'Four-turn usable-label estimates assume unchanged yield on historically retained failure groups; they are not measured counterfactual labels.',
                 'Canonical action equality is exact tool/argument equality, not semantic equivalence of nearby clicks.',
                 'Reasoning diversity is lexical Jaccard distance, not semantic or model-embedding variance.']}
    panel_candidates = defaultdict(list)
    with (output/'state_metrics.jsonl').open('w') as state_out:
        for root in roots:
            run_counts, sampled_counts, bins = Counter(), Counter(), defaultdict(list)
            candidate_counts, parse_errors, unique_counts = Counter(), Counter(), Counter()
            collections, coverage, timings = [], [], []
            configs = sorted((root/'iterations').glob('*/calibration.json'))
            for config_path in configs:
                work = config_path.parent
                if not (work/'collection_complete.json').exists():
                    continue
                headers, index_map = {}, {}
                for p in sorted((work/'groups').glob('*/*.json')):
                    header = json.loads(p.read_text())
                    headers[str(header['group_id'])] = (p, header)
                    for row in header['rows']:
                        index_map[row['index']] = {'accepted': header['accepted'], 'group_id': str(header['group_id']),
                            'task_id': row.get('task_id'), 'reward': row.get('reward'), 'archive': str(p.with_suffix('.pt'))}
                aux_path = work/'failure_auxiliary.json'
                retained_groups = set()
                if aux_path.exists():
                    aux = json.loads(aux_path.read_text())
                    retained_groups = {str(r['group_index']) for r in aux['records']}
                    if len(retained_groups) != aux['failure_groups']:
                        raise ValueError('Cannot identify every retained failure group')
                failure_lengths = Counter()
                for group_id in retained_groups:
                    for row in headers[group_id][1]['rows']:
                        failure_lengths[str(row['trajectory_id'])] += 1
                failure_counts, batch_counts = Counter(), Counter()
                for p in sorted((work/'labels').glob('*.json')):
                    raw = p.read_text()
                    data = json.loads(raw)
                    for r in data.get('records', []):
                        reason = disposition(r)
                        run_counts[reason] += 1
                        batch_counts[reason] += 1
                        if r.get('sampled'):
                            sampled_counts[reason] += 1
                        candidate_counts['actor_requests'] += r.get('candidate_requests', 0)
                        candidate_counts['selector_requests'] += r.get('selector_requests', 0)
                        if str(r.get('trajectory_id')) in failure_lengths:
                            failure_counts['recorded_turns'] += 1
                            failure_counts['attempted'] += bool(r.get('sampled'))
                            failure_counts['usable'] += bool(r.get('eligible'))
                        if isinstance(r.get('selector_seconds'), (int, float)):
                            timings.append(r['selector_seconds'])
                        if not r.get('candidates'):
                            continue
                        d = diversity(r)
                        candidate_counts['sets'] += 1
                        candidate_counts['individual_candidates'] += len(r['candidates'])
                        candidate_counts['parseable_candidates'] += d.get('valid_candidates', 0)
                        candidate_counts['stopped_candidates'] += d.get('stopped_candidates', 0)
                        parse_errors.update(d.get('parse_errors', {}))
                        if 'unique_actions' in d:
                            unique_counts[d['unique_actions']] += 1
                        idx = (r.get('context') or {}).get('parent_sample_index')
                        join = index_map.get(idx, {})
                        uid = f'{root.name}:{work.name}:{r["trajectory_id"]}:{r["turn"]}'
                        entry = dict(**join)
                        entry.update(id=uid, run=root.name, iteration_index=int(work.name),
                                     label_file=str(p), record_turn=r['turn'], task_id=str(r.get('task_id')),
                                     trajectory_id=str(r['trajectory_id']), sample_index=idx,
                                     label_file_sha256=digest(raw), disposition=reason, **d)
                        state_out.write(json.dumps(entry)+'\n')
                        bins[d['stratum']].append({k:v for k,v in d.items() if isinstance(v,(int,float))})
                        if (d['stratum'] in ('all_same_action','partially_duplicated_actions','all_distinct_actions')
                                and d['stopped_candidates'] == 5 and join.get('archive')):
                            bucket = panel_candidates[d['stratum']]
                            bucket.append(entry)
                            if len(bucket) > 1000:
                                bucket.sort(key=lambda e:digest(e['id']))
                                del bucket[500:]
                if failure_lengths:
                    attempts4 = sum(min(4, n) for n in failure_lengths.values())
                    coverage.append({'iteration_index':int(work.name), 'groups':len(retained_groups),
                        'trajectories':len(failure_lengths), 'turns':sum(failure_lengths.values()),
                        'observed_attempts':failure_counts['attempted'], 'observed_usable':failure_counts['usable'],
                        'four_turn_attempts':attempts4, 'trajectory_lengths':dict(Counter(failure_lengths.values()))})
                collections.append({'iteration_index':int(work.name), 'label_dispositions':dict(batch_counts)})
                print(json.dumps({'run':root.name,'collection':work.name,'turns':sum(batch_counts.values())}),flush=True)
                time.sleep(.03)
            cov = {k:sum(r[k] for r in coverage) for k in ['groups','trajectories','turns','observed_attempts','observed_usable','four_turn_attempts']}
            if cov['observed_attempts']:
                cov.update(observed_label_yield=cov['observed_usable']/cov['observed_attempts'],
                           projected_four_turn_usable=cov['four_turn_attempts']*cov['observed_usable']/cov['observed_attempts'],
                           attempted_coverage_multiplier=cov['four_turn_attempts']/cov['observed_attempts'])
            result['runs'][root.name] = {'collections':len(collections), 'all_turn_dispositions':dict(run_counts),
                'sampled_turn_dispositions':dict(sampled_counts),
                'candidate_counts':dict(candidate_counts), 'individual_parse_errors':dict(parse_errors),
                'unique_action_counts':dict(unique_counts),
                'candidate_strata':{k:{'states':len(v), **{f:mean(v,f) for f in ['unique_actions','unique_tool_sequences','coordinate_clusters_10','coordinate_clusters_25','response_tokens_mean','response_tokens_std','reasoning_jaccard_distance','action_frequency_entropy']}} for k,v in bins.items()},
                'selector_seconds_median':statistics.median(timings) if timings else None,
                'failure_coverage':cov, 'failure_coverage_by_collection':coverage, 'collections_detail':collections}
    used_tasks, panel = set(), []
    for name in ['all_same_action','partially_duplicated_actions','all_distinct_actions']:
        n = 0
        for row in sorted(panel_candidates[name],key=lambda e:digest(e['id'])):
            if row['task_id'] in used_tasks:
                continue
            used_tasks.add(row['task_id']); row['split']='development' if n < per_stratum//2 else 'validation'
            panel.append(row);n+=1
            if n == per_stratum:
                break
    result['replay_panel'] = {'states':len(panel), 'unique_tasks':len(used_tasks),
                              'strata':dict(Counter(r['stratum'] for r in panel)),
                              'split':dict(Counter(r['split'] for r in panel))}
    result['wall_seconds'] = time.monotonic()-start
    result['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'confidence_panel.json').write_text(json.dumps(panel,indent=2)+'\n')
    write_report(result, output)
    return result


def write_report(result, output):
    completed = list(output.glob('confidence-replay-*/confidence-report.html'))
    confidence_note = 'Confidence measurements are pending.'
    if completed:
        latest = max(completed, key=lambda p:p.stat().st_mtime)
        confidence_note = '<a href="'+html.escape(str(latest.relative_to(output)))+'">Confidence replay completed: results and plots</a>.'
    body = ['<!doctype html><meta charset="utf-8"><title>ARM label coverage audit</title>',
            '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#18212b}table{border-collapse:collapse;margin:20px 0}td,th{padding:8px 14px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}small{color:#555}</style>',
            '<h1>ARM label coverage and candidate diversity</h1><p>CPU audit of archived training data. '+confidence_note+'</p>']
    for name,d in result['runs'].items():
        body += ['<h2>'+html.escape(name)+'</h2><p>'+str(d['collections'])+' completed collections.</p>',
                 '<table><tr><th>Attempt result</th><th>Count</th><th>% of sampled turns</th></tr>']
        den=sum(d['sampled_turn_dispositions'].values())
        for k,v in sorted(d['sampled_turn_dispositions'].items(),key=lambda x:-x[1]):
            body.append(f'<tr><td>{html.escape(k)}</td><td>{v:,}</td><td>{100*v/den:.2f}%</td></tr>')
        body += ['</table><table><tr><th>Candidate category</th><th>States</th><th>Reasoning lexical distance</th></tr>']
        for k,v in d['candidate_strata'].items():
            value=v['reasoning_jaccard_distance']
            body.append(f'<tr><td>{html.escape(k)}</td><td>{v["states"]:,}</td><td>{value:.3f}</td></tr>' if value is not None else f'<tr><td>{html.escape(k)}</td><td>{v["states"]:,}</td><td>—</td></tr>')
        body += ['</table>']
        c=d['failure_coverage']
        if c['observed_attempts']:
            body.append(f'<p>Retained failure pool: {c["observed_attempts"]:,} attempted / {c["observed_usable"]:,} usable labels over {c["turns"]:,} turns. Four sampled turns per trajectory would attempt {c["four_turn_attempts"]:,} labels ({c["attempted_coverage_multiplier"]:.2f}×); projected usable labels {c["projected_four_turn_usable"]:.0f} at unchanged yield. This last number is a projection.</p>')
    body += ['<h2>Limitations</h2><ul>']+[f'<li>{html.escape(x)}</li>' for x in result['limitations']]+['</ul>']
    (output/'report.html').write_text('\n'.join(body))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(240,245))
    audit(a.run_root,a.output)
