"""Write a local live dashboard and deduplicated job events; never send messages."""
import hashlib
import html
import json
from pathlib import Path

def result_text(row):
    if not row.get('complete'):
        return ''
    if row.get('kind') == 'mig':
        return 'MIG capacity, selection and two browser trajectories passed; diagnostic only'
    if row.get('kind') == 'coverage':
        report=row.get('coverage_summary',{})
        before=report.get('same_pool_twenty_percent',{})
        after=report.get('four_turn_coverage',{})
        return (f"Failure labels {before.get('labels',0)} -> {after.get('labels',0)} on "
                f"{after.get('groups',0)} groups; zero optimizer updates")
    if row.get('kind') == 'quality':
        return (f"{row.get('completed_tasks',0)}/192 candidate sets; "
                f"{row.get('teacher_labels',0)}/231 valid teacher labels; "
                f"${row.get('api_charged_or_reserved_usd',0):.2f} charged/reserved")
    if row.get('kind') == 'task_success':
        stages=row.get('task_success_summary',{}).get('stages',{})
        values=[]
        for actor,s in stages.items():
            modes=s['by_mode']
            values.append(f"{actor}: ARM {100*modes['arm']['overall']:.1f}% overall")
        return '; '.join(values)
    if row.get('kind') == 'eval':
        n, valid, wins = (row.get(key, 0) for key in ('completed_tasks', 'valid', 'successes'))
        if not n:
            return ''
        valid_rate = f'{100*wins/valid:.2f}%' if valid else 'undefined'
        return f'{wins}/{n}: {100*wins/n:.2f}% overall; {valid_rate} valid-only ({valid} valid)'
    if row.get('kind') == 'rescue':
        summary = row.get('yield_summary', {})
        if not summary.get('complete'):
            return ''
        n = summary['selected_tasks']
        modes, paired = summary['by_mode'], summary['paired']
        return (f"Rescued tasks: ARM {modes['arm']['successes']}/{n}; "
                f"one actor retry {modes['actor0']['successes']}/{n}; "
                f"five actor retries {paired['actor5_any_success']}/{n}")
    return ''


def compact(row):
    out = {key: row.get(key) for key in ('job', 'label', 'kind', 'stage',
        'durable_iteration', 'collection', 'completed_optimizer_updates',
        'completed_tasks', 'smoke_attempts', 'screen_attempts', 'retry_attempts', 'complete',
        'validation_updates', 'validation_expected_updates', 'validation_passed',
        'teacher_labels', 'api_charged_or_reserved_usd', 'candidate_sets_by_actor')}
    out['state'] = row.get('slurm', {}).get('state', 'UNKNOWN')
    out['elapsed'] = row.get('slurm', {}).get('elapsed', '')
    out['node'] = row.get('slurm', {}).get('node', '')
    out['alerts'] = row.get('alerts', [])
    out['result'] = result_text(row)
    out['root'] = row.get('root', '')
    return out


def record_events(report, control):
    """Persist transitions once, including across supervisor restarts."""
    control = Path(control)
    state_path = control/'event-state.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    events_path = control/'events.jsonl'
    existing = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    ids = {event['event_id'] for event in existing}
    current, emitted = {}, []
    for row in report['jobs']:
        small = compact(row)
        signature = {key: small[key] for key in ('state', 'stage', 'durable_iteration',
            'collection', 'complete', 'alerts', 'result')}
        old = previous.get(row['job'])
        current[row['job']] = signature
        # Seed existing history silently. Only new transitions trigger events.
        if old is None or signature == old:
            continue
        if signature['result'] and signature['result'] != old.get('result'):
            kind = 'result_ready'
        elif signature['alerts'] != old.get('alerts') and signature['alerts']:
            kind = 'attention_needed'
        elif signature['durable_iteration'] != old.get('durable_iteration'):
            kind = 'checkpoint_saved'
        else:
            kind = 'state_changed'
        identity = dict(job=row['job'], kind=kind, signature=signature)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key in ids:
            continue
        event = dict(small)
        event.update(event_id=key, checked_utc=report['checked_utc'], kind=kind,
                     job_kind=small['kind'])
        emitted.append(event); ids.add(key)
    if emitted:
        with events_path.open('a') as handle:
            for event in emitted:
                handle.write(json.dumps(event)+'\n')
            handle.flush()
    atomic_text(state_path, json.dumps(current, indent=2)+'\n')
    return (existing+emitted)[-30:]


def atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(text)
    temporary.replace(path)


def write_dashboard(report, control, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [compact(row) for row in report['jobs']]
    events = record_events(report, control)
    data = dict(checked_utc=report['checked_utc'], jobs=rows,
                slurm_jobs=report.get('live_slurm_jobs', []), events=events,
                notification_delivery='Local files only; no automatic chat or external messages',
                lightweight_check_seconds=60, detailed_check_seconds=900)
    atomic_text(output/'live-status.json', json.dumps(data, indent=2)+'\n')
    e = lambda value: html.escape(str(value if value is not None else '—'), quote=True)
    terminal = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
    finished = [row for row in rows if row['state'].split()[0].rstrip('+') in terminal]
    active = [row for row in rows if row not in finished]
    finished = sorted(finished, key=lambda row:int(row['job']), reverse=True)[:8]
    rendered = []
    for row in active+finished:
        progress = (f"saved {row['durable_iteration']}; current {row['collection']}"
                    if row['kind']=='train' else
                    f"{row['validation_updates'] or 0}/{row['validation_expected_updates'] or '?'} validation updates"
                    if row['kind']=='diagnostic' and row['validation_expected_updates'] is not None else
                    f"{row['completed_tasks']}/192 candidate sets; {row['teacher_labels']}/231 labels"
                    if row['kind']=='quality' else
                    f"{row['completed_tasks']}/{row.get('planned_tasks',200)} ARM task runs"
                    if row['kind']=='task_success' else
                    f"smoke {row['smoke_attempts']}; screen {row['screen_attempts']}; retries {row['retry_attempts']}"
                    if row['kind']=='rescue' else f"{row['completed_tasks']} tasks saved")
        detail = row['result'] or '; '.join(row['alerts']) or row['stage'] or 'Waiting'
        rendered.append('<tr>'+''.join(f'<td>{e(value)}</td>' for value in
            (row['job'],row['label'],row['state'],progress,detail))+'</tr>')
    known = {row['job'] for row in rows}
    other = [job for job in data['slurm_jobs'] if job['job'] not in known]
    other_html = ''.join('<tr>'+''.join(f'<td>{e(job.get(key))}</td>' for key in
        ('job','name','state','elapsed','node'))+'</tr>' for job in other)
    event_items = []
    for event in reversed(events):
        detail = event['result'] or '; '.join(event['alerts'])
        if not detail:
            detail = (f"Durable iteration {event['durable_iteration']}"
                      if event['kind']=='checkpoint_saved' else event['state'])
        event_items.append(f"<li>{e(event['checked_utc'])} · {e(event['job'])} · "
                           f"{e(event['kind'])}: {e(detail)}</li>")
    event_html = ''.join(event_items)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta http-equiv="refresh" content="60"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenWebRL live jobs</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 22px;color:#172033;background:#f6f8fb}}
table{{border-collapse:collapse;background:white;width:100%;margin:16px 0 30px}}th,td{{text-align:left;padding:10px 14px;border-bottom:1px solid #dce3ed;vertical-align:top}}
th{{background:#e9eff7}}h1{{margin-bottom:6px}}p{{color:#44536b}}a{{color:#175bb5}}li{{margin-bottom:8px}}
</style><h1>OpenWebRL live jobs</h1>
<p>Checked {e(data['checked_utc'])}. Refreshes every 60 seconds; detailed health checks every 15 minutes.
Result rates appear only after completion is verified. This page does not send chat notifications.</p>
<p><a href="../../ARM_SUMMARY.md">Methods, results and curves</a> · <a href="live-status.json">Machine-readable status</a></p>
<h2>Tracked ARM jobs and recent completions</h2><table><tr><th>Job</th><th>Run</th><th>State</th><th>Progress</th><th>Result / attention</th></tr>{''.join(rendered)}</table>
<h2>Other current Slurm jobs — scheduler status only</h2><table><tr><th>Job</th><th>Name</th><th>State</th><th>Elapsed</th><th>Node / reason</th></tr>{other_html or '<tr><td colspan="5">None</td></tr>'}</table>
<h2>Recent events</h2><ul>{event_html or '<li>Monitor initialized; waiting for new transitions.</li>'}</ul></html>'''
    atomic_text(output/'live-status.html', page)
    return data
