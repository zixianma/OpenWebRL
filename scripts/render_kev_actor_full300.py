#!/usr/bin/env python3
"""Render a private Kev actor review from an explicit, hash-bound offline audit.

Reads saved artifacts only. Unreviewed or changed results never acquire a valid
badge or contribute to audited scores. Does not mutate any worker/run metadata.
"""
import argparse
import base64
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

if __package__:
    from .render_jev_kev_review import target_mapping
else:
    from render_jev_kev_review import target_mapping

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / 'scripts/templates/kev_actor_full300.html'


def read(path):
    return json.loads(Path(path).read_text())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def task_key(task_id):
    # The actor runner hashes the JSON string, including its surrounding quotes.
    return sha(json.dumps(task_id, sort_keys=True).encode())


def live_status_snapshot(root, now=None):
    """Saved live liveness only; never reinterpret the immutable result audit."""
    root = Path(root).resolve()
    now = time.time() if now is None else now
    paths = dict(pointer=root.parent.parent / 'current_kev27b_actor_full300.json',
                 heartbeat=root / 'heartbeat.json', supervisor=root / 'supervisor-latest.json',
                 approval=root / 'approval.json')
    values, sources = {}, {}
    for name, path in paths.items():
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('Expected an object')
            values[name] = value
            sources[name] = dict(path=str(path), sha256=sha(raw))
        except (OSError, ValueError, TypeError) as exc:
            values[name] = {}
            sources[name] = dict(path=str(path), unavailable=type(exc).__name__)
    pointer, heartbeat, supervisor = (values[n] for n in ('pointer', 'heartbeat', 'supervisor'))
    job = str(pointer.get('job_id', ''))

    def fresh(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and 0 <= now - value <= 120)

    submitted = next((a.get('submitted_unix') for a in values['approval'].get('attempts', [])
                      if isinstance(a, dict) and str(a.get('job_id', '')) == job), None)
    if not isinstance(submitted, (int, float)) or not 0 < submitted <= now:
        submitted = None
    snapshot = dict(observed_unix=now, source='saved_live_controller_snapshot',
        separate_from_result_audit=True, sources=sources, job_id=job or None,
        verified_complete=pointer.get('verified_complete') is True,
        heartbeat_unix=heartbeat.get('updated_unix'), supervisor_checked_unix=supervisor.get('checked_unix'),
        scheduler_state=supervisor.get('state'), active_workers=heartbeat.get('active_workers', 0),
        attempt_submitted_unix=submitted, freshness_seconds=120, running=False)
    if pointer.get('verified_complete'):
        snapshot['reason'] = 'verified_complete'
    elif (not job.isdecimal() or pointer.get('run_directory') != str(root)
          or pointer.get('status') not in ('running', 'submitted')):
        snapshot['reason'] = 'missing_or_mismatched_current_pointer'
    elif str(heartbeat.get('job_id', '')) != job or not fresh(heartbeat.get('updated_unix')):
        snapshot['reason'] = 'missing_stale_or_mismatched_controller_heartbeat'
    elif (paths['supervisor'].exists() and
          (str(supervisor.get('job_id', '')) != job or not fresh(supervisor.get('checked_unix'))
           or str(supervisor.get('state', '')).split()[:1] != ['RUNNING']
           or supervisor.get('stale_heartbeat') or supervisor.get('selector_halted')
           or supervisor.get('verified_complete'))):
        snapshot['reason'] = 'stale_mismatched_or_nonrunning_supervisor'
    elif not isinstance(heartbeat.get('active_workers'), int) or heartbeat['active_workers'] <= 0:
        snapshot['reason'] = 'no_active_workers'
    else:
        snapshot.update(running=True, reason='current_controller_has_active_workers')
    return snapshot


def unfinished_status(directory, live):
    """Classify unaudited tasks using current-attempt evidence, not old audits."""
    directory = Path(directory)
    if (directory / 'result.json').exists():
        return 'unaudited'
    if not directory.exists():
        return 'pending'
    if live.get('running') and not live.get('verified_complete'):
        # Per-task heartbeats can legitimately remain unchanged for a 600-second
        # browser call. The durable task creation must belong to this attempt;
        # an old unfinished directory cannot inherit a replacement job's status.
        try:
            created = (directory / 'task.json').stat().st_mtime
            boundary = live.get('attempt_submitted_unix')
            if boundary is not None and boundary <= created <= live['observed_unix']:
                return 'running'
        except OSError:
            pass
    return 'interrupted'


class AuditedFiles:
    def __init__(self, root, directory, record):
        self.root, self.directory = root.resolve(), directory.resolve()
        self.hashes = record['artifact_sha256']
        for name in ('task.json', 'result.json', 'trajectory.json'):
            if str((directory / name).relative_to(root)) not in self.hashes:
                raise ValueError('Audit is missing ' + name)
        for name in self.hashes:
            path = (root / name).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError('Audit artifact escapes its task directory')
            self.bytes(path)

    def bytes(self, path):
        path = Path(path).resolve()
        expected = self.hashes.get(str(path.relative_to(self.root)))
        raw = path.read_bytes()
        if expected is None or sha(raw) != expected:
            raise ValueError('Artifact absent from audit or changed: ' + path.name)
        return raw

    def json(self, path):
        return json.loads(self.bytes(path))

    def states(self):
        return sorted(self.root / name for name in self.hashes
                      if Path(name).name.startswith('state-') and name.endswith('.json.gz'))


class Images:
    def __init__(self, directory):
        self.directory, self.urls = directory, {}

    def add(self, raw):
        if not raw:
            return None
        key = sha(raw)
        if key not in self.urls:
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as original:
                preview = original.convert('RGB')
                preview.thumbnail((1600, 1600))
                buffer = io.BytesIO()
                preview.save(buffer, 'JPEG', quality=82, optimize=True)
            encoded = buffer.getvalue()
            path = self.directory / (sha(encoded) + '.jpg')
            if not path.exists():
                path.write_bytes(encoded)
            elif path.read_bytes() != encoded:
                raise ValueError('Content-addressed preview changed')
            self.urls[key] = self.directory.name + '/' + path.name
        return key


def safe_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace(
        '<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def action_matches(entry, decision):
    return all(entry.get(k) == decision.get(k)
               for k in ('choice', 'operation', 'target', 'confidence', 'latency_ms', 'usage'))


def build_frames(files, trajectory, config, images):
    """Keep observations, recorded executions and target geometry distinct."""
    decisions, history = trajectory['decisions'], trajectory['history']
    executed = defaultdict(list)
    cursor = 0
    for entry in history:
        match = next((i for i in range(cursor, len(decisions))
                      if action_matches(entry, decisions[i])), None)
        if match is None:
            raise ValueError('Executed history no longer matches audited decisions')
        executed[match].append(entry)
        cursor = match + 1
    observed, frames = {}, []
    seen_decisions = seen_history = seen_helpers = 0
    final_name = f"state-{config['max_decisions'] + 1:04d}.json.gz"

    def decision_event(index):
        decision = decisions[index]
        mapping = target_mapping(decision, observed)
        return dict(number=index + 1, output={k: v for k, v in decision.items()
            if k in ('choice', 'operation', 'target', 'confidence', 'target_confidence',
                     'probabilities', 'operation_probabilities', 'target_probabilities',
                     'raw_answers', 'model', 'usage', 'latency_ms')},
            requested_target_criteria=decision['request']['questions'].get(
                decision['operation'].lower() + '_target', {}).get('criteria'),
            target_mapping=mapping, recorded_actions=executed[index],
            execution_status='recorded_executed' if executed[index] else
                ('terminal_signal' if decision['operation'] in ('DONE', 'BLOCKED')
                 else 'no_recorded_execution'))

    for path in files.states():
        state = json.loads(gzip.decompress(files.bytes(path)))
        page = state.get('page') or {}
        count = len(state['decisions'])
        if count < seen_decisions or state['decisions'] != decisions[:count]:
            raise ValueError('Snapshot decision prefix changed')
        if state['history'] != history[:len(state['history'])]:
            raise ValueError('Snapshot execution prefix changed')
        picture = images.add(base64.b64decode(page['screenshot'], validate=True)) if page.get('screenshot') else None
        if page.get('fingerprint'):
            # As in the pilot renderer, an unexecuted attempt can itself contain
            # the matching decision state. Never resolve an index on a different
            # post-action fingerprint. The box describes observed bounds only.
            observed.setdefault(page['fingerprint'], dict(actions=page.get('actions', []),
                image=picture, w=page.get('w'), h=page.get('h')))
        final = path.name == final_name
        fresh = final and state.get('final_screenshot_fresh') is True and bool(picture)
        if fresh and files.bytes(files.directory / 'final.jpg') != base64.b64decode(page['screenshot'], validate=True):
            raise ValueError('Final JPEG differs from audited terminal snapshot')
        frames.append(dict(number=int(path.name[6:10]), snapshot=path.name,
            status=state['status'], outcome='final' if final else ('initial' if not frames else 'observation'),
            final_screenshot_fresh=bool(fresh), screenshot=picture,
            url=page.get('url'), title=page.get('title', ''), page_text=page.get('text', ''),
            elapsed_ms=state.get('elapsed_ms'), total_actions=len(state['history']),
            total_decisions=count, decisions=[decision_event(i) for i in range(seen_decisions, count)],
            recorded_actions=state['history'][seen_history:],
            text_calls=state['text_calls'][seen_helpers:],
            available_actions=[{k: a[k] for k in ('id', 'node', 'kind', 'label', 'role', 'value', 'rect') if k in a}
                               for a in page.get('actions', [])],
            omitted_actions=page.get('omitted_actions', 0)))
        seen_decisions, seen_history, seen_helpers = count, len(state['history']), len(state['text_calls'])
    if seen_decisions < len(decisions):
        frames.append(dict(number=None, snapshot=None, status=trajectory['terminal'],
            outcome='unobserved', final_screenshot_fresh=False, screenshot=None,
            url=None, title='', page_text='', elapsed_ms=None, total_actions=len(history),
            total_decisions=len(decisions), decisions=[decision_event(i) for i in range(seen_decisions, len(decisions))],
            recorded_actions=history[seen_history:], text_calls=trajectory['text_calls'][seen_helpers:],
            available_actions=[], omitted_actions=0))
    return frames


def build(root, audit_path, image_dir):
    root, audit_path = Path(root).resolve(), Path(audit_path).resolve()
    plan, audit = read(root / 'plan.json'), read(audit_path)
    if audit.get('audit_version') != 1 or audit['plan_sha256'] != sha((root / 'plan.json').read_bytes()):
        raise ValueError('Audit version or plan identity does not match')
    planned = plan['tasks']
    if len(planned) != 300 or len({t['task_id'] for t in planned}) != 300 or audit['planned'] != 300:
        raise ValueError('Expected the complete 300-task plan')
    rows = {r['task_id']: r for r in audit['per_task']}
    if len(rows) != len(audit['per_task']) or not set(rows) <= {t['task_id'] for t in planned}:
        raise ValueError('Duplicate or unexpected audited task')
    issues = defaultdict(list)
    for issue in audit['issues']:
        issues[issue.get('task_id')].append(issue)
    note_path = root / 'judge-review-notes.json'
    notes = read(note_path) if note_path.exists() else []
    notes_by_task = defaultdict(list)
    for note in notes:
        if note['task_id'] not in {t['task_id'] for t in planned} or note.get('canonical_result_unchanged') is not True:
            raise ValueError('Review note has an unknown task or changes canonical scores')
        notes_by_task[note['task_id']].append(note)
    # Notes are separate human interpretations, frozen into the private review;
    # their presence never upgrades audit status or changes a canonical score.
    notes_source = dict(path=str(note_path), sha256=sha(note_path.read_bytes())) if notes else None
    live = live_status_snapshot(root)
    images, tasks = Images(image_dir), []
    for task in planned:
        tid = task['task_id']
        directory = root / 'actor/tasks' / task_key(tid)
        row = rows.get(tid)
        item = dict(**task, status=unfinished_status(directory, live), frames=[], result=None, audit_issues=issues[tid],
                    review_notes=notes_by_task[tid])
        if row and not issues[tid]:
            try:
                files = AuditedFiles(root, directory, row)
                result, trajectory = files.json(directory / 'result.json'), files.json(directory / 'trajectory.json')
                if files.json(directory / 'task.json') != task or trajectory['task'] != task:
                    raise ValueError('Task identity does not match plan')
                if result['task_id'] != tid or result['score'] != row['score'] or result['valid'] != row['valid']:
                    raise ValueError('Saved score differs from explicit audit')
                expected = 'evidence_verified' if result['valid'] else 'diagnosed_invalid'
                if row['status'] != expected:
                    raise ValueError('Unsupported per-task audit status')
                frames = build_frames(files, trajectory, plan['protocol'], images)
                rejection_path = directory / 'execution-rejections.jsonl'
                rejection_key = str(rejection_path.relative_to(root))
                rejections = ([json.loads(line) for line in files.bytes(rejection_path).decode().splitlines() if line.strip()]
                              if rejection_key in files.hashes else [])
                item.update(status=('success' if result['score'] == 1 else 'failure') if result['valid'] else 'invalid',
                    result=result, frames=frames, history=trajectory['history'], text_calls=trajectory['text_calls'],
                    recorded_execution_rejections=rejections, diagnosis=row.get('diagnosis'),
                    audit_status=row['status'], decisions=len(trajectory['decisions']), actions=len(trajectory['history']))
            except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
                item['status'] = 'unaudited'
                item['audit_issues'].append(dict(task_id=tid, error_type=type(exc).__name__, error=str(exc)))
        tasks.append(item)
    # Frame decoding can take minutes. Refresh only unaudited live states after
    # that work so a newly started task is not compared with an old timestamp.
    # A failed audit check remains unaudited even if its source is later moved.
    live = live_status_snapshot(root)
    for item in tasks:
        if (item['status'] in ('pending', 'running', 'interrupted') or
                (item['status'] == 'unaudited' and item['task_id'] not in rows
                 and not item['audit_issues'])):
            item['status'] = unfinished_status(root / 'actor/tasks' / task_key(item['task_id']), live)
    statuses = Counter(t['status'] for t in tasks)
    return dict(rendered_unix=time.time(), title='Kev 27B as the browser actor', tasks=tasks,
        protocol=plan['protocol'], resources=plan['resources'], limits=plan['limits'],
        audit={k: v for k, v in audit.items() if k != 'per_task'},
        audit_path=str(audit_path), audit_sha256=sha(audit_path.read_bytes()),
        review_notes_source=notes_source, live_status_snapshot=live,
        statuses=dict(statuses), canonical_scores_unchanged=True, images=images.urls,
        image_assets=True, privacy='Private local review; do not publish task payloads.')


def render(root, audit_path, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(REPO) or output.suffix != '.html':
        raise ValueError('Private review must be an HTML file outside the repository')
    if output.is_relative_to(root / 'actor') or output.is_relative_to(root / 'attempts'):
        raise ValueError('Review output must not overlap saved task or attempt artifacts')
    output.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output.with_suffix('.assets')
    asset_dir.mkdir(parents=True, exist_ok=True)
    payload = build(root, audit_path, asset_dir)
    template = TEMPLATE.read_text()
    if template.count('__REVIEW_DATA__') != 1:
        raise ValueError('Review template must contain exactly one data placeholder')
    temporary = output.with_suffix('.html.partial')
    temporary.write_text(template.replace('__REVIEW_DATA__', safe_json(payload)))
    temporary.replace(output)
    receipt = dict(html=str(output), sha256=sha(output.read_bytes()), bytes=output.stat().st_size,
        tasks=len(payload['tasks']), statuses=payload['statuses'], audit=str(Path(audit_path).resolve()),
        audit_sha256=payload['audit_sha256'], image_assets=str(asset_dir),
        unique_screenshots=len(payload['images']), public=False, canonical_scores_unchanged=True)
    receipt['review_notes'] = sum(len(t['review_notes']) for t in payload['tasks'])
    receipt['live_status_snapshot'] = payload['live_status_snapshot']
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.root, args.audit, args.output), indent=2))
