#!/usr/bin/env python3
"""Analyze collector-audited complete WebPRM pilot results without model calls."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.webprm_baselines import ARMS, paired_summary
from openwebrl.jev_eval import parse_verdict, write_json


def load_records(path):
    """Accept the collector's JSON array or the original audited JSONL format."""
    content=Path(path).read_text()
    records=json.loads(content) if content.lstrip().startswith('[') else [
        json.loads(line) for line in content.splitlines() if line.strip()]
    if not isinstance(records,list):
        raise ValueError('Results must be an array of records')
    return records


def audit_episodes(root, plan, budget_snapshot=None, accounting_episodes=()):
    """Independently reconcile saved cells, judge evidence and API reservations.

    A partial cohort is auditable but has no paired success estimate. Invalid
    attempted cells remain in complete-cohort denominators; absent or explicitly
    unattempted cells prevent completion. An optional archived budget snapshot
    limits accounting to that exact prefix of the live shared ledger, allowing
    a primary cohort to remain reproducible after separate supplemental calls.
    Additional episode roots supply request/response accounting evidence only;
    their outcomes, identities and completion markers never join this cohort.
    """
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    identity=manifest['identity']
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if digest!=manifest['identity_sha256']:
        raise ValueError('Manifest identity digest differs')
    if (identity['task_ids']!=plan['pilot_task_ids'] or identity['arms']!=list(ARMS)
            or identity['task_file_sha256']!=plan['task_file_sha256']):
        raise ValueError('Manifest differs from fixed plan')
    records=load_records(root/'results.json')
    expected={(task_id,arm) for task_id in plan['pilot_task_ids'] for arm in ARMS}
    indexed={}
    evidence={arm:Counter() for arm in ARMS}
    failures={arm:Counter() for arm in ARMS}
    paid_requests={}
    paid_responses={}
    judge=identity['judge']
    seen_journals=set()

    def read_journal(journal, counts, *, arm=None, valid=False, accounting_only=False):
        resolved=journal.resolve()
        if resolved in seen_journals:
            raise ValueError('Request journal supplied more than once')
        seen_journals.add(resolved)
        last_judge=None
        pending=None
        with journal.open() as handle:
            for line in handle:
                event=json.loads(line)
                if event['event']=='request':
                    pending=event
                    local=event['local']
                    kind='critic' if local else 'judge' if event['payload']['model']==judge else 'actor'
                    counts[kind+'_calls']+=1
                    if not local:
                        reservation=event['reservation']
                        if reservation in paid_requests:
                            raise ValueError('API reservation reused across requests')
                        paid_requests[reservation]=(arm,event['payload']['model'])
                elif event['event']=='response':
                    if pending is None or pending['reservation']!=event['reservation']:
                        raise ValueError('Response lacks matching prior request')
                    response=event['response']
                    if response.get('model')!=pending['payload']['model']:
                        if valid or accounting_only:
                            raise ValueError('Saved response used a different response model')
                    kind='critic' if pending['local'] else 'judge' if response.get('model')==judge else 'actor'
                    usage=response.get('usage',{})
                    for field in ('prompt_tokens','completion_tokens'):
                        counts[kind+'_'+field]+=usage.get(field,0)
                    if not pending['local']:
                        paid_responses[event['reservation']]=usage
                        if pending['payload']['model']==judge:
                            last_judge=response
                    pending=None
                elif event['event']=='response_error':
                    if (pending is None or pending['reservation']!=event['reservation']
                            or pending['local']!=event['local']
                            or pending['payload']['model']!=event['model']
                            or type(event.get('status_code')) is not int
                            or not 400 <= event['status_code'] <= 599
                            or not isinstance(event.get('error_body'),str)):
                        raise ValueError('HTTP error lacks matching prior request or valid status')
                    counts['http_errors']+=1
                    pending=None
                else:
                    raise ValueError('Unexpected request journal event')
        return last_judge

    for row in records:
        key=row['task_id'],row['arm']
        if key in indexed or key not in expected:
            raise ValueError('Duplicate or out-of-cohort record')
        if (type(row['valid']) is not bool or type(row['success']) is not int
                or row['success'] not in (0,1) or row['success'] and not row['valid']
                or type(row.get('attempted',True)) is not bool):
            raise ValueError('Malformed result fields')
        folder=root/key[0]/key[1]
        if row!=json.loads((folder/'result.json').read_text()) or row.get('identity_sha256')!=digest:
            raise ValueError('Result differs from durable cell or manifest identity')
        indexed[key]=row
        counts=evidence[key[1]]
        counts['records']+=1
        counts['attempted']+=row.get('attempted',True)
        counts['valid']+=row['valid']
        counts['successes']+=row['success']
        if not row['valid']:
            failures[key[1]][row.get('error_type','unknown')+': '+row.get('error','unspecified')]+=1
        journal=folder/'requests.jsonl'
        last_judge=None
        if journal.exists():
            last_judge=read_journal(journal,counts,arm=key[1],valid=row['valid'])
        if row['valid']:
            if not row.get('attempted',True):
                raise ValueError('Unattempted result cannot be valid')
            provenance=json.loads((folder/'provenance.json').read_text())
            if provenance.get('identity_sha256')!=digest or provenance.get('judge_model')!=judge:
                raise ValueError('Episode provenance differs from manifest')
            rollout=json.loads((folder/'rollout.json').read_text())
            if not rollout or len(rollout)!=row['turns']:
                raise ValueError('Rollout length differs from valid result')
            verdict=json.loads((folder/'verdict.json').read_text())
            if (not last_judge or last_judge['model']!=judge
                    or len(last_judge['choices'])!=1 or last_judge['choices'][0]['finish_reason']!='stop'):
                raise ValueError('Complete terminal judge response absent')
            text=last_judge['choices'][0]['message']['content'] or ''
            if (verdict['model']!=judge or verdict['raw']!=text or verdict['terminal']!=row['terminal']
                    or verdict['parsed']!=row['success'] or parse_verdict(text)!=row['success']):
                raise ValueError('Saved judge verdict does not support recorded outcome')
            from PIL import Image
            for path in [folder/'terminal.jpg']+[folder/f'step_{turn:02d}.jpg' for turn in range(row['turns'])]:
                with Image.open(path) as saved_image:
                    saved_image.verify()
    supplemental=[]
    accounting_roots={root.resolve()}
    for accounting_root in accounting_episodes:
        accounting_root=Path(accounting_root).resolve()
        if accounting_root in accounting_roots:
            raise ValueError('Accounting episode root supplied more than once or overlaps cohort root')
        if not accounting_root.is_dir():
            raise ValueError('Accounting episode root does not exist')
        accounting_roots.add(accounting_root)
        journals=sorted(accounting_root.glob('*/*/requests.jsonl'))
        counts=Counter()
        for journal in journals:
            read_journal(journal,counts,accounting_only=True)
        supplemental.append(dict(episodes_path=str(accounting_root),journals=len(journals),
                                 evidence=dict(counts)))
    budget_path=manifest.get('config',{}).get('budget_path')
    if budget_path is None:
        budget_path=root/'budget.json'
    else:
        budget_path=Path(budget_path)
        if not budget_path.is_absolute():
            budget_path=root/budget_path
    live_budget=json.loads(budget_path.read_text())
    budget=live_budget
    budget_scope=dict(scope='live_ledger',ledger_calls=len(live_budget['calls']))
    if budget_snapshot is not None:
        snapshot_path=Path(budget_snapshot)
        snapshot_bytes=snapshot_path.read_bytes()
        budget=json.loads(snapshot_bytes)
        if (budget['cap']!=live_budget['cap']
                or budget['calls']!=live_budget['calls'][:len(budget['calls'])]):
            raise ValueError('Archived budget snapshot is not an exact prefix of the live ledger with the same cap')
        budget_scope=dict(scope='archived_snapshot',snapshot_path=str(snapshot_path.resolve()),
            snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
            ledger_calls=len(budget['calls']),
            excluded_live_calls=len(live_budget['calls'])-len(budget['calls']),
            scope_note='Only the archived prefix is audited; later live calls require a separate combined audit')
    if budget['cap']!=identity['api_usd_cap'] or budget['cap']!=plan['approval']['api_cap_usd']:
        raise ValueError('API cap differs from approved manifest')
    charge=settled=reserved=0.
    supplemental_settled=supplemental_reserved=0.
    orphan_reservations=0
    ledger_ids=set()
    for position,call in enumerate(budget['calls']):
        if call['id']!=position:
            raise ValueError('API ledger IDs are not unique sequential reservations')
        ledger_ids.add(position)
        if not math.isfinite(call['reserved']) or call['reserved']<0:
            raise ValueError('Invalid API reservation')
        if call['state']=='reserved':
            amount=call['reserved']
            reserved+=amount
        elif call['state'] in ('settled','overrun'):
            usage=call['usage']
            if any(type(usage[field]) is not int or usage[field]<0 for field in ('prompt_tokens','completion_tokens')):
                raise ValueError('Invalid settled token usage')
            rates=identity['prices_usd_per_token'][call['model']]
            amount=usage['prompt_tokens']*rates[0]+usage['completion_tokens']*rates[1]
            if not math.isclose(amount,call['charged'],rel_tol=1e-10,abs_tol=1e-9):
                raise ValueError('API ledger charge differs from token usage')
            if position not in paid_responses or paid_responses[position]!=usage:
                raise ValueError('Settled API usage lacks matching saved response')
            settled+=amount
        else:
            raise ValueError('Unknown API reservation state')
        charge+=amount
        if position in paid_requests:
            arm,model=paid_requests[position]
            if model!=call['model']:
                raise ValueError('API ledger model differs from saved request')
            if arm is not None:
                evidence[arm]['api_'+('reserved' if call['state']=='reserved' else 'settled')+'_usd']+=amount
            elif call['state']=='reserved':
                supplemental_reserved+=amount
            else:
                supplemental_settled+=amount
        else:
            orphan_reservations+=1
    if set(paid_requests)-ledger_ids:
        raise ValueError('Paid request lacks durable reservation')
    if not math.isclose(charge,budget['charged'],rel_tol=1e-10,abs_tol=1e-8) or charge>budget['cap']+1e-8:
        raise ValueError('API accounting total differs or exceeds cap')
    if supplemental:
        budget_scope.update(accounting_only_roots=supplemental,
            accounting_only_settled_usd=supplemental_settled,
            accounting_only_reserved_usd=supplemental_reserved,
            ledger_path=str(budget_path.resolve()),
            accounting_note='Additional roots reconcile the shared ledger only; cohort outcomes and arm evidence exclude them')
    attempted={key for key,row in indexed.items() if row.get('attempted',True)}
    missing=sorted(expected-attempted)
    audit=dict(status='partial' if missing else 'completed',planned_episodes=len(expected),
        records=len(records),attempted_episodes=len(attempted),valid=sum(row['valid'] for row in records),
        missing_episodes=[dict(task_id=task_id,arm=arm) for task_id,arm in missing],
        identity_sha256=digest,evidence_by_arm={arm:dict(counts) for arm,counts in evidence.items()},
        failures_by_arm={arm:dict(counts) for arm,counts in failures.items()},
        api=dict(**budget_scope,cap_usd=budget['cap'],charged_usd=charge,settled_usd=settled,
                 unresolved_reserved_usd=reserved,orphan_reservations=orphan_reservations))
    if not missing:
        complete=json.loads((root/'complete.json').read_text())
        if complete['episodes']!=len(expected) or complete['valid']!=audit['valid']:
            raise ValueError('Completion marker differs from independent evidence audit')
    return records,audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--audited-records', type=Path)
    inputs.add_argument('--episodes', type=Path, help='Independently audit collector directory before analysis')
    p.add_argument('--budget-snapshot', type=Path,
        help='Audit an archived exact prefix of the live shared budget (requires --episodes)')
    p.add_argument('--accounting-episodes', type=Path, action='append', default=[],
        help='Additional episode directory used only to reconcile the shared ledger; repeatable (requires --episodes)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resamples', type=int, default=10000)
    args = p.parse_args()
    if args.budget_snapshot and not args.episodes:
        p.error('--budget-snapshot requires --episodes')
    if args.accounting_episodes and not args.episodes:
        p.error('--accounting-episodes requires --episodes')
    plan = json.loads(args.plan.read_text())
    if args.episodes:
        records,audit=audit_episodes(args.episodes,plan,budget_snapshot=args.budget_snapshot,
                                   accounting_episodes=args.accounting_episodes)
        summary=paired_summary(records,plan['pilot_task_ids'],args.resamples) if audit['status']=='completed' else {}
        summary['audit']=audit
    else:
        records=load_records(args.audited_records)
        summary=paired_summary(records,plan['pilot_task_ids'],args.resamples)
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
