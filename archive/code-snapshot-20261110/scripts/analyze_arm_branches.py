#!/usr/bin/env python3
"""Audit durable branch receipts and write aggregate, paired diagnostic results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.arm_branch_analysis import paired_summary, paired_by_phase
from openwebrl.arm_branch_teacher import parse
from openwebrl.controlled_sft_state import digest


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_activation_failure(root, row, attempt, release):
    """Retain a proven post-commit release-read failure as an invalid attempt."""
    assert row['valid'] is False and row['reward'] is None
    assert row['reason'] == 'anchor_not_captured' and row['browser_closed'] is True
    assert str(Path(row['result_path']).parent) in release['attempt_directories']
    assert attempt['ended'] >= release['released_unix']
    log = Path(root)/'scheduler-attempts'/str(attempt['job_id'])/(attempt['item_id']+'.log')
    text = log.read_text()
    assert "release = json.loads((self.state_directory/'release.json').read_text())" in text
    assert 'json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)' in text
    for path in (Path(row['result_path']).parent/'events').glob('[0-9]*.json'):
        event = read(path)
        assert event['event'] != 'candidate_pre_dispatch_verified'
        assert not (event['event'] == 'forced_response' and event['record']['role'] == 'candidate')
    return dict(state=row['state'], candidate=row['candidate'], repeat=row['repeat'],
                attempt_id=row['attempt_id'], kind='release_json_read_failed_after_commit',
                log_sha256=sha(log))


def audit(root):
    root=Path(root); plan=read(root/'plan.json'); approval=read(root/'approval.json')
    assert digest(plan)==approval['plan_sha256']
    assert plan['limits']==approval['limits'] and plan['resources']==approval['resources']
    with sqlite3.connect(root/'state.sqlite3') as db:
        db.row_factory=sqlite3.Row
        attempts=[dict(x) for x in db.execute('SELECT * FROM attempts ORDER BY started')]
        counters=dict((x['key'],x['value']) for x in db.execute('SELECT * FROM counters'))
        unsettled=db.execute("SELECT COUNT(*) FROM reservations WHERE status='reserved'").fetchone()[0]
    latest={x['item_id']:x for x in attempts}
    records=[]; judgments=[]; complete_states=[]; pending_states=[]; rejected=Counter(); diversity=[]
    state_phases={}; activation_failures=[]
    job_phases={str(a['job_id']):Path(a.get('repair','original')).name for a in approval['attempts']}
    regular_teacher_usage=Counter(); response_models=Counter(); judge_models=Counter(); request_count=0
    def result(item_id):
        attempt=latest.get(item_id)
        if not attempt or attempt['status']!='finished': return None
        row=json.loads(attempt['result_json'])
        assert sha(row['result_path'])==row['result_sha256']
        assert row['attempt_id']==attempt['attempt_id'] and row['browser_closed'] is True
        return row
    for state in plan['states']:
        sid=state['id']; directory=root/'states'/sid
        if (directory/'reject.json').exists():
            rejected[read(directory/'reject.json')['reason']]+=1
        if not (directory/'release.json').exists(): continue
        release=read(directory/'release.json'); ref=read(directory/'anchor-reference.json')
        assert sha(ref['path'])==ref['sha256']==release['anchor_sha256']
        assert release['cohort_committed_before_candidate_dispatch'] is True
        anchor=read(ref['path']); identities=set(); proof_directories=set()
        assert len(release['ready_receipts'])==15
        for proof in release['ready_receipts']:
            assert sha(proof['path'])==proof['sha256']; ready=read(proof['path'])
            assert ready['anchor_sha256']==ref['sha256'] and ready['checks'] and all(x['passed'] for x in ready['checks'])
            assert str(Path(proof['path']).parent)==ready['attempt_directory']
            proof_directories.add(ready['attempt_directory'])
            identities.add((ready['candidate'],ready['repeat']))
        assert identities=={(c,r) for c in range(5) for r in range(3)}
        assert len(set(release['attempt_directories']))==15
        assert proof_directories==set(release['attempt_directories'])
        branch_rows=[result(f'{sid}-c{c}-r{r}') for c in range(5) for r in range(3)]
        teacher=result(sid+'-teacher')
        if any(x is None for x in branch_rows) or teacher is None:
            pending_states.append(sid); continue
        for row in branch_rows:
            assert row['state']==sid and row['mode']=='branch'
            if not row['released']:
                activation_failures.append(verify_activation_failure(root, row,
                    latest[f"{sid}-c{row['candidate']}-r{row['repeat']}"], release))
            assert str(Path(row['result_path']).parent) in release['attempt_directories']
            assert latest[f"{sid}-c{row['candidate']}-r{row['repeat']}"]['attempt_id']==row['attempt_id']
            for event_path in (Path(row['result_path']).parent/'events').glob('[0-9]*.json'):
                event=read(event_path)
                if event['event']=='judge_received':
                    model=event['record']['response'].get('model')
                    assert model=='o4-mini-2025-04-16';judge_models[model]+=1
        assert sha(teacher['judgments_path'])==teacher['judgments_sha256']
        panel=read(teacher['judgments_path']); assert len(panel)==14
        expected={('before',None,t) for t in (0,1,2,'permuted')}
        expected|={('after',r,t) for r in range(3) for t in (0,1,2)}
        expected.add(('after',0,'permuted'))
        assert {(x['condition'],x['execution_repeat'],x['teacher_repeat']) for x in panel}==expected
        for row in panel:
            assert row['state']==sid and row['form']=='index'
            if row['valid']:
                assert set(row['output'])=={'selection'}
                assert row['candidate']==row['order'][row['output']['selection']-1]
                if isinstance(row['teacher_repeat'],int):
                    for token in ('input_tokens','output_tokens'):
                        regular_teacher_usage[row['condition']+'_'+token]+=row['usage'][token]
        regular_input_identities=set()
        for event_path in (Path(teacher['result_path']).parent/'events').glob('[0-9]*.json'):
            event=read(event_path); payload=event['record']
            if event['event']=='teacher_request':
                request=payload['request']; assert digest(request)==payload['request_sha256']
                assert request['model']=='gpt-6-luna' and request['reasoning']=={'effort':'high'}
                assert request['max_output_tokens']==4096
                assert request['text']['format']['schema']['additionalProperties'] is False
                assert set(request['text']['format']['schema']['properties'])=={'selection'}
                content=request['input'][0]['content']; view=json.loads(content[0]['text'])
                assert set(view)<= {'task','history','before_observation','candidates','immediate_results'}
                condition=payload['label']['condition']
                assert ('immediate_results' in view)==(condition=='after')
                if isinstance(payload['label']['teacher_repeat'],int):
                    view.pop('immediate_results',None)
                    regular_input_identities.add(digest(dict(view=view,original_image=content[2],instructions=request['instructions'])))
                request_count+=1
            elif event['event']=='teacher_response' and payload['status']==200:
                raw=payload['response']; response_models[raw.get('model')]+=1
                # Invalid/incomplete calls remain invalid controls, not silently repaired here.
                if raw.get('status')=='completed': parse(raw,'index',5)
        assert len(regular_input_identities)==1, 'Paired teacher requests changed common inputs'
        complete_states.append(sid); records.extend(branch_rows); judgments.extend(panel)
        state_phases[sid]='+'.join(sorted({job_phases[str(latest[f"{sid}-c{c}-r{r}"]['job_id'])]
            for c in range(5) for r in range(3)}))
        diversity.append(len({x['action_key'] for x in anchor['candidates']}))
    analysis=paired_summary(records,judgments)
    analysis['activation_failures']=activation_failures
    jobs=[str(x['job_id']) for x in approval['attempts']]
    raw=subprocess.check_output(['sacct','-X','-j',','.join(jobs),'--format=JobIDRaw,State,ElapsedRaw','-n','-P'],text=True)
    scheduler=[]
    for line in raw.splitlines():
        jid,state,seconds=line.split('|')[:3]
        if jid in jobs:scheduler.append(dict(job_id=jid,state=state,elapsed_seconds=int(seconds)))
    assert {x['job_id'] for x in scheduler}==set(jobs)
    total_seconds=sum(x['elapsed_seconds'] for x in scheduler)
    assert total_seconds<=plan['resources']['total_seconds']
    for count,limit in [('browser_sessions','browser_sessions'),('local_sft_generations','local_sft_generations'),
                        ('luna_selector_requests','luna_selector_requests'),('luna_usd','luna_usd'),
                        ('om2w_judge_http_attempts','canonical_judge_http_attempts'),('canonical_judge_usd','canonical_judge_usd')]:
        assert counters.get(count,0)<=plan['limits'][limit]+1e-8
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED'}
    final_accounting=(all(x['state'].split()[0] in terminal for x in scheduler)
                      and not unsettled and counters.get('active_browsers',0)==0)
    aggregate={k:v for k,v in analysis.items() if k not in ('by_state','excluded_states','activation_failures')}
    aggregate['committed_activation_failure_count']=len(activation_failures)
    aggregate['post_release_continuation_attempts']=len(records)-len(activation_failures)
    aggregate['activation_failure_note']='Committed release-read failures stay invalid and count zero; no execution is invented and no attempt is replaced.'
    aggregate['exclusion_counts']=dict(Counter(x['reason'] for x in analysis['excluded_states']))
    aggregate['execution_phase_sensitivity']=paired_by_phase(records,judgments,state_phases)
    aggregate['execution_phase_note']='All completed primary outcomes are retained. Repair-phase cohorts differ; phase comparisons are descriptive, not randomized effects.'
    aggregate.update(epoch=time.time(),plan_sha256=digest(plan),protocol_stage=plan['protocol_stage'],
        target_states=100,target_continuations=1500,completed_states=len(complete_states),
        completed_continuations=len(records),valid_continuations=sum(x['valid'] for x in records),
        released_states_pending=len(pending_states),replay_exclusions=dict(rejected),
        candidate_diversity_counts=dict(Counter(diversity)),verified_teacher_requests=request_count,
        valid_teacher_judgments=sum(x['valid'] for x in judgments),
        teacher_response_models=dict(response_models),judge_response_models=dict(judge_models),regular_teacher_tokens=dict(regular_teacher_usage),
        all_attempt_counters=counters,scheduler=scheduler,all_attempt_allocation_seconds=total_seconds,
        allocation_seconds_cap=plan['resources']['total_seconds'],unsettled_reservations=unsettled,
        final_accounting_verified=final_accounting,
        collection_target_verified=len(records)==1500 and analysis['analyzed_states']==100 and sum(x['valid'] for x in judgments)==1400 and not activation_failures,
        verified_complete=False,
        status='partial' if len(records)<1500 else 'collected_requires_final_artifact_review',
        scope='Continuation and teacher diagnostic on replayable training states; not a full OM2W evaluation. '
              'All-attempt counters include earlier failed protocol stages. No effect claim before adequate paired sample.')
    return aggregate,analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();aggregate,private=audit(args.root)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    (args.output/'private-state-analysis.json').write_text(json.dumps(private,indent=2)+'\n')
    print(json.dumps({k:aggregate[k] for k in ('status','completed_states','completed_continuations','valid_continuations','analyzed_states','final_accounting_verified')},indent=2))


if __name__=='__main__':main()
