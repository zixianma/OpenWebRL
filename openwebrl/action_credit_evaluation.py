"""Own and audit the pinned held-out full300 cohorts, without launching jobs.

The supplied executor must perform real inference, preserve monitor archives,
meter every judge HTTP through the billing Store's auxiliary evaluation API,
and reconcile attempts after a crash. This module never registers training
draws or creates a new campaign/cap. Initial0 is shared once through the
separately approved verification Store; iteration10 belongs to each arm.
Runtime restoration evidence remains a trusted local physical-validator
record, not proof that a text marker alone restores a model. CPU fixtures stay
cpu_mock and cannot be reused by a production owner.
"""
from __future__ import annotations

import asyncio
import fcntl
import inspect
import json
import math
import time
from types import SimpleNamespace
from pathlib import Path
import zipfile

from .action_credit import digest
from .action_credit_campaign import _artifact_file
from .action_credit_native_controller import _directory, _hash, _json, _write
from .action_credit_judge import JudgeHTTPProtocol
from .action_credit_eval_budget import accounting_settings
from .action_credit_state import COST_DIMENSIONS, _safe_path
from .rl_recipe import parse_judge_verdict

PINNED_FULL300_SHA256='d8f81bfc99dfba8b1363d9f50511878dd5db72ff38225c8a196c222aa0c63488'


def native_evaluation_controls():
    """Resolved deterministic-monitor defaults, including inherited history knobs.

    Baseline run_h200_browser.sh sets one screenshot/full reasoning/tool feedback;
    browser_training_config.yaml sets32K/three parse failures/600s/no memory.
    Slime defaults keep all history and512 tool chars; native judge feedback is300.
    These are explicit resolved values, so environment defaults cannot change them.
    """
    return dict(context_num_screenshots=1,judge_max_attached_imgs=3,
        turn_history_reasoning_mode='full',browser_history_reasoning_max_turns=0,
        browser_include_tool_response=1,browser_tool_response_max_chars=512,
        browser_history_env_feedback_max_chars=300,rollout_max_context_len=32768,
        max_consecutive_parse_failures=3,serial_action_variant=None,
        browser_observation_memory=False,browser_memory_max_entries=8,
        browser_memory_max_chars=4000,partial_rollout=False,rollout_task_timeout_secs=600.)


def evaluation_protocol():
    return dict(dataset='Online-Mind2Web monitor',tasks=300,temperature=0.,top_p=1.,top_k=1,
        response_tokens=4096,max_turns=30,judge='gpt-4.1/action_history',browser='local_process',
        browser_response_format_mode='browser_env',repeats=1,wandb_project='openwebrl-evals',optimizer_updates=0,
        native_generation_controls=native_evaluation_controls())


def _require(value,message):
    if not value: raise ValueError(message)


def _ids(reference):
    path=_artifact_file(reference)
    if path.suffix=='.jsonl': rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        value=json.loads(path.read_text())
        rows=value['task_ids'] if isinstance(value,dict) else value
    def identity(row):
        if not isinstance(row,dict):return str(row)
        task=row.get('task_id');native=row.get('metadata',{}).get('task_id')
        _require(task is None or native is None or str(task)==str(native),'Conflicting evaluation task identities')
        _require(task is not None or native is not None,'Evaluation record has no task identity')
        return str(task if task is not None else native)
    ids=tuple(identity(row) for row in rows)
    _require(len(ids)==300 and len(set(ids))==300 and all(ids),'Exactly300 unique pinned evaluation task IDs required')
    return ids


def _checkpoint(reference,policy,iteration,kind):
    value=_json(reference)
    if iteration==0:
        _require(reference['sha256']==policy,'Shared0 must use the pinned original actor inventory')
    else:
        _require(value.get('completed_updates')==10 and value.get('policy_sha256')==policy
            and value.get('verification_kind')==kind,'Evaluation checkpoint is not the committed iteration10 policy')
    _require(value.get('files'),'Checkpoint inventory has no retained files')
    for file in value['files']: _artifact_file(file)


def _sources(reference):
    value=_json(reference); root=_safe_path(value['source_root'])
    required=('openwebrl/eval_monitor.py','openwebrl/reward_browser.py','openwebrl/generate_browser.py',
              'openwebrl/rl_recipe.py','openwebrl/response_format.py','openwebrl/action_credit_evaluation.py')
    running=Path(__file__).resolve().parents[1]
    for name in required:
        _require(name in value['files'],'Evaluation source inventory omits '+name)
        path=_safe_path(root/name)
        _require(root in path.parents and _hash(path)==value['files'][name]
                 and _hash(running/name)==value['files'][name],'Evaluation source changed')


def _archive(reference,task_id):
    import torch
    path=_artifact_file(reference)
    with zipfile.ZipFile(path) as archive:
        _require(archive.testzip() is None,'Rollout ZIP CRC failure')
    # Only exact private native monitor archives from this trusted executor.
    value=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    _require(value['task_id']==task_id and isinstance(value['turns'],list),'Rollout task identity changed')
    return value


def _http_inventory(store,state,attempts,result):
    calls={key:value for key,value in state.get('subrequests',{}).items() if value['parent_attempt_id'] in attempts}
    records=result.get('http_attempts',[])
    rows={row['request_id']:row for row in records}
    _require(len(rows)==len(records) and set(rows)==set(calls),'Retained HTTP inventory omits an evaluation retry')
    if not calls:return
    protocol=JudgeHTTPProtocol(**result['judge_http_protocol'])
    _require(protocol.model=='gpt-4.1' and protocol.sha256==store.manifest.to_dict()['science']['judge']['http_protocol_sha256'],
        'Evaluation changed approved HTTP protocol or billing store')
    for key,call in calls.items():
        row=rows[key];payload=_json(row['request'])
        _require(digest(payload)==call['payload_sha256'] and payload['model']==protocol.model
            and payload['max_completion_tokens']==protocol.max_completion_tokens,'Judge request differs from its charged admission')
        _require(call['reserved']==dict(api_requests=1,api_microdollars=protocol.reserved_microdollars_per_http,
            gpu_seconds=0,browser_dispatches=0),'Evaluation HTTP reserved against another cost protocol')
        if call['status']=='succeeded':
            response=_json(row['response']); usage=response['usage']
            counts=[usage[k] for k in ('prompt_tokens','completion_tokens')]
            _require(all(type(x) is int and x>=0 for x in counts),'Judge usage missing')
            amount=math.ceil((counts[0]*protocol.input_microdollars_per_million_tokens+
                              counts[1]*protocol.output_microdollars_per_million_tokens)/1_000_000)
            cost=dict(api_requests=1,api_microdollars=amount,gpu_seconds=0,browser_dispatches=0)
            _require(digest(response)==call['completion']['response_sha256'] and call['actual']==cost
                and call['charged']==cost,'Retained provider response/usage differs from actual HTTP charge')
        else:
            _require(call['status'] in ('failed','unknown') and call['completion'] is not None
                and call['charged']=={k:max(call['actual'][k],call['reserved'][k]) for k in COST_DIMENSIONS},
                'Unresolved or lost failed HTTP liability')
            _artifact_file(row['error'])


def _actor_inventory(store,state,attempts,result,policy):
    calls={key:value for key,value in state.get('online_actor_requests',{}).items()
           if value.get('kind')=='evaluation' and value['attempt_id'] in attempts}
    rows=result.get('actor_attempts',[]);records={row['request_id']:row for row in rows}
    _require(len(records)==len(rows) and set(records)==set(calls),'Actor inventory omits an evaluation dispatch')
    if state.get('online_actor_requests'):
        cap=accounting_settings(store)['actor_requests_cap']
        _require(len(state['online_actor_requests'])<=cap,'Shared local actor request cap exceeded')
    received=set()
    for key,call in calls.items():
        row=records[key];payload=_json(row['request'])
        _require(digest(payload)==call['payload_sha256'] and call['policy_sha256']==policy
            and call['task_id']==attempts[call['attempt_id']]['task_id'] and type(call['turn']) is int
            and 0<=call['turn']<30,'Actor request differs from admitted policy/task/turn')
        sampling=payload.get('sampling_params',{})
        _require(all(sampling.get(k)==v for k,v in dict(temperature=0.,top_p=1.,top_k=1,max_new_tokens=4096).items())
            and sampling.get('n',1)==1,'Actual actor payload differs from frozen T0 evaluation sampling')
        if call['status']=='received':
            _require(digest(_json(row['response']))==call['completion']['response_sha256'],
                     'Actor response differs from its retained transport digest')
            received.add((call['attempt_id'],call['turn']))
        else:
            _require(call['status'] in ('failed','unknown') and call['completion'] is not None,
                     'Evaluation retains unresolved actor dispatch')
            _artifact_file(row['error'])
    return received


def _native_status(turns):
    """Recompute native valid-only membership and binary success, not judge alone."""
    if not turns:return False,None
    terminal=max(turns,key=lambda turn:turn['metadata'].get('turn_index',0))
    value=terminal.get('reward')
    valid=(not any(turn.get('remove_sample') or str(turn.get('status','')).rsplit('.',1)[-1].lower()=='aborted' for turn in turns)
           and type(value) in (int,float) and math.isfinite(value))
    if not valid:return False,None
    from .reward_browser import compute_format_reward
    reward=terminal['metadata']['reward'];judge=reward.get('judge')
    _require(reward.get('valid') is True and reward.get('judge_timeout') is False
        and type(judge) in (int,float) and math.isfinite(judge) and judge in (0.,1.),
        'Valid native trajectory requires a finite binary judge disposition')
    fmt=compute_format_reward(SimpleNamespace(browser_response_format_mode='browser_env'),terminal['response'],
        turn_level='turn_index' in terminal['metadata'])
    _require(reward.get('format')==fmt,'Archived format score differs from actual response syntax')
    reason=str(terminal['metadata'].get('terminate_reason','')).lower()
    completed=str(terminal['status']).rsplit('.',1)[-1].lower()=='completed'
    if reason=='format_error_failed' or not completed:
        _require(judge==0.,'Native noncompleted/format-failed trajectory cannot receive judge success')
    combined=-1. if reason=='format_error_failed' else (0. if fmt==0. else float(judge==1.))
    _require(reward.get('combined')==combined and value==combined,'Archived terminal reward differs from native gating')
    return True,float(combined==1.)


def _turn_actor_provenance(row,turns,received,request_ref,state,result):
    """Only a native empty ABORTED placeholder may lack actor response evidence."""
    attempt=row['attempt_id']
    if 'no_response' in row:
        proof=_json(row['no_response'])
        _require(len(turns)==1,'No-response evidence must describe one native placeholder')
        turn=turns[0];reason=turn['metadata'].get('terminate_reason','')
        _require(turn.get('response')=='' and turn.get('response_length')==0 and turn.get('tokens')==[]
            and str(turn.get('status','')).rsplit('.',1)[-1].lower()=='aborted'
            and _native_status(turns)==(False,None),'No-response placeholder contains output or a valid outcome')
        _require(not any(a==attempt for a,t in received),'No-response placeholder concealed a received actor response')
        _require(reason=='no_generated_turns' or reason.startswith('generation_error:'),
                 'Unexpected native pre-actor failure reason')
        request=_json(request_ref);sources=_json(request['source_inventory'])
        _require(proof.get('request_sha256')==request_ref['sha256'] and proof.get('task_id')==row['task_id']
            and proof.get('attempt_id')==attempt and proof.get('rollout_sha256')==row['rollout']['sha256']
            and proof.get('reason')==reason
            and proof.get('generation_source_sha256')==sources['files']['openwebrl/generate_browser.py'],
                 'No-response evidence differs from native archive/source identity')
        task_rows=[json.loads(line) for line in _artifact_file(request['task_manifest']).read_text().splitlines() if line.strip()]
        task_row=next(r for r in task_rows if str(r['metadata']['task_id'])==row['task_id'])
        _require(proof.get('sample_sha256')==digest(dict(prompt=task_row['prompt'],metadata=task_row['metadata'])),
                 'Native placeholder was not the exact held-out input Sample')
        return
    from .action_credit_eval_executor import native_actor_output
    records={r['request_id']:r for r in result.get('actor_attempts',[])}
    calls=state.get('online_actor_requests',{})
    for turn in turns:
        index=turn['metadata'].get('turn_index')
        key=turn['metadata'].get('arm',{}).get('evaluation_actor_request_id')
        _require((attempt,index) in received and key in records and key in calls
            and calls[key]['attempt_id']==attempt and calls[key]['turn']==index and calls[key]['status']=='received',
                 'Actual rollout turn lacks its exact admitted received actor response')
        output=native_actor_output(_json(records[key]['response']))
        tokens=turn.get('tokens',[]);length=turn.get('response_length')
        _require(length==len(output[1]) and turn.get('response')==output[0] and tokens[-length:]==output[1],
                 'Archived response differs from actual received actor tokens/text')


def _cohort_state(state,cohort_id):
    """Only settled cohort evidence must stay fixed; unrelated spending may advance."""
    attempts={k:v for k,v in state['evaluation_attempts'].items() if v['cohort_id']==cohort_id}
    return dict(cohort=state['evaluation_cohorts'][cohort_id],attempts=attempts,
        http={k:v for k,v in state.get('subrequests',{}).items() if v['parent_attempt_id'] in attempts},
        actor={k:v for k,v in state.get('online_actor_requests',{}).items()
               if v.get('kind')=='evaluation' and v['attempt_id'] in attempts})


def _missing_evidence(row,attempt,request_ref):
    """A failed dispatch retains failure/cleanup records, never an invented rollout."""
    _require(not any(k in row for k in ('rollout','record','verdict')),'Missing attempt cannot invent a completed rollout')
    failure,cleanup=_json(row['failure']),_json(row['cleanup'])
    for value in (failure,cleanup):
        _require(value.get('request_sha256')==request_ref['sha256'] and value.get('task_id')==row['task_id']
            and value.get('attempt_id')==row['attempt_id'],'Missing attempt evidence identity changed')
    _require(failure.get('reason')==attempt['completion']['reason'] and failure.get('reason'),
             'Missing attempt lost its retained failure reason')
    _require(cleanup.get('status')=='complete' and cleanup.get('processes_alive')==[],
             'Missing browser attempt cleanup remains unresolved')
    for value in (failure,cleanup):
        _require(_artifact_file(value['log']).stat().st_size>0,'Missing attempt evidence log is empty')
    for reference in row.get('partial_artifacts',[]):_artifact_file(reference)


def verify_full300(billing_store,request_ref,result_ref,*,diagnostic_only=False):
    """Read actual300 archives/verdicts and recompute all denominators."""
    request,result=_json(request_ref),_json(result_ref)
    kind='cpu_mock' if diagnostic_only else 'production_runtime'
    _require(diagnostic_only or request['task_manifest']['sha256']==PINNED_FULL300_SHA256,
             'Production evaluation must use the frozen native monitor cohort')
    _require(request['protocol']==evaluation_protocol() and request['protocol_sha256']==digest(evaluation_protocol()),'Evaluation protocol changed')
    _require(request['diagnostic_only'] is diagnostic_only and result.get('verification_kind')==kind,'CPU diagnostic cannot authorize production evaluation')
    _require(request['accounting_manifest_sha256']==billing_store.manifest.sha256,'Evaluation uses a different billing campaign')
    ids=_ids(request['task_manifest'])
    _require(list(ids)==request['task_ids'],'Pinned evaluation task order changed')
    for key in ('protocol','policy_sha256','checkpoint_inventory','source_inventory'):
        _require(result.get(key)==request[key],'Evaluation result differs from request: '+key)
    _require(result.get('request_sha256')==request_ref['sha256'],'Result belongs to another immutable cohort request')
    _sources(request['source_inventory'])
    _checkpoint(request['checkpoint_inventory'],request['policy_sha256'],request['iteration'],kind)
    runtime=_json(result['runtime'])
    _require(runtime.get('verification_kind')==kind and runtime.get('request_sha256')==request_ref['sha256']
        and runtime.get('checkpoint_inventory')==request['checkpoint_inventory']
        and runtime.get('policy_sha256')==request['policy_sha256'] and runtime.get('protocol')==request['protocol']
        and runtime.get('worker_returncode')==0 and runtime.get('launcher_returncode')==0
        and runtime.get('optimizer_updates')==0,'Evaluation runtime/restore identity or exit status differs')
    proof=_json(runtime['physical_restore_review'])
    _require(proof.get('verification_kind')==kind and proof.get('policy_sha256')==request['policy_sha256']
        and proof.get('checkpoint_inventory')==request['checkpoint_inventory'] and proof.get('restored_on_gpu') is True
        and proof.get('source_inventory')==request['source_inventory'],'Actual checkpoint restoration review missing')
    log=_artifact_file(proof['worker_log']).read_text(errors='replace')
    _require(isinstance(proof.get('restore_line'),str) and proof['restore_line'] and proof['restore_line'] in log,
             'Physical restore review lacks its retained worker evidence')
    rows=result['tasks']; _require(len(rows)==300 and {x['task_id'] for x in rows}==set(ids),'Evaluation must retain each exact task once')
    state=billing_store.snapshot(); _require(not state['halt'],'Billing campaign halted')
    cohort=state['evaluation_cohorts'][request['cohort_id']]
    _require(cohort['request_ref']==request_ref and cohort['request']==request,'Billing cohort request changed')
    cohort_attempts={k:v for k,v in state['evaluation_attempts'].items() if v['cohort_id']==request['cohort_id']}
    received=_actor_inventory(billing_store,state,cohort_attempts,result,request['policy_sha256'])
    selected_ids={row['attempt_id'] for row in rows}
    prior_rows=result.get('prior_attempts',[])
    prior_ids={row['attempt_id'] for row in prior_rows}
    _require(len(prior_ids)==len(prior_rows) and prior_ids==set(cohort_attempts)-selected_ids,
             'Prior browser attempt inventory is incomplete or duplicated')
    artifact_rows=list(rows)
    for row in prior_rows:
        attempt=cohort_attempts[row['attempt_id']]
        _require(row['task_id']==attempt['task_id'] and attempt['status'] in ('invalid','missing'),
                 'Prior attempt identity/status violates no-valid-retry policy')
        if attempt['status']=='missing':_missing_evidence(row,attempt,request_ref)
        else:artifact_rows.append(row)
    successes=valid=0; outcomes={}; selected_attempts=set()
    for row in artifact_rows:
        selected=row['attempt_id'] in selected_ids
        task=row['task_id']; record,verdict=_json(row['record']),_json(row['verdict'])
        rollout=_archive(row['rollout'],task); turns=rollout['turns']
        _require(record['task_id']==task and record['rollout_file']==Path(row['rollout']['path']).name
            and record['turns']==len(turns) and record['error_type']==rollout['error_type']
            and record['judge_model']=='gpt-4.1' and record['judge_prompt_variant']=='action_history',
            'Task archive/monitor record/protocol mismatch')
        _require(len(turns)<=30 and all(t.get('response_length',len(t.get('tokens',())))<=4098 for t in turns),
            'Rollout exceeds frozen horizon/response allowance')
        if not diagnostic_only:_turn_actor_provenance(row,turns,received,request_ref,state,result)
        terminal=max(turns,key=lambda t:t['metadata'].get('turn_index',0)) if turns else None
        reward=terminal['metadata'].get('reward',{}) if terminal else {}
        _require(record['reward_metadata']==reward,'Record reward differs from actual archived terminal')
        _require(verdict.get('task_id')==task and verdict.get('request_sha256')==request_ref['sha256']
            and verdict.get('rollout_sha256')==row['rollout']['sha256'] and verdict.get('judge_model')=='gpt-4.1'
            and verdict.get('judge_prompt_variant')=='action_history','Verdict identity differs from retained rollout')
        calls={key:value for key,value in state.get('subrequests',{}).items() if value['parent_attempt_id']==row['attempt_id']}
        http_rows={value['request_id']:value for value in result.get('http_attempts',[])}
        request_ids=verdict.get('http_request_ids',[])
        _require(len(set(request_ids))==len(request_ids) and set(request_ids)==set(calls),
                 'Verdict omits a charged judge HTTP attempt')
        actual_responses=[http_rows[key]['response'] for key in request_ids if calls[key]['status']=='succeeded']
        _require(verdict.get('http_responses',[])==actual_responses,'Verdict response differs from charged provider evidence')
        status=verdict['status']; score=verdict['reward']
        native_valid,native_score=_native_status(turns)
        _require((status=='valid')==native_valid,'Verdict validity differs from native valid-only membership')
        expected=dict(trajectories=1,valid_trajectories=int(native_valid),invalid_trajectories=int(not native_valid),
                      successes=int(native_score==1.))
        _require(all(record['metrics'].get(k)==v for k,v in expected.items()),'Native per-task metrics differ from archived trajectory')
        if status=='valid':
            _require(terminal is not None and reward.get('valid') is True and reward.get('judge_timeout') is False
                and not any(t.get('remove_sample') or str(t.get('status','')).lower()=='aborted' for t in turns)
                and type(score) in (int,float) and score in (0.,1.) and score==native_score,'Valid outcome differs from native terminal evidence')
            text=reward['judge_text']
            if text.startswith('Judge not run for '):
                reason=str(terminal['metadata'].get('terminate_reason','')).lower()
                native_status=str(terminal['status']).rsplit('.',1)[-1].upper()
                expected_text=('Judge not run for terminate_reason=format_error_failed' if reason=='format_error_failed'
                    else 'Judge not run for status=Status.'+native_status)
                _require(text==expected_text and score==0. and (reason=='format_error_failed'
                    or native_status!='COMPLETED'),'Unjudged successful trajectory or noncanonical native disposition')
            else:
                responses=[_json(ref) for ref in verdict['http_responses']]
                parsed=parse_judge_verdict(text)
                _require(any(r.get('choices') and r['choices'][0]['message']['content']==text for r in responses)
                    and parsed in (0.,1.) and parsed==reward['judge'],'Verdict text not present in actual provider responses')
            if selected:
                valid+=1; successes+=score==1.
        else:
            _require(status=='invalid' and score is None and verdict.get('invalid_reason'),'Invalid task needs explicit reason and no imputed reward')
        attempt=state['evaluation_attempts'][row['attempt_id']]
        _require(attempt['cohort_id']==request['cohort_id'] and attempt['task_id']==task
            and attempt['status']==status and attempt['completion']['outcome']['artifact_sha256']==row['verdict']['sha256']
            and attempt['completion']['outcome'].get('reward')==score,'Saved verdict differs from settled evaluation attempt')
        if selected:
            _require(cohort['tasks'][task]['selected_attempt']==row['attempt_id'],'Result substituted an unselected evaluation attempt')
            _require(row['attempt_id'] not in selected_attempts,'One evaluation attempt reused for multiple tasks')
            selected_attempts.add(row['attempt_id']); outcomes[task]=dict(status=status,reward=score)
        del rollout,turns
    totals=billing_store._totals(state); caps=billing_store.manifest.to_dict()['budget_caps']
    _require(all(totals[k]<=caps[k] for k in COST_DIMENSIONS),'Combined training/evaluation cap exceeded')
    if accounting_settings(billing_store)['verification']:
        billing_store.validate_accounting_snapshot(state)
    _require(all(v['status'] in ('valid','invalid','missing') for v in cohort_attempts.values()),'Evaluation retains unresolved attempts')
    for attempt in cohort_attempts.values():
        expected=dict(gpu_seconds=0,api_requests=0,api_microdollars=0,browser_dispatches=1)
        _require(attempt['actual']==expected and attempt['charged']=={k:max(expected[k],attempt['reserved'][k]) for k in COST_DIMENSIONS},
                 'Evaluation attempt duplicated API/GPU cost or lost its browser dispatch')
    _http_inventory(billing_store,state,cohort_attempts,result)
    _require(all(v['status']!='reserved' for v in state.get('subrequests',{}).values()
        if v['parent_attempt_id'] in cohort_attempts),'Evaluation retains unsettled HTTP liabilities')
    metrics=dict(tasks=300,successes=successes,valid=valid,invalid=300-valid,
        overall_rate=successes/300,valid_only_rate=successes/valid if valid else None)
    _require(result['metrics']==metrics,'Aggregate does not match actual300 verdicts')
    wandb=_json(result['wandb'])
    _require(wandb.get('project')=='openwebrl-evals' and wandb.get('state')=='finished'
        and wandb.get('request_sha256')==request_ref['sha256'] and wandb.get('metrics')==metrics
        and isinstance(wandb.get('run_id'),str) and wandb['run_id'],'Separate finished W&B evidence differs')
    fresh=billing_store.snapshot()
    _require(_cohort_state(fresh,request['cohort_id'])==_cohort_state(state,request['cohort_id']),
             'Audited evaluation cohort changed during audit')
    _require(not fresh['halt'],'Billing campaign halted during audit')
    totals=billing_store._totals(fresh)
    _require(all(totals[k]<=caps[k] for k in COST_DIMENSIONS),'Combined training/evaluation cap exceeded')
    if accounting_settings(billing_store)['verification']:
        billing_store.validate_accounting_snapshot(fresh)
    if fresh.get('online_actor_requests'):
        _require(len(fresh['online_actor_requests'])<=accounting_settings(billing_store)['actor_requests_cap'],
                 'Shared local actor request cap exceeded')
    return dict(schema_version=1,verification_kind=kind,verified_complete=not diagnostic_only,
        diagnostic_only=diagnostic_only,request=request_ref,result=result_ref,cohort_id=request['cohort_id'],
        checkpoint_inventory=request['checkpoint_inventory'],policy_sha256=request['policy_sha256'],iteration=request['iteration'],
        metrics=metrics,task_outcomes=outcomes,charged_resources=totals,ledger_revision=fresh['revision'],
        verifier_source_sha256=_hash(__file__))


class Full300EvaluationOwner:
    """Bootstrap owner. ``executor.run_or_resume(store,request_ref,path,context)``
    must await all physical workers and return a saved result artifact reference.
    No default executor, new API authority, or implicit redispatch is provided.
    """
    def __init__(self,*,task_manifest_ref,original_actor_inventory_ref,source_inventory_ref,
                 shared_initial_root,arm_output_root,verification_store,executor,diagnostic_only=False):
        _require(callable(getattr(executor,'run_or_resume',None)),'Real owned evaluation executor required')
        _require(type(diagnostic_only) is bool and (diagnostic_only or task_manifest_ref['sha256']==PINNED_FULL300_SHA256),
                 'Production evaluation must use the frozen native monitor cohort')
        if not diagnostic_only:
            from .action_credit_verification import VerificationStore, approved_authority
            _require(type(verification_store) is VerificationStore,
                     'Shared initial full300 requires its separate purpose-scoped verification store')
            approved_authority(verification_store.manifest)
        self.tasks,self.original,self.source=task_manifest_ref,original_actor_inventory_ref,source_inventory_ref
        self.ids=_ids(self.tasks); self.shared=_safe_path(shared_initial_root); self.output=_safe_path(arm_output_root)
        self.verification_store,self.executor,self.diagnostic=verification_store,executor,diagnostic_only

    async def _cohort(self,store,iteration,checkpoint,policy,context):
        source=self.source
        if iteration==0:
            # Identical per-arm inventory files have different paths. Shared0
            # has one billing authority and one canonical request, so its ID
            # must not depend on which authorized arm acquires the lock first.
            source=store.manifest.to_dict().get('source_inventory',self.source)
            _require(source['sha256']==self.source['sha256'] and _json(source)==_json(self.source),
                     'Shared initial source inventory differs from the arm source')
        request=dict(schema_version=1,task_ids=list(self.ids),task_manifest=self.tasks,protocol=evaluation_protocol(),
            protocol_sha256=digest(evaluation_protocol()),checkpoint_inventory=checkpoint,policy_sha256=policy,
            iteration=iteration,source_inventory=source,accounting_manifest_sha256=store.manifest.sha256,
            artifact_schema='native_monitor_full300_v1',diagnostic_only=self.diagnostic)
        request['cohort_id']=digest(request)
        root=_directory((self.shared if iteration==0 else self.output)/request['cohort_id'])
        with (root/'owner.lock').open('a+b') as lock:
            deadline=getattr(context.get('launch'),'deadline_unix',None)
            _require(self.diagnostic or deadline is not None,'Evaluation owner requires the allocation deadline')
            while True:
                if deadline is not None and time.time()>=deadline:raise TimeoutError('Shared evaluation owner exceeded allocation deadline')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if deadline is not None and time.time()>=deadline:raise TimeoutError('Shared evaluation owner exceeded allocation deadline')
                    await asyncio.sleep(.25)
            request_path=root/'request.json'
            ref=dict(path=str(request_path),sha256=_hash(request_path)) if request_path.exists() else _write(request_path,request)
            _require(_json(ref)==request,'Existing cohort request is immutable')
            store.register_evaluation_cohort(request['cohort_id'],ref)
            saved=root/'result-reference.json'
            if saved.exists(): result_ref=_json(dict(path=str(saved),sha256=_hash(saved)))
            else:
                if deadline is not None and time.time()>=deadline:raise TimeoutError('Evaluation dispatch exceeded allocation deadline')
                value=self.executor.run_or_resume(store,ref,root,context)
                result_ref=await value if inspect.isawaitable(value) else value
                _artifact_file(result_ref)
                candidate=root/('candidate-result-'+digest(result_ref)+'.json')
                if not candidate.exists(): _write(candidate,result_ref)
            review=await asyncio.to_thread(verify_full300,store,ref,result_ref,diagnostic_only=self.diagnostic)
            # Only a verified complete cohort becomes the idempotent pointer.
            # Failed candidates remain; the executor reconciles them on retry.
            if not saved.exists(): _write(saved,result_ref)
            target=root/('review-'+digest(review)+'.json')
            return dict(path=str(target),sha256=_hash(target)) if target.exists() else _write(target,review)

    async def initial(self,store,contract,context):
        _require(contract.update_id==0 and contract.policy_sha256==self.original['sha256'],'Initial eval changed original actor')
        _require(not set(self.ids)&set(contract.train_task_ids),'Held-out eval IDs overlap training tasks')
        return await self._cohort(self.verification_store,0,self.original,contract.policy_sha256,context)

    async def after_update(self,store,count,result,context):
        if count!=10: return None
        state=store.snapshot(); update=state['updates']['9']; native=state['native_completed_updates']['9']
        _require(update['phase']=='consumed' and result.get('committed') is True
            and native['checkpoint_inventory']['sha256']==update['training_receipt']['new_checkpoint_sha256'],
            'Iteration10 evaluation requires exact committed native checkpoint')
        return await self._cohort(store,10,native['checkpoint_inventory'],update['training_receipt']['new_policy_sha256'],context)

    async def final(self,store,training,context):
        _require(training.get('training_complete') is True and training.get('completed_updates')==10,'Training endpoint incomplete')
        return await self.after_update(store,10,dict(committed=True),context)
