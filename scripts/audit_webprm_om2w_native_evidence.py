#!/usr/bin/env python3
"""Read-only action/terminal/image consistency review of saved OM2W outcomes.

This does not relabel HTTP/bot-check pages or assert judge semantic accuracy.
Native validity follows the frozen v3 contract; interruptions remain separate.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

from scripts.audit_webprm_om2w_saved_calls import read,sha,ref,require,signature,REPO,expected_dispositions


def audit_control_selection(plan, rollout):
    """Replay actual baseline control indices, including terminal overrides and dropped slots."""
    import random
    for turn,row in enumerate(rollout):
        proposal=row['proposal'];pool=proposal['candidates'];chosen=row['selected_index']
        if not pool:
            require(chosen is None,'Empty control pool must execute no action')
            continue
        if plan['selector'] not in ('first5','random5'):
            continue
        if proposal['terminal']:
            expected=0
        else:
            require(proposal['requested_raw_slots']==5,'Control used a different candidate-generation budget')
            expected=0 if plan['selector']=='first5' else random.Random(plan['seed']+turn).randrange(len(pool))
        require(chosen==expected and row['selection_trace']==[], 'Control selection differs from its frozen policy')


def cleanup_result(folder,result,expected_identity):
    path=folder/'native-evaluation/result.json'
    if path.exists():return read(path),'normal'
    # The frozen watchdog preserves a settled verdict if browser cleanup hangs.
    # The absent post-close file is never manufactured or called graceful closure.
    watchdog=read(folder/'watchdog.json')
    require(result['cleanup_status']=='interrupted' and result['browser_close_error']=='cleanup_timeout'
            and result['watchdog']==watchdog and watchdog['reason']=='cleanup_timeout'
            and watchdog['last_phase']['phase']=='cleanup'
            and watchdog['last_phase']['identity_sha256']==result['identity_sha256']==expected_identity
            and watchdog['remaining_owned_processes']=={} and watchdog['child_exitcode'] is not None,
            'Absent post-close native result lacks exact settled-verdict cleanup watchdog')
    return result,'watchdog_reaped_after_settled_verdict'


def fallback_evidence(folder,result,plan,trajectory,rollout,harness,native_execution_code,digest,revision):
    fallback=result.get('protocol_fallback');path=folder/'protocol-fallback.json'
    terminal=trajectory[-1]
    if fallback is None:
        require(not path.exists() and not terminal.get('origin') and not terminal.get('protocol'),
                'Undeclared fallback evidence')
        return 'actor'
    protocol='released_viewer_na_fallback_v1';origin='released_viewer_normal_turn_limit'
    expected=dict(schema_version=1,protocol=protocol,origin=origin,termination_reason='turn_limit',
        action='send_msg_to_user [N/A]',answer='N/A',actor_selected=False,actor_turns=30,
        source_revision=revision,source_file='viewer/src/server/service/executors/step_loop_executor.py',status='executed')
    require(path.exists() and read(path)==fallback and all(fallback.get(k)==v for k,v in expected.items())
            and fallback['actor_selected'] is False and result['turn_limit_protocol']==protocol
            and plan['max_turns']==len(rollout)==result['turns']==30,
            'Fallback does not match frozen full-turn protocol')
    require(terminal==dict(action_type='send_msg_to_user',answer='N/A',raw_action=expected['action'],origin=origin,protocol=protocol)
            and fallback['pre_fallback_trajectory_sha256']==digest(trajectory[:-1])
            and 'observation' in trajectory[-2],'Fallback replaced earlier evidence')
    for row in rollout:
        chosen=row['selected_index']
        if chosen is not None:
            require(harness.parse_action(row['proposal']['candidates'][chosen]['action'])['kind']!='send_message_to_user',
                    'Fallback overwrote an actor terminal')
    before=fallback['chat_messages_before'];after=fallback['chat_messages_after'];appended=fallback['appended_messages']
    require(isinstance(before,list) and isinstance(after,list) and after[:len(before)]==before
            and after[len(before):]==appended and len(appended)==1 and appended[0]['role']=='assistant'
            and appended[0]['message']=='N/A','Fallback lacks its literal executed chat')
    dispatch=fallback['native_dispatch'];code=native_execution_code(plan['arbiter_source'],harness.parse_action(expected['action']),
        dispatch['element_bounds'],key_mapping=harness.key_mapping)
    import hashlib
    require(dispatch['action']==expected['action'] and dispatch['code']==code
            and dispatch['code_sha256']==hashlib.sha256(code.encode()).hexdigest(),'Fallback native dispatch differs')
    return 'protocol_turn_limit'


def audit(package,timeout=600):
    started=time.monotonic();package=Path(package).resolve();proposal=read(package/'proposal.json')
    source=package/'source';sys.path.insert(0,str(source))
    from openwebrl.webprm_webarena import digest
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness,native_execution_code
    from openwebrl.webprm_source import ARBITER_REVISION
    from PIL import Image
    spec=importlib.util.spec_from_file_location('om2w_native_collector',REPO/'scripts/collect_webprm_om2w_table5.py')
    collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    # Reuse the immutable action replay implementation that shipped with workers.
    spec=importlib.util.spec_from_file_location('om2w_frozen_action_auditor',source/'scripts/audit_webprm_table5_cell.py')
    action_auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(action_auditor)
    require(sha(source/'scripts/audit_webprm_table5_cell.py')==proposal['source_sha256'][str(source/'scripts/audit_webprm_table5_cell.py')],
            'Action auditor is outside frozen source manifest')
    rows=collector.stage_records(package);campaign=Path(proposal['campaign_root'])
    require(len(rows)==expected_dispositions(proposal) and all(r is not None for _,r,_ in rows),'Fixed study dispositions unresolved')
    counters=Counter();records=[];fingerprints={}
    for cell,result,state in rows:
        require(time.monotonic()-started<timeout,'Offline native review deadline exhausted')
        if not result['valid']:
            counters['invalid']+=1
            records.append(dict(cell_id=cell['cell_id'],raw_valid=result.get('raw_native_valid',False),
                effective_valid=False,reason=result.get('error_type'),state=state));continue
        import hashlib
        key=hashlib.sha256(cell['cell_id'].encode()).hexdigest()
        assigned=[p for p in campaign.glob('attempt-*/cells/'+key)
                  if (p/'episodes'/('om2w_'+cell['task_id'])/(cell['actor']+'__'+cell['selector'])/'result.json').exists()]
        require(len(assigned)==1,'Native cell has no unique executed result: '+cell['cell_id'])
        assigned=assigned[0];plan=read(assigned/'worker-plan.json')
        folder=assigned/'episodes'/('om2w_'+cell['task_id'])/(cell['actor']+'__'+cell['selector']);native=folder/'native-evaluation'
        paths=[folder/'result.json',folder/'rollout.json',folder/'provenance.json',folder/'om2w-judge.json',folder/'judge-final.jpg']
        paths += [native/name for name in ('trajectory.json','terminal.json','terminal.png','native-verdict.json','outcome-before-close.json')]
        paths.append(folder/'watchdog.json')
        if (native/'result.json').exists():paths.append(native/'result.json')
        for p in paths:fingerprints[str(p)]=signature(p)
        trajectory=read(native/'trajectory.json');rollout=read(folder/'rollout.json');terminal=read(native/'terminal.json')
        verdict=read(native/'native-verdict.json');before=read(native/'outcome-before-close.json');closed,closure=cleanup_result(folder,result,digest(plan))
        judge=read(folder/'om2w-judge.json');provenance=read(folder/'provenance.json')
        require(trajectory and 'observation' in trajectory[0] and trajectory[0]['info']['reset_id']==result['reset_id'],
                'Native trajectory lacks its original live reset identity')
        action_auditor.audit_rollout(plan,trajectory,rollout,result)
        if 'candidate_controls' in proposal:
            audit_control_selection(plan,rollout)
        termination=fallback_evidence(folder,result,plan,trajectory,rollout,load_arbiter_harness(plan['arbiter_source']),
            native_execution_code,digest,ARBITER_REVISION)
        require(result['terminal']==terminal and terminal['screenshot_sha256']==sha(native/'terminal.png')
                and terminal['termination']==judge['termination'] and terminal['url']==judge['url'],
                'Native terminal URL/image/termination differs from saved judge')
        require(before['valid'] is True and closed['valid'] is True and before['success']==closed['success']==verdict['success']==bool(result['success'])
                and before['score']==closed['score']==verdict['score']==result['success']
                and before['terminal']==closed['terminal']==terminal,'Before/after cleanup verdict changed')
        require(provenance['identity_sha256']==result['identity_sha256']==digest(plan)
                and provenance['actor_model']==cell['actor'] and provenance['paper_exact_protocol'] is False,
                'Native provenance differs from actual assigned plan')
        dimensions=[]
        for p,format_name in [(native/'terminal.png','PNG'),(folder/'judge-final.jpg','JPEG')]:
            with Image.open(io.BytesIO(p.read_bytes())) as image:
                require(image.format==format_name and image.width>0 and image.height>0,'Undecodable terminal image')
                dimensions.append(image.size);image.verify()
        require(dimensions[0]==dimensions[1], 'Judge and native snapshot viewports differ')
        counters.update(valid=1,successes=result['success']);counters[termination]+=1;counters[closure]+=1
        records.append(dict(cell_id=cell['cell_id'],raw_valid=True,effective_valid=True,success=result['success'],
            termination=termination,cleanup_status=result['cleanup_status'],artifacts=[ref(p) for p in paths]))
    for p,before in fingerprints.items():require(signature(p)==before,'Native evidence changed during audit')
    return dict(schema_version=1,audit_verified=True,scope='Saved action, native terminal and image consistency under frozen v3 validity; no semantic judge accuracy or historical paper parity asserted',
        created_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        auditor=ref(Path(__file__).resolve()),action_auditor=ref(source/'scripts/audit_webprm_table5_cell.py'),
        collector=ref(REPO/'scripts/collect_webprm_om2w_table5.py'),counts=dict(counters),records=records,
        raw_effective_policy='No native label changes; interrupted raw verdicts remain absent. HTTP>=400 proceeds under frozen v3, with no retrospective filtering.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();target=args.output
    require(target.is_absolute() and target.resolve()==target and Path('/gpfs/scrubbed/zixianma/openwebrl-runtime') in target.parents,'Private approved output required')
    result=audit(args.package)
    with target.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
