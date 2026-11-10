import json
from pathlib import Path
import sys
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import watch_luna_qwen_full300 as watcher


def test_continuation_preserves_exact_caps_and_both_pools():
    message = watcher.continuation_message()
    assert len(message.encode()) < 1000
    assert 'x16h' in message and 'x4h' in message and '39600' in message and '6600' in message
    assert 'all1500' in message and 'adds no budget' in message


def test_stall_distinguishes_long_live_episode_from_missing_progress(tmp_path):
    plan = dict(output=str(tmp_path / 'sft/results'), wandb_prefix='study-sft')
    worker = dict(worker_id='gpu-1-0-0', started_epoch=100)
    directory = Path(plan['output']) / 'workers' / worker['worker_id']
    watcher.write(directory / 'heartbeat.json', dict(task_key='task', updated_unix=100))
    row = watcher.worker_snapshot(tmp_path, 'sft', plan, worker, '1', 1101)
    assert row['stalled'] and row['stall_kind'] == 'progress'
    request = Path(plan['output']) / 'attempts/task/attempt/requests/a.json'
    watcher.write(request, dict(status='received'))
    row = watcher.worker_snapshot(tmp_path, 'sft', plan, worker, '1', request.stat().st_mtime + 5)
    assert not row['stalled'] and row['last_request_age_seconds'] == 5
    row = watcher.worker_snapshot(tmp_path, 'sft', plan, dict(worker, returncode=1), '1', 1e12)
    assert not row['stalled']


def test_dispatch_signature_is_stable_for_progress_but_detects_failures():
    latest = dict(pools={'gpu': dict(job_id='1', state='RUNNING', controller={'stage':'running'},
                                    pointer_stale=False)}, api_halts={}, stalled_workers=[],
                  approval_mismatch=False, record_errors=[], completed=10)
    before = watcher.dispatch_state(latest)
    latest['completed'] = 11
    assert watcher.dispatch_state(latest) == before
    latest['pools']['gpu']['controller']['failed_families'] = {'sft': {'returncode': 1}}
    assert watcher.dispatch_state(latest) != before


def test_unified_tracker_keeps_other_sessions_and_reused_baseline(tmp_path):
    untouched = [dict(id='AS04', mode='sft_reference', status='verified_complete_reuse'),
                 dict(id='AS08', mode='jev', status='running', job_id='other')]
    watcher.write(tmp_path / 'experiment-tracker.json', dict(rows=[
        dict(id='AS05', mode='sft_luna5', protocol_group='luna_full300'), *untouched],
        excluded=[dict(mode='qwen_luna10')]))
    latest = dict(epoch=100, approval_mismatch=False,
        modes={'sft_luna5': dict(episodes=7, successes=2, valid=6)},
        pools={'gpu': dict(job_id='1', state='RUNNING', controller={'stage':'running'})})
    watcher.update_tracker(latest, tmp_path)
    saved = watcher.read(tmp_path / 'experiment-tracker.json')
    assert saved['rows'][0]['recorded_episodes'] == 7 and saved['rows'][0]['job_id'] == '1'
    assert saved['rows'][1:] == untouched and saved['excluded'] == [dict(mode='qwen_luna10')]


def test_snapshot_prefers_latest_approved_attempt_and_keeps_pool_isolation(tmp_path, monkeypatch):
    watcher.write(tmp_path / 'plan.json', {'families': {}})
    watcher.write(tmp_path / 'approval.json', dict(approved=True, limits=watcher.LIMITS,
        pools=dict(gpu=dict(total_seconds=57600), cpu=dict(total_seconds=14400)),
        attempts=[dict(pool='gpu',job_id='1'),dict(pool='cpu',job_id='2'),dict(pool='gpu',job_id='3')]))
    watcher.write(tmp_path / 'pools/gpu/current.json', dict(job_id='1', stage='failed'))
    monkeypatch.setattr(watcher, 'scheduler', lambda job: dict(job_id=job,state='PENDING',raw=''))
    latest = watcher.snapshot(tmp_path)
    assert latest['pools']['gpu']['job_id'] == '3' and latest['pools']['gpu']['pointer_stale']
    assert latest['pools']['cpu']['job_id'] == '2'
    assert not latest['approval_mismatch'] and latest['expected_episodes'] == 1500


def test_peer_counts_are_fresh_and_tracker_preserves_separate_owner_and_protocol(tmp_path, monkeypatch):
    directory = tmp_path / 'peer'
    watcher.write(tmp_path / 'pointer.json', dict(run_directory=str(directory), job_id='7',
        verified_complete=False, agent_thread='peer-owner', continuation_active=True))
    watcher.write(directory / 'jev/results/a.json', dict(valid=True, reward={'combined': 1}))
    watcher.write(directory / 'heartbeat.json', dict(updated_unix=90, stage='jev'))
    watcher.write(directory / 'supervisor-latest.json', dict(checked_unix=95))
    watcher.write(directory / 'approval.json', dict(approved=True, resources=dict(total_seconds=36000),
        attempts=[dict(job_id='7')]))
    monkeypatch.setattr(watcher.subprocess, 'check_output', lambda *a, **k: '7|RUNNING|10\n')
    peer = watcher.peer_snapshot(tmp_path, 'jev', 'pointer.json', 100)
    assert (peer['episodes'], peer['successes'], peer['valid']) == (1, 1, 1)
    assert peer['heartbeat_age_seconds'] == 10 and peer['owning_agent_thread'] == 'peer-owner'
    watcher.write(tmp_path / 'experiment-tracker.json', dict(rows=[
        dict(id='AS08', mode='jev', protocol_group='sft_jev_kev_full300')]))
    watcher.update_tracker(dict(epoch=100, peers={'jev':peer}, approval_mismatch=False), tmp_path)
    row = watcher.read(tmp_path / 'experiment-tracker.json')['rows'][0]
    assert row['recorded_episodes'] == 1 and row['owning_agent_thread'] == 'peer-owner'
    assert row['protocol_group'] == 'sft_jev_kev_full300' and row['status'] == 'running'


def test_peer_charges_every_unique_attempt_and_keeps_owner_report_separate(tmp_path, monkeypatch):
    directory = tmp_path / 'peer'
    resources = dict(gpus=1, cpus=8, memory_gib=120, total_seconds=36000)
    watcher.write(tmp_path / 'pointer.json', dict(run_directory=str(directory), job_id='344794',
        remaining_approved_seconds=29373, continuation_active=True, approved_resources=resources))
    watcher.write(directory / 'approval.json', dict(approved=True, resources=resources, attempts=[
        dict(job_id=job) for job in ['344662', '344708', '344794', '344708']]))
    # A fresh service snapshot is not an owner review.
    watcher.write(directory / 'supervisor-latest.json', dict(checked_unix=999))
    watcher.write(directory / 'supervisor-ack.json', dict(epoch=800))
    watcher.write(directory / 'agent-review-latest.json', dict(reviewed_unix=750, checked_unix=950))
    calls = []

    def sacct(command, **kwargs):
        calls.append(command)
        return ('344662|FAILED|1003\n344708|FAILED|3813\n344794|FAILED|13189\n'
                '344794.batch|FAILED|13189\n344794.0|FAILED|13189\n')

    monkeypatch.setattr(watcher.subprocess, 'check_output', sacct)
    peer = watcher.peer_snapshot(tmp_path, 'jev', 'pointer.json', 1000)
    assert len(calls) == 1 and calls[0][calls[0].index('-j') + 1] == '344662,344708,344794'
    assert calls[0][-1] == '--format=JobIDRaw,State,ElapsedRaw'
    assert peer['consumed_scheduler_seconds'] == 18005 and peer['remaining_approved_seconds'] == 17995
    assert peer['owner_reported_remaining_seconds'] == 29373 and peer['accounting_error'] is None
    assert len(peer['scheduler_attempts']) == 3 and peer['state'] == 'FAILED'
    assert peer['approved_resources'] == resources
    assert peer['continuation_active'] is True and peer['continuation_active_source'] == 'owner_reported'
    assert peer['owner_last_review_epoch'] == 800 and peer['owner_last_review_age_seconds'] == 200
    assert peer['owner_last_review_source'] == 'supervisor-ack'
    assert peer['owner_review_document_epoch'] == 750
    watcher.write(directory / 'agent-review-latest.json', dict(reviewed_unix=900))
    assert watcher.peer_owner_review(directory, 1000)['owner_last_review_epoch'] == 900


@pytest.mark.parametrize('problem', ['missing_approval', 'missing_attempt', 'invalid_elapsed',
    'negative_elapsed', 'unknown_state', 'duplicate_row', 'query_failure'])
def test_peer_accounting_failure_is_unknown_never_stale_or_zero(tmp_path, monkeypatch, problem):
    directory = tmp_path / 'peer'
    watcher.write(tmp_path / 'pointer.json', dict(run_directory=str(directory), job_id='8',
        remaining_approved_seconds=29373, continuation_active=True))
    if problem != 'missing_approval':
        watcher.write(directory / 'approval.json', dict(approved=True, resources=dict(total_seconds=36000),
            attempts=[dict(job_id='7', scheduler_seconds=1003), dict(job_id='8')]))
    replies = dict(missing_attempt='8|FAILED|100\n', invalid_elapsed='7|FAILED|\n8|FAILED|100\n',
        negative_elapsed='7|FAILED|-1\n8|FAILED|100\n', unknown_state='7|UNKNOWN|1\n8|FAILED|100\n',
        duplicate_row='7|FAILED|1003\n7|FAILED|1003\n8|FAILED|100\n')

    def sacct(command, **kwargs):
        if problem == 'query_failure':
            raise subprocess.CalledProcessError(1, command)
        return replies.get(problem, '')

    monkeypatch.setattr(watcher.subprocess, 'check_output', sacct)
    peer = watcher.peer_snapshot(tmp_path, 'jev', 'pointer.json', 1000)
    assert peer['remaining_approved_seconds'] is None and peer['consumed_scheduler_seconds'] is None
    assert peer['accounting_error'] and peer['owner_reported_remaining_seconds'] == 29373
    assert peer['owner_last_review_epoch'] is None and peer['owner_last_review_age_seconds'] is None
    latest = dict(pools={}, api_halts={}, stalled_workers=[], approval_mismatch=False,
        record_errors=[], completed=0, peers={'jev': dict(peer, accounting_error=None)})
    before = watcher.dispatch_state(latest)
    latest['peers']['jev'] = peer
    assert watcher.dispatch_state(latest) != before


def test_completed_cpu_tracker_requires_matching_independent_audit(tmp_path):
    watcher.write(tmp_path / 'experiment-tracker.json', dict(rows=[
        dict(id='AS07', mode='luna', protocol_group='luna_full300')]))
    latest = dict(epoch=100, approval_mismatch=False,
        modes={'luna': dict(episodes=300, successes=110, valid=279)},
        pools={'cpu': dict(job_id='2', state='COMPLETED', controller={'stage':'collection_complete'})})
    watcher.update_tracker(latest, tmp_path)
    row = watcher.read(tmp_path / 'experiment-tracker.json')['rows'][0]
    assert not row['verified_complete'] and row['status'] == 'collected_pending_final_verification'
    proof = dict(complete=True, cpu_collection_verified=True, cpu_terminal_accounting=True,
        final_repair_worker_exits_successful=True, active_workers=0, integrity_errors=[],
        completion_blockers=[], wandb_reconciliation={'remote_API_called':True},
        scheduler={'2':{'state':'COMPLETED'}}, expected_episodes=300, committed_episodes=300,
        valid=279, successes=110)
    watcher.write(tmp_path / 'supervisor/cpu-completion-readiness.json', proof)
    watcher.update_tracker(latest, tmp_path)
    row = watcher.read(tmp_path / 'experiment-tracker.json')['rows'][0]
    assert row['verified_complete'] and row['status'] == 'verified_complete'
    for change in ({'scheduler':{'1':{'state':'COMPLETED'}}}, {'successes':109},
                   {'active_workers':1}, {'complete':False}, {'integrity_errors':['bad_hash']}):
        watcher.write(tmp_path / 'supervisor/cpu-completion-readiness.json', dict(proof, **change))
        watcher.update_tracker(latest, tmp_path)
        row = watcher.read(tmp_path / 'experiment-tracker.json')['rows'][0]
        assert not row['verified_complete'] and row['status'] == 'collected_pending_final_verification'


def test_budget_pool_metadata_follows_retry_without_changing_approved_totals(tmp_path):
    from copy import deepcopy
    budget = dict(approved=True,
        gpu_pool=dict(gpus=8,cpus=64,memory_gib=960,total_hours=16,total_gpu_hours=128,
                      job_id='10',scheduler_state='PENDING',scheduler_reason='Priority'),
        cpu_pool=dict(gpus=0,cpus=16,memory_gib=32,total_hours=4,
                      job_id='20',scheduler_state='RUNNING',scheduler_reason='stale'),
        shared_api=dict(luna_usd=100,luna_requests=39600,judge_usd=25,judge_requests=6600))
    watcher.write(tmp_path/'experiment-tracker.json',dict(rows=[dict(id='AS04',mode='sft_reference')],
        approved_luna_budgets=budget,proposed_luna_budgets=deepcopy(budget)))
    watcher.write(tmp_path/'approval.json',dict(attempts=[
        dict(pool='gpu',job_id='10'),dict(pool='cpu',job_id='20'),dict(pool='cpu',job_id='21')]))
    latest=dict(epoch=100,approval_mismatch=False,pools={
        'gpu':dict(job_id='10',state='PENDING',raw='10|PENDING|0:00|Resources'),
        'cpu':dict(job_id='21',state='COMPLETED',raw='21|COMPLETED|10:00|0:0')})
    watcher.update_tracker(latest,tmp_path)
    saved=watcher.read(tmp_path/'experiment-tracker.json')
    for key in ('approved_luna_budgets','proposed_luna_budgets'):
        result=saved[key]
        assert result['cpu_pool']['job_id']=='21' and result['cpu_pool']['scheduler_state']=='COMPLETED'
        assert result['cpu_pool']['original_job_id']=='20' and 'scheduler_reason' not in result['cpu_pool']
        assert result['gpu_pool']['original_job_id']=='10' and result['gpu_pool']['scheduler_reason']=='Resources'
        for pool in ('gpu_pool','cpu_pool'):
            assert {k:v for k,v in result[pool].items() if k not in ('job_id','original_job_id','scheduler_state','scheduler_reason')} == {
                k:v for k,v in budget[pool].items() if k not in ('job_id','scheduler_state','scheduler_reason')}
        assert result['shared_api']==budget['shared_api']
    # A later transition must never rewrite the first attempt identity.
    latest['pools']['cpu'].update(job_id='22',state='RUNNING')
    watcher.update_tracker(latest,tmp_path)
    assert watcher.read(tmp_path/'experiment-tracker.json')['approved_luna_budgets']['cpu_pool']['original_job_id']=='20'


def test_archived_worker_keeps_original_logs_after_replacement(tmp_path):
    plan = dict(output=str(tmp_path/'qwen/results'))
    worker = dict(worker_id='gpu-10-0-0', returncode='historical')
    directory = Path(plan['output'])/'workers'/worker['worker_id']
    watcher.write(directory/'status.json', dict(job_id='10', updated_unix=100))
    logs = tmp_path/'workers/gpu/10'/worker['worker_id']
    watcher.write(logs/'device-assignment.json', dict(gpu_uuid='GPU-original'))
    (logs/'actor.log').write_text('preserved original log\n')
    row = watcher.worker_snapshot(tmp_path, 'qwen', plan, worker, '11', 1000)
    assert row['job_id'] == '10' and row['device']['gpu_uuid'] == 'GPU-original'
    assert row['log_mtimes']['actor'] > 0 and not row['active'] and not row['stalled']


def test_direct_peer_uses_coherent_owner_counts_without_mixing_protocol_or_budget(tmp_path, monkeypatch):
    directory = tmp_path/'peer'
    resources = dict(gpus=1, cpus=8, memory_gib=120, total_seconds=14400)
    watcher.write(tmp_path/'current_kev27b_actor_full300.json', dict(
        job_id='7', run_directory=str(directory), agent_thread='other-owner',
        continuation_active=True, verified_complete=False))
    watcher.write(directory/'approval.json', dict(approved=True, resources=resources,
        attempts=[dict(job_id='7')]))
    # Collection has advanced by one episode since the coherent summary.
    watcher.write(directory/'supervisor-latest.json', dict(job_id='7', checked_unix=95,
        completed_results=78, summary=dict(completed=77, valid=76, successes=8),
        heartbeat=dict(updated_unix=94), stale_heartbeat=False))
    monkeypatch.setattr(watcher.subprocess, 'check_output', lambda *a, **k: '7|RUNNING|3000\n')
    peer = watcher.direct_actor_peer_snapshot(tmp_path, 100)
    assert (peer['episodes'], peer['valid'], peer['successes']) == (77, 76, 8)
    assert peer['remaining_approved_seconds'] == 11400 and not peer['verified_complete']
    watcher.write(tmp_path/'experiment-tracker.json', dict(rows=[
        dict(id='AS12', mode='kev27b_actor', protocol_group='kev27b_direct_actor',
             approved_resources=resources, matched_protocol=False),
        dict(id='AS07', mode='luna', protocol_group='other')],
        new_luna_episodes=1500))
    watcher.update_tracker(dict(epoch=100, approval_mismatch=False, direct_actor_peer=peer), tmp_path)
    result = watcher.read(tmp_path/'experiment-tracker.json')
    row = result['rows'][0]
    assert row['recorded_episodes'] == 77 and row['status'] == 'running'
    assert row['owning_agent_thread'] == 'other-owner' and row['approved_resources'] == resources
    assert not row['matched_protocol'] and result['new_luna_episodes'] == 1500
    # A stale supervisor from a previous allocation cannot supply fresh counts.
    watcher.write(tmp_path/'current_kev27b_actor_full300.json', dict(job_id='8', run_directory=str(directory)))
    assert not watcher.direct_actor_peer_snapshot(tmp_path, 100)['counts_available']
