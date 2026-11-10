from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_luna_qwen_full300 as controller
from openwebrl.luna_qwen_metrics import digest, write_json


def approval_fixture():
    plan = dict(pools={
        'gpu': dict(resources=dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960, total_seconds=72000)),
        'cpu': dict(resources=dict(gpus=0, cpus=16, memory_gib=32, total_seconds=14400))},
        limits=dict(luna_usd=100., luna_calls=39600, judge_usd=25., judge_calls=6600))
    approval = dict(approved=True, plan_sha256=digest(plan), limits=deepcopy(plan['limits']),
        pools={name: deepcopy(value['resources']) for name, value in plan['pools'].items()},
        attempts=[dict(pool='gpu', job_id='1', time_limit_seconds=36000),
                  dict(pool='gpu', job_id='2', time_limit_seconds=36000),
                  dict(pool='cpu', job_id='3', time_limit_seconds=14400)])
    return plan, approval


def test_both_pools_require_exact_user_approval_and_plan():
    plan, approved = approval_fixture()
    controller.check_approval(approved, plan, 'gpu', '2', 36000)
    for changed in (dict(approved=False), dict(plan_sha256='changed'),
                    dict(limits=dict(approved['limits'], luna_usd=101.))):
        with pytest.raises(ValueError, match='approval'):
            controller.check_approval(dict(approved, **changed), plan, 'gpu', '2', 36000)
    with pytest.raises(ValueError, match='approved pool'):
        controller.check_approval(approved, plan, 'cpu', '2', 0)


def test_retry_counts_failed_time_and_never_borrows_other_pool_budget():
    plan, approval = approval_fixture()
    controller.check_approval(approval, plan, 'gpu', '2', 36000)
    with pytest.raises(ValueError, match='never resets'):
        controller.check_approval(approval, plan, 'gpu', '2', 36001)
    controller.check_approval(approval, plan, 'cpu', '3', 0)
    with pytest.raises(ValueError, match='never resets'):
        controller.check_approval(approval, plan, 'cpu', '3', 1)
    approval['attempts'].append(deepcopy(approval['attempts'][0]))
    with pytest.raises(ValueError, match='exactly once'):
        controller.check_approval(approval, plan, 'gpu', '2', 0)


def test_prior_accounting_is_same_pool_only_and_requires_final_state(monkeypatch):
    _, approval = approval_fixture()
    queries = []
    def output(command, **kwargs):
        queries.append(command[command.index('-j') + 1])
        return '1|FAILED|120|cpu=64,gres/gpu=8|\n'
    monkeypatch.setattr(controller.subprocess, 'check_output', output)
    used, rows = controller.previous_elapsed(approval, 'gpu', '2')
    assert used == 120 and len(rows) == 1 and queries == ['1']
    # The other pool may run at the same time without blocking this one.
    assert controller.previous_elapsed(approval, 'cpu', '3') == (0, [])
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **k: '1|RUNNING|120|\n')
    with pytest.raises(ValueError, match='final scheduler accounting'):
        controller.previous_elapsed(approval, 'gpu', '2')


def test_initial_eight_slots_start_both_families_and_free_slots_are_reused():
    pending, active = dict(sft=300, qwen=300), dict(sft=0, qwen=0)
    assignments = []
    for slot in range(8):
        family = controller.choose_family(pending, active, 'sft' if slot < 4 else 'qwen')
        assignments.append(family)
        active[family] += 1
    assert assignments == ['sft'] * 4 + ['qwen'] * 4
    assert controller.choose_family(dict(sft=0, qwen=20), dict(sft=0, qwen=4)) == 'qwen'
    assert controller.choose_family(dict(sft=1, qwen=0), dict(sft=1, qwen=0)) is None


def test_failed_family_is_not_blindly_restarted_while_healthy_family_continues():
    pending, active = dict(sft=299, qwen=10), dict(sft=0, qwen=3)
    assert controller.choose_family(pending, active, initial='sft', disabled={'sft'}) == 'qwen'
    assert controller.choose_family(dict(sft=299, qwen=0), dict(sft=0, qwen=0), disabled={'sft'}) is None
    assert controller.choose_family(dict(luna=300), dict(luna=0)) == 'luna'


def test_pending_task_blocks_require_every_mode_and_preserve_completed_records(tmp_path):
    schedule = [dict(task_id='a', modes=['sft_luna5', 'sft_luna10']),
                dict(task_id='b', modes=['sft_luna5', 'sft_luna10'])]
    assert controller.pending_blocks(tmp_path, schedule) == 2
    path = tmp_path / 'records' / (digest(['a', 'sft_luna5'])[:24] + '.json')
    write_json(path, dict(retained=True))
    original = path.read_bytes()
    assert controller.pending_blocks(tmp_path, schedule) == 2
    write_json(tmp_path / 'records' / (digest(['a', 'sft_luna10'])[:24] + '.json'), {})
    assert controller.pending_blocks(tmp_path, schedule) == 1
    assert path.read_bytes() == original


def allocation_info(pool='gpu', **changes):
    fields = dict(JobId='2', JobState='RUNNING', UserId='test(100)', NumNodes='1',
        NumCPUs='64' if pool == 'gpu' else '16', TimeLimit='20:00:00' if pool == 'gpu' else '04:00:00',
        EndTime='2026-10-05T20:00:00',
        AllocTRES='cpu=64,mem=960G,gres/gpu=8,gres/gpu:h200=8' if pool == 'gpu' else 'cpu=16,mem=32G')
    fields.update(changes)
    return ' '.join(f'{k}={v}' for k, v in fields.items())


def test_actual_allocation_must_match_exact_gpu_and_cpu_profiles():
    now = datetime(2026, 10, 5, 16, 0, 0).timestamp()
    gpu = controller.validate_allocation(allocation_info(), 'gpu', '2', 72000, now=now, uid=100)
    assert gpu['remaining_seconds'] == 14400 - 180
    cpu = controller.validate_allocation(allocation_info('cpu'), 'cpu', '2', 14400, now=now, uid=100)
    assert cpu['resources']['gpus'] == 0
    for changed in (dict(NumCPUs='128'), dict(AllocTRES='cpu=64,mem=960G,gres/gpu=16,gres/gpu:h200=16'),
                    dict(AllocTRES='cpu=64,mem=960G,gres/gpu=8,gres/gpu:a100=8')):
        with pytest.raises(ValueError, match='exact approved pool'):
            controller.validate_allocation(allocation_info(**changed), 'gpu', '2', 72000, now=now, uid=100)
    with pytest.raises(ValueError, match='exact approved pool'):
        controller.validate_allocation(allocation_info('cpu', AllocTRES='cpu=16,mem=32G,gres/gpu=1'),
            'cpu', '2', 14400, now=now, uid=100)
    with pytest.raises(ValueError, match='time limit exceeds'):
        controller.validate_allocation(allocation_info(), 'gpu', '2', 36000, now=now, uid=100)


def test_srun_resources_override_inherited_gpu_count_and_do_not_overlap():
    gpu = controller.worker_command(Path('/frozen'), 'gpu', '1', Path('/cfg'))
    assert '--gpus=h200:1' in gpu and '--gres=gpu:h200:1' in gpu and '--cpus-per-task=8' in gpu
    assert '--mem=120G' in gpu and '--exact' in gpu and '--overlap' not in gpu
    cpu = controller.worker_command(Path('/frozen'), 'cpu', '3', Path('/cfg'))
    assert '--gres=none' in cpu and '--cpus-per-task=4' in cpu and '--mem=8G' in cpu
    assert not any(flag.startswith('--gpus') for flag in cpu)


def test_final_cohort_verification_checks_artifact_hash_and_record_identity(tmp_path):
    artifact = tmp_path / 'result.json'
    artifact.write_text('{}')
    plan = dict(output=str(tmp_path), protocol=dict(setting='fixed'),
                schedule=[dict(task_id='a', modes=['qwen_luna5'])])
    path = tmp_path / 'records' / (digest(['a', 'qwen_luna5'])[:24] + '.json')
    write_json(path, dict(task_id='a', mode='qwen_luna5', artifact=str(artifact),
        artifact_sha256=controller.sha(artifact), protocol_sha256=digest(plan['protocol'])))
    assert controller.verify_records(plan) == 1
    artifact.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Saved result changed'):
        controller.verify_records(plan)


def test_runtime_credentials_use_authorized_project_file_without_overriding_inheritance(tmp_path, monkeypatch):
    env_file = tmp_path / '.env'
    env_file.write_text('OPENAI_API_KEY=file-placeholder\nWANDB_API_KEY=wandb-placeholder\n'
                        'UNRELATED_SECRET=excluded\nWANDB_PROJECT=wrong-project\n')
    monkeypatch.setattr(controller, 'PROJECT_ENV', env_file)
    monkeypatch.setattr(controller, 'environment', lambda: {'OPENAI_API_KEY': 'inherited-placeholder'})
    env = controller.worker_environment()
    assert env['OPENAI_API_KEY'] == 'inherited-placeholder'
    assert env['WANDB_API_KEY'] == 'wandb-placeholder'
    assert env['WANDB_PROJECT'] == 'openwebrl-evals'
    assert 'UNRELATED_SECRET' not in env
    assert env['JUDGE_API_BASE'] == 'https://api.openai.com/v1'
    monkeypatch.setattr(controller, 'environment', lambda: {})
    env_file.write_text('WANDB_API_KEY=placeholder\n')
    with pytest.raises(ValueError, match='OPENAI_API_KEY is absent'):
        controller.worker_environment()
