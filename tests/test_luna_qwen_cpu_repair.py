from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

CONTROLLER = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/luna-qwen-full300-20261004/repairs/luna-pixel-coordinates-20261005/run_cpu_repair.py')
spec = importlib.util.spec_from_file_location('luna_cpu_repair', CONTROLLER)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def registry_fixture():
    baseline = dict(approved=True, plan_sha256='unchanged', limits=repair.LIMITS,
        pools=dict(cpu=repair.RESOURCES, gpu=dict(gpus=8,total_seconds=57600)),
        attempts=[dict(pool='gpu',job_id='gpu',time_limit_seconds=57600),
                  dict(pool='cpu',job_id='old',time_limit_seconds=14400)])
    original = deepcopy(baseline)
    attempt = dict(pool='cpu',job_id='retry',time_limit_seconds=7620)
    original['attempts'].append(attempt)
    plan = {'repair_id':'pixel-fix'}
    approval = dict(approved=True,repair_plan_sha256=repair.digest(plan),
        resources=repair.RESOURCES,limits=repair.LIMITS,attempts=[attempt])
    return original, baseline, approval, plan


def test_original_registry_may_append_cpu_retry_but_cannot_reset_caps_or_rewrite_attempts():
    original, baseline, approval, plan = registry_fixture()
    repair.check_registry(original,baseline,approval,plan)
    original['attempts'][1]['time_limit_seconds'] = 1
    with pytest.raises(ValueError,match='rewritten'):
        repair.check_registry(original,baseline,approval,plan)
    original, baseline, approval, plan = registry_fixture()
    original['pools']['cpu']['total_seconds'] += 60
    with pytest.raises(ValueError,match='budget fields changed'):
        repair.check_registry(original,baseline,approval,plan)


def test_repair_cannot_register_gpu_or_unregistered_job():
    original, baseline, approval, plan = registry_fixture()
    approval['attempts']=[dict(pool='gpu',job_id='gpu',time_limit_seconds=57600)]
    with pytest.raises(ValueError,match='appended original CPU'):
        repair.check_registry(original,baseline,approval,plan)
    approval['attempts']=[dict(pool='cpu',job_id='unregistered',time_limit_seconds=7620)]
    with pytest.raises(ValueError,match='appended original CPU'):
        repair.check_registry(original,baseline,approval,plan)


def test_final_failed_attempt_runtime_is_charged_to_original_four_hours():
    _, _, approval, _ = registry_fixture()
    assert repair.check_attempt(approval,'retry',6761)['time_limit_seconds']==7620
    with pytest.raises(ValueError,match='four-hour total'):
        repair.check_attempt(approval,'retry',6781)
    approval['attempts'][0]['time_limit_seconds'] = 14400
    with pytest.raises(ValueError,match='four-hour total'):
        repair.check_attempt(approval,'retry',6761)


def test_only_audited_safe_originals_or_registered_repair_records_survive(tmp_path):
    family = dict(output=str(tmp_path),protocol={'fixed':True},
        schedule=[dict(task_id=str(i),modes=['luna']) for i in range(300)])
    plan = dict(family_plan_sha256='original-family',retained_records={},archived_records={})
    approval = dict(attempts=[dict(job_id='retry')])
    name = repair.digest(['0','luna'])[:24]+'.json'
    path = tmp_path/'records'/name
    row = dict(task_id='0',mode='luna',plan_sha256='original-family',job_id='old',worker_id='old-worker')
    repair.write(path,row)
    verify = lambda path, protocol:json.loads(path.read_text())
    with pytest.raises(ValueError,match='Coordinate-exposed original'):
        repair.check_primary_records(plan,approval,family,verify)
    plan['retained_records'][name] = repair.sha(path)
    assert repair.check_primary_records(plan,approval,family,verify)==1
    row['reward']=1
    repair.write(path,row)
    with pytest.raises(ValueError,match='unaffected record changed'):
        repair.check_primary_records(plan,approval,family,verify)
    plan['retained_records']={}
    row.update(job_id='retry',worker_id='cpu-retry-repair-0')
    repair.write(path,row)
    assert repair.check_primary_records(plan,approval,family,verify)==1


def test_corrupted_archive_blocks_recovery_and_duplicate_tasks_block_endpoint(tmp_path):
    family = dict(output=str(tmp_path),protocol={},
        schedule=[dict(task_id=str(i),modes=['luna']) for i in range(300)])
    archived=tmp_path/'archived.json';archived.write_text('{}')
    plan=dict(retained_records={},archived_records={'old.json':dict(path=str(archived),sha256=repair.sha(archived))})
    assert repair.check_primary_records(plan,dict(attempts=[]),family,None)==0
    archived.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Superseded original evidence'):
        repair.check_primary_records(plan,dict(attempts=[]),family,None)
    family['schedule'][-1]['task_id']='0'
    with pytest.raises(ValueError,match='300 unique'):
        repair.check_primary_records(plan,dict(attempts=[]),family,None)
