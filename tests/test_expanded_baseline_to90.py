"""Continuation tests: spend caps, native scheduler restoration, safe boundaries."""
import ast
import asyncio
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_expanded_baseline_to90 as prep
import resume_expanded_baseline_to90 as run


def approval(tmp):
    a=json.loads((run.CONTROL/'approval.json').read_text())
    (tmp/'approval.json').write_text(json.dumps(a))


def test_all_native_continuation_stages_preserve_science_and_resume(monkeypatch):
    monkeypatch.setattr(run, 'endpoint', lambda: 90)
    for i in prep.MILESTONES:
        p=run.training_plan(i,9999999999)
        argv=prep.verify_native(p)
        assert p['fresh_iteration0'] is False
        assert p['environment']['WANDB_RESUME']=='must'
        assert argv[argv.index('--num-rollout')+1]=='90'
        assert str(run.SOURCE) in p['command'][1]
        assert p['environment']['OPENWEBRL_TRAINING_DEADLINE_EPOCH']=='9999999999'


def test_due_eval_precedes_more_training_and_final_eval_precedes_completion(monkeypatch):
    monkeypatch.setattr(run, 'endpoint', lambda: 90)
    assert run.next_stage(20,[10,20])==('train',30)
    assert run.next_stage(21,[10,20])==('train',30)
    assert run.next_stage(30,[10,20])==('eval',30)
    assert run.next_stage(90,list(range(10,90,10)))==('eval',90)
    assert run.next_stage(90,list(range(10,100,10)))==('complete',90)
    with pytest.raises(ValueError):run.next_stage(0,[])


def test_reduced_endpoint_requires_all_evals_and_blocks_later_training(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'CONTROL', tmp_path)
    (tmp_path/'endpoint.json').write_text(json.dumps(dict(user_requested=True, target_iteration=60)))
    assert run.next_stage(59, [10,20,30,40,50]) == ('train',60)
    assert run.next_stage(60, [10,20,30,40,50]) == ('eval',60)
    assert run.next_stage(60, [10,20,40,50,60]) == ('eval',30)
    assert run.next_stage(60, [10,20,30,40,50,60]) == ('complete',60)
    with pytest.raises(ValueError): run.next_stage(61, [])
    with pytest.raises(ValueError): run.training_plan(70, 9999999999)
    (tmp_path/'endpoint.json').write_text(json.dumps(dict(user_requested=True, target_iteration=100)))
    with pytest.raises(ValueError): run.endpoint()


def test_finish_does_not_release_budget_before_final_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'endpoint', lambda:60)
    with patch.object(run.subprocess, 'run') as cancel:
        with pytest.raises(ValueError): run.finish('1', dict(iteration=59), [10,20,30,40,50])
        cancel.assert_not_called()


def test_finish_releases_only_registered_pending_successors(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'CONTROL', tmp_path)
    monkeypatch.setattr(run, 'endpoint', lambda:60)
    (tmp_path/'attempts.json').write_text(json.dumps(dict(attempts=[dict(job_id=i) for i in ['1','2','3']])))
    with patch.object(run, 'status') as status, patch.object(run.subprocess, 'run') as cancel, \
         patch.object(run.subprocess, 'check_output', return_value='1|RUNNING\n2|PENDING\n3|COMPLETED\n999|PENDING\n'):
        run.finish('1', dict(iteration=59), [10,20,30,40,50,60])
    cancel.assert_called_once_with(['scancel','2'], check=True)
    assert status.call_args.kwargs['target_iteration'] == 60


def test_finish_handles_aged_out_attempt_and_keeps_current_job(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'CONTROL', tmp_path)
    monkeypatch.setattr(run, 'endpoint', lambda:60)
    (tmp_path/'attempts.json').write_text(json.dumps(dict(attempts=[
        dict(job_id=i) for i in ['old', 'current', 'next']])))
    with patch.object(run, 'status'), patch.object(run.subprocess, 'run') as cancel, \
         patch.object(run.subprocess, 'check_output', return_value='current|RUNNING\nnext|PENDING\n') as query:
        run.finish('current', dict(iteration=59), [10,20,30,40,50,60])
    assert query.call_count == 1
    assert '-j' not in query.call_args.args[0]
    cancel.assert_called_once_with(['scancel', 'next'], check=True)
    receipt=json.loads((tmp_path/'completion-release.json').read_text())
    assert receipt['released_jobs'] == ['next']


def test_finish_does_not_hide_scheduler_query_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'CONTROL', tmp_path)
    monkeypatch.setattr(run, 'endpoint', lambda:60)
    with patch.object(run, 'status'), patch.object(run.subprocess, 'run') as cancel, \
         patch.object(run.subprocess, 'check_output', side_effect=run.subprocess.CalledProcessError(1, ['squeue'])):
        with pytest.raises(run.subprocess.CalledProcessError):
            run.finish('current', dict(iteration=59), [10,20,30,40,50,60])
    cancel.assert_not_called()
    assert not (tmp_path/'completion-release.json').exists()


def test_final_eval_worker_stops_only_legacy_controller(monkeypatch):
    import evaluate_baseline_checkpoint
    monkeypatch.setattr(run, 'endpoint_policy', lambda:dict(target_iteration=60, legacy_controller_jobs=['1']))
    with patch.object(run, 'approval'), patch.object(run.base, 'eval_plan', return_value={}), \
         patch.object(evaluate_baseline_checkpoint, 'run') as evaluate, \
         patch.object(run, 'finish_legacy_controller') as finish:
        run.worker('1', 50)
        finish.assert_not_called()
        run.worker('2', 60)
        finish.assert_not_called()
        run.worker('1', 60)
        finish.assert_called_once_with('1')
        with pytest.raises(ValueError): run.worker('1', 70)
        assert evaluate.call_count == 3


def test_legacy_finalizer_never_returns_to_old_training_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'CONTROL', tmp_path)
    monkeypatch.setattr(run, 'endpoint', lambda:60)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    with patch.object(Path, 'read_text', return_value='/job_1/step'), \
         patch.object(run.base, 'checkpoint_review', return_value=dict(iteration=59)), \
         patch.object(run, 'evaluated', return_value=[10,20,30,40,50,60]), \
         patch.object(run, 'finish') as finish, patch.object(run.subprocess, 'run') as cancel, \
         patch.object(run.time, 'sleep', side_effect=SystemExit('cancelled')):
        with pytest.raises(SystemExit): run.finish_legacy_controller('1')
    finish.assert_called_once()
    cancel.assert_called_once_with(['scancel','1'], check=True)


def test_new_evaluation_milestones_use_exact_checkpoint_and_eval_project(tmp_path):
    training=tmp_path/'run';(training/'rollout').mkdir(parents=True)
    for i in (70,80,90):
        cp=training/f'iter_{i-1:07d}';cp.mkdir()
        for name in ['common.pt','.metadata','__0_0.distcp']:(cp/name).write_bytes(b'CPU fixture')
        (training/'rollout'/f'global_dataset_state_dict_{i-1}.pt').write_bytes(b'CPU fixture')
    with patch.object(run.base,'RUN',training),patch.object(run.base,'RUN_ID','expanded-to90-cpu-fixture'),\
         patch.object(run.base,'SOURCE',run.SOURCE):
        plans=[run.base.eval_plan('CPU_ONLY',i) for i in (70,80,90)]
    assert len({p['output'] for p in plans})==3
    for i,p in zip((70,80,90),plans):
        assert p['checkpoint_index']==i-1
        assert p['environment']['WANDB_PROJECT']=='openwebrl-evals'
        assert p['environment']['TP_SIZE']=='2' and p['gpus']==8
        assert len(set(p['expected_rollout_task_ids']))==300
        assert p['optimizer_updates_requested']==0


def test_native_scheduler_restores_original60_state_when_loop_extends90():
    ns=dict(Optional=Optional,math=math,logging=logging,logger=logging.getLogger(__name__),
            MegatronOptimizer=object,Namespace=SimpleNamespace,log_single_rank=lambda *a,**kw:None)
    for path,name in [
        (run.base.RUNTIME/'src/Megatron-LM/megatron/core/optimizer_param_scheduler.py','OptimizerParamScheduler'),
        (run.SOURCE/'slime/backends/megatron_utils/model.py','get_optimizer_param_scheduler')]:
        node=next(n for n in ast.parse(path.read_text()).body if getattr(n,'name',None)==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    def make(horizon):
        args=SimpleNamespace(num_rollout=horizon,rollout_batch_size=48,n_samples_per_prompt=5,
            global_batch_size=256,lr_decay_iters=None,lr_wsd_decay_iters=None,
            lr_warmup_fraction=None,lr_warmup_iters=0,lr_warmup_init=0,lr=1e-6,
            min_lr=0,lr_decay_style='constant',start_weight_decay=.1,end_weight_decay=.1,
            weight_decay_incr_style='constant',use_checkpoint_opt_param_scheduler=True,
            override_opt_param_scheduler=False,lr_wsd_decay_style='exponential')
        opt=SimpleNamespace(param_groups=[dict(lr_mult=1.,wd_mult=1.)])
        return ns['get_optimizer_param_scheduler'](args,opt),opt
    prior,opt=make(60);prior.step(312*256);saved=prior.state_dict()
    new,new_opt=make(90);new.load_state_dict(saved)
    assert new.num_steps==prior.num_steps and new.lr_decay_steps==prior.lr_decay_steps
    new.step(14*256)
    assert new.num_steps==(312+14)*256 and new_opt.param_groups[0]['lr']==1e-6


def test_boundary_guard_runs_before_generation_without_changing_default():
    tree=ast.parse((run.SOURCE/'train.py').read_text())
    guard=next(n for n in tree.body if getattr(n,'name',None)=='_expanded_budget_boundary')
    ns={};exec(compile(ast.Module(body=[guard],type_ignores=[]),'guard','exec'),ns)
    fn=ns['_expanded_budget_boundary']
    assert not fn({},100)
    assert not fn({'OPENWEBRL_TRAINING_DEADLINE_EPOCH':'10000'},100)
    assert fn({'OPENWEBRL_TRAINING_DEADLINE_EPOCH':'4000'},100)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For)
              and isinstance(n.target,ast.Name) and n.target.id=='rollout_id')
    assert isinstance(loop.body[0],ast.If)
    assert loop.body[0].test.func.id=='_expanded_budget_boundary'
    assert any(isinstance(n,ast.Break) for n in ast.walk(loop.body[0]))


def test_additional_budget_counts_failed_attempts_and_queued_reservations(tmp_path):
    approval(tmp_path)
    rows=[dict(job_id='1',maximum_seconds=86400),dict(job_id='2',maximum_seconds=86400),
          dict(job_id='3',maximum_seconds=86400),dict(job_id='4',maximum_seconds=86400)]
    def write_rows():
        (tmp_path/'attempts.json').write_text(json.dumps(dict(attempts=rows)))
    def response(cmd,**kw):
        if cmd[0]=='sacct':return '1|FAILED|40000|\n2|RUNNING|100|\n3|PENDING|0|\n4|PENDING|0|\n'+('5|PENDING|0|\n' if len(rows)==5 else '')
        return 'allocation fixture'
    write_rows()
    with patch.object(run,'CONTROL',tmp_path),patch.object(run.subprocess,'check_output',side_effect=response),\
         patch.object(run,'allocation',return_value=dict(maximum_seconds=86000)):
        assert 85600<run.accounting('2')-run.time.time()<86000
        ledger=json.loads((tmp_path/'attempts.json').read_text())
        assert ledger['attempts'][0]['charged_seconds']==40000
        rows.append(dict(job_id='5',maximum_seconds=86400));write_rows()
        with pytest.raises(ValueError,match='exceed approved96h'):run.accounting('2')


def test_cumulative_judge_extension_retains_spend_and_requests(tmp_path):
    spec=importlib.util.spec_from_file_location('expanded_budget_to90',run.SOURCE/'openwebrl/expanded_baseline_budget.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    (tmp_path/'approval-request.json').write_text(json.dumps(dict(approved=True,judge_cap_usd=200)))
    root=tmp_path/'judge-budget';root.mkdir()
    (root/'ledger.json').write_text(json.dumps(dict(calls=7610,charged_or_reserved_usd=78.86)))
    create=AsyncMock(return_value=SimpleNamespace(usage=SimpleNamespace(model_dump=lambda:dict(prompt_tokens=5000,completion_tokens=100))))
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert mod.BudgetedJudge(client,tmp_path).cap==200
    ext=tmp_path/'to90-20261005';ext.mkdir();approval(ext)
    judge=mod.BudgetedJudge(client,tmp_path);assert judge.cap==600
    args=dict(model='gpt-4.1',messages=[dict(role='user',content='same judge request')],seed=42)
    asyncio.run(judge.create(**args));create.assert_awaited_once_with(**args)
    saved=json.loads((root/'ledger.json').read_text())
    assert saved['calls']==7611 and saved['charged_or_reserved_usd']==pytest.approx(78.8708)
    with pytest.raises(RuntimeError):judge.reserve(600)


def test_judge_extension_requires_approved_exact_cap(tmp_path):
    spec=importlib.util.spec_from_file_location('expanded_budget_guard',run.SOURCE/'openwebrl/expanded_baseline_budget.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    (tmp_path/'approval-request.json').write_text(json.dumps(dict(approved=True,judge_cap_usd=200)))
    ext=tmp_path/'to90-20261005';ext.mkdir();approval(ext)
    a=json.loads((ext/'approval.json').read_text());a['approved']=False
    (ext/'approval.json').write_text(json.dumps(a))
    assert mod.BudgetedJudge(None,tmp_path).cap==200
    a['approved']=True;a['judge_lifetime_cap_usd']=601
    (ext/'approval.json').write_text(json.dumps(a))
    with pytest.raises(ValueError,match='Invalid cumulative'):mod.BudgetedJudge(None,tmp_path)
