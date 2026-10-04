"""CPU-only checks for cross-stage data identity, recovery and spend bounds."""
import asyncio
import ast
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
import prepare_expanded_baseline as prep
import run_expanded_baseline as run

spec=importlib.util.spec_from_file_location('expanded_budget',ROOT/'openwebrl/expanded_baseline_budget.py')
budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)


def approval(root,approved=True):
    (root/'approval-request.json').write_text(json.dumps(dict(approved=approved,
        resources=dict(jobs=1,gpus=8,gpu_type='H200',hours=24,gpu_hours=192,cpus=64,memory_gib=960),judge_cap_usd=200,
        target_iteration=60,full300_evaluations=[10,20,30,40,50,60])))


def test_inherited_experiment_settings_do_not_reach_fresh_run():
    with patch.dict(os.environ,dict(OPENWEBRL_ARM_TURN_BONUS_CONFIG='/bad/config',
        OPENWEBRL_REPLAY_FIRST_BATCH='/bad/batch',SLIME_LOAD_CHECKPOINT='/bad/checkpoint',
        SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH='/bad/blacklist',WANDB_RUN_ID='old-run',
        OPENWEBRL_RECORD_EVAL_CONFIG='/bad/eval',OPENWEBRL_PENDING_EVAL_ITERATION='100')):
        env=prep.clean_env();env.update(prep.environment(1))
    assert 'SLIME_LOAD_CHECKPOINT' not in env
    assert not any(k.startswith(('OPENWEBRL_ARM','OPENWEBRL_REPLAY_')) for k in env)
    assert env['WANDB_RUN_ID']==prep.RUN_ID and env['TP_SIZE']=='2'
    assert env['TRAIN_DATA'].endswith('combined-tasks.parquet')
    assert 'OPENWEBRL_RECORD_EVAL_CONFIG' not in env


def test_native_scheduler_initializes_and_preserves_state_across_stage_boundaries():
    # Execute the actual native scheduler constructor and factory on CPU.
    # Omit GPU-dependent imports and substitute only the optimizer container
    # and distributed log sink; all schedule logic and assertions are native.
    scheduler_path=prep.RUNTIME/'src/Megatron-LM/megatron/core/optimizer_param_scheduler.py'
    factory_path=prep.SOURCE/'slime/backends/megatron_utils/model.py'
    ns=dict(Optional=Optional,math=math,logging=logging,logger=logging.getLogger(__name__),
            MegatronOptimizer=object,Namespace=SimpleNamespace,log_single_rank=lambda *a,**kw:None)
    for path,name in [(scheduler_path,'OptimizerParamScheduler'),(factory_path,'get_optimizer_param_scheduler')]:
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
    # Reproduce the reported production failure before verifying the fix.
    with pytest.raises(AssertionError):make(1)
    saved=None
    for target in prep.TRAINING_STAGES:
        plan=prep.training_command(target,resume=target>1)
        env=plan['environment']
        assert env['OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT']==str(target-1)
        scheduler,opt=make(int(env['NUM_ROLLOUT']))
        assert scheduler.lr_decay_steps==14336
        if saved is not None:
            scheduler.load_state_dict(saved)
            assert scheduler.num_steps==saved['num_steps']
        scheduler.step(256)
        assert opt.param_groups[0]['lr']==1e-6 and opt.param_groups[0]['weight_decay']==.1
        saved=scheduler.state_dict()


def test_explicit_browser_gate_matches_pool_without_changing_scientific_yaml():
    import yaml
    relative='openwebrl/browser_training_config.yaml'
    old=yaml.safe_load((prep.PARENT/relative).read_text())
    new=yaml.safe_load((prep.SOURCE/relative).read_text())
    assert {k for k in set(old)|set(new) if old.get(k)!=new.get(k)}=={'browser_rollout_concurrency'}
    assert new['browser_rollout_concurrency']==64
    source=prep.SOURCE/'openwebrl/generate_browser.py'
    node=next(n for n in ast.parse(source.read_text()).body if getattr(n,'name',None)=='_get_browser_rollout_concurrency')
    ns=dict(os=os,Any=object)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    # Production gives the explicit YAML gate precedence over environment.
    with patch.dict(os.environ,{'SLIME_BROWSER_ROLLOUT_CONCURRENCY':'64'}):
        assert ns['_get_browser_rollout_concurrency'](SimpleNamespace(**old))==32
        assert ns['_get_browser_rollout_concurrency'](SimpleNamespace(**new))==64
    assert prep.environment(10,resume=True)['BROWSER_CONCURRENCY']=='64'


@pytest.mark.parametrize('browser_root', [prep.BROWSER_PARENT, prep.SOURCE])
def test_frozen_browser_runtime_includes_actual_child_launch_environment(browser_root):
    # The helper alone is insufficient: the actual WebEnv.setup call must pass
    # its result to Chromium. Execute the preserved parent's real setup method
    # with only browser/network interfaces replaced, so this catches the
    # previous omission of web_env.py from the copied runtime files.
    assert 'openwebrl/env/web_env.py' in prep.BROWSER_RUNTIME_FILES
    path=browser_root/'openwebrl/env/web_env.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='WebEnv')
    setup=next(n for n in cls.body if getattr(n,'name',None)=='setup')
    launch=AsyncMock(return_value='browser')
    playwright=SimpleNamespace(chromium=SimpleNamespace(launch=launch,name='chromium'))
    child_env={'__EGL_VENDOR_LIBRARY_FILENAMES':'/verified/mesa.json','CUDA_VISIBLE_DEVICES':'0,1'}
    calls=[]
    def environment(flags):
        calls.append(flags)
        return child_env
    ns=dict(async_playwright=lambda:SimpleNamespace(start=AsyncMock(return_value=playwright)),
            browser_process_environment=environment,logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[setup],type_ignores=[]),str(path),'exec'),ns)
    obj=SimpleNamespace(browser_args=['--disable-gpu'],proxy_settings=None,
        should_record=False,start_url='about:blank',auth_info=None,
        _initialize_context=AsyncMock(),screen_size=(1280,720),dpr=1,
        timeout=30000,screenshot_timeout=30000)
    actor_env=dict(os.environ)
    asyncio.run(ns['setup'](obj))
    assert calls==[['--disable-gpu']]
    assert launch.call_args.kwargs['env'] is child_env
    assert dict(os.environ)==actor_env


def test_eval_milestones_use_own_paths_project_and_exact_checkpoint(tmp_path):
    # build_plan checks component presence; GPU/model restoration remains a
    # production startup check, not something these synthetic files establish.
    training=tmp_path/'run';(training/'rollout').mkdir(parents=True)
    assert prep.TRAINING_STAGES == (1,10,20,30,40,50,60)
    for i in prep.EVAL_ITERATIONS:
        cp=training/f'iter_{i-1:07d}';cp.mkdir()
        for f in ['common.pt','.metadata','__0_0.distcp']:(cp/f).write_bytes(b'CPU fixture')
        (training/'rollout'/f'global_dataset_state_dict_{i-1}.pt').write_bytes(b'CPU fixture')
    with patch.object(run,'RUN',training),patch.object(run,'RUN_ID','expanded-cpu-path-test'):
        plans=[run.eval_plan('CPU_ONLY',i) for i in prep.EVAL_ITERATIONS]
    assert len({p['output'] for p in plans})==6
    for i,p in zip(prep.EVAL_ITERATIONS,plans):
        assert p['checkpoint_index']==i-1
        assert p['environment']['WANDB_PROJECT']=='openwebrl-evals'
        assert p['environment']['TP_SIZE']=='2' and p['gpus']==8
        assert p['environment']['TRAIN_DATA']==str(prep.CONTROL/'combined-tasks.parquet')
        assert p['environment']['OPENWEBRL_EVAL_ROLLOUT_DIR'].endswith(f'iter{i}/rollouts')
        assert len(set(p['expected_rollout_task_ids']))==300
        assert p['optimizer_updates_requested']==0


def test_judge_requests_unchanged_and_failures_keep_reservations(tmp_path):
    approval(tmp_path)
    create=AsyncMock(return_value=SimpleNamespace(usage=SimpleNamespace(model_dump=lambda:dict(prompt_tokens=5000,completion_tokens=100))))
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    judge=budget.BudgetedJudge(client,tmp_path)
    kwargs=dict(model='gpt-4.1',messages=[dict(role='user',content='Judge this task')],seed=42)
    asyncio.run(judge.create(**kwargs));create.assert_awaited_once_with(**kwargs)
    first=json.loads((tmp_path/'judge-budget/ledger.json').read_text())
    assert first['charged_or_reserved_usd']==pytest.approx(.0108)
    create.side_effect=TimeoutError('interrupted')
    with pytest.raises(TimeoutError):asyncio.run(judge.create(**kwargs))
    interrupted=json.loads((tmp_path/'judge-budget/ledger.json').read_text())
    assert interrupted['calls']==2 and interrupted['charged_or_reserved_usd']>first['charged_or_reserved_usd']
    # A replacement object/process shares the durable charged/reserved balance.
    other=budget.BudgetedJudge(client,tmp_path)
    with pytest.raises(RuntimeError):other.reserve(200)
    assert (tmp_path/'judge-budget/halt.json').exists()
    assert create.await_count==2


def test_no_approval_or_unregistered_job_fails_closed(tmp_path):
    approval(tmp_path,False)
    with pytest.raises(PermissionError):budget.BudgetedJudge(None,tmp_path)
    with patch.object(run,'CONTROL',tmp_path):
        with pytest.raises(PermissionError):run.verify_approval('99')
        approval(tmp_path,True)
        (tmp_path/'attempts.json').write_text(json.dumps(dict(attempts=[])))
        with pytest.raises(PermissionError):run.verify_approval('99')


def test_retry_does_not_reset24hour_budget(tmp_path):
    approval(tmp_path)
    attempts=[dict(job_id='old',maximum_seconds=86400),dict(job_id='new',maximum_seconds=86400)]
    (tmp_path/'attempts.json').write_text(json.dumps(dict(attempts=attempts)))
    def response(cmd,**kw):
        if cmd[0]=='sacct':
            job=cmd[cmd.index('-j')+1]
            return f'{job}|'+('FAILED|3600|1:0|\n' if job=='old' else 'RUNNING|60|0:0|\n')
        return 'fixture allocation'
    with patch.object(run,'CONTROL',tmp_path),patch.object(run.subprocess,'check_output',side_effect=response),patch.object(run,'allocation',return_value=dict(maximum_seconds=85000)):
        with pytest.raises(ValueError,match='remaining original budget'):run.accounting('new')
        attempts[1]['maximum_seconds']=82800
        (tmp_path/'attempts.json').write_text(json.dumps(dict(attempts=attempts)))
        _,end=run.accounting('new')
        assert 82400<end-run.time.time()<82600


def test_missing_rollout_archive_prevents_completion(tmp_path):
    (tmp_path/'rollouts').mkdir()
    (tmp_path/'status.json').write_text(json.dumps(dict(complete=True,returncode=0)))
    (tmp_path/'evaluation_manifest.json').write_text(json.dumps(dict(expected_rollout_task_ids=['a'])))
    (tmp_path/'rollouts/a.json').write_text(json.dumps(dict(task_id='a',judge_model='gpt-4.1',
        judge_prompt_variant='action_history',rollout_file='missing.pt')))
    with pytest.raises(FileNotFoundError):run.audit_eval(tmp_path)
