#!/usr/bin/env python3
"""Freeze the outcome-only expanded task pool and reviewable training recipe.

CPU metadata/data preparation only. Does not allocate GPUs or launch training.
"""
import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
POOL = RUNTIME/'arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929'
CONTROL = POOL/'outcome-only-expanded-20261002'
ORIGINAL = REPO/'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet'
ADDITIONAL = RUNTIME/'reference-arm-task-screen-sharded-20261002-v2/screen-tasks.jsonl'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def instruction(row):
    return re.sub(r'\s+', ' ', row['prompt'][-1]['content']).strip().casefold()

def freeze(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refuse to replace frozen input: '+str(path))
    else:
        path.write_bytes(data)

def prepare():
    import pyarrow as pa
    import pyarrow.parquet as pq
    original = pq.read_table(ORIGINAL).to_pylist()
    additional = [json.loads(line) for line in ADDITIONAL.read_text().splitlines()]
    assert len(original) == 2102 and len(additional) == 2000
    old_ids = {r['metadata']['task_id'] for r in original}
    new_ids = {r['metadata']['task_id'] for r in additional}
    assert len(old_ids) == 2102 and len(new_ids) == 2000 and not old_ids & new_ids
    assert not set(map(instruction, original)) & set(map(instruction, additional))
    assert all(set(r) == {'prompt','metadata'} for r in original+additional)
    assert all(set(r['metadata']) == {'task_id','task','start_url'} for r in original+additional)
    assert all(r['prompt'][-1]['content'] == r['metadata']['task'] for r in original+additional)
    # Interleave sources before the unchanged native epoch shuffle; no outcome
    # information, difficulty score, ARM label, or teacher answer enters rows.
    rows = sorted(original+additional, key=lambda r:hashlib.sha256(('42/'+r['metadata']['task_id']).encode()).hexdigest())
    CONTROL.mkdir(parents=True, exist_ok=True)
    freeze(CONTROL/'combined-tasks.jsonl', ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode())
    p = CONTROL/'combined-tasks.parquet'
    if p.exists():
        assert pq.read_table(p).to_pylist() == rows
    else:
        pq.write_table(pa.Table.from_pylist(rows), p)
    assert pq.read_table(p).to_pylist() == rows
    sources = {'original':str(ORIGINAL), 'original_sha256':sha(ORIGINAL),
               'additional':str(ADDITIONAL), 'additional_sha256':sha(ADDITIONAL)}
    provenance = {'original_ids':sorted(old_ids),'additional_ids':sorted(new_ids),'sources':sources}
    freeze(CONTROL/'private-provenance.json',(json.dumps(provenance,indent=2)+'\n').encode())
    plan = {
      'experiment':'Outcome-only RL, original2102 plus selected2000 tasks',
      'prepared_data':True,'compute_approved':False,'submitted':False,
      'training_controller_preflight_pending':True,
      'data':{'path':str(p),'sha256':sha(p),'jsonl_sha256':sha(CONTROL/'combined-tasks.jsonl'),
              'original_tasks':2102,'new_tasks':2000,'total_tasks':4102,
              'cross_pool_id_overlap':0,'cross_pool_normalized_instruction_overlap':0,
              'new_task_proportion':2000/4102,'all_new_tasks_retained':True,
              'no_filtering_or_weighting_by_saved_actor_or_arm_outcomes':True,
              'model_visible_fields':'task instruction and start URL only; native prompt/metadata schema',
              'sampling':'unchanged native shuffled pool, no source reweighting or new adaptive curriculum',
              'private_task_payloads':True,'sources':sources},
      'initialization':{'model':'OpenWebRL/OpenWebRL-4B-SFT','checkpoint':'/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT',
                        'iteration':0,'optimizer':'fresh','scheduler':'fresh','task_cursor':0,'seed':42,'new_wandb_lineage':True},
      'training':{'reward':'unchanged native outcome-only','arm_selection':False,'arm_bonus':False,
                  'groups_per_iteration':48,'rollouts_per_task':5,
                  'dynamic_filter':'slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std',
                  'advantage_estimator':'grpo','ppo_epochs':2,'global_batch':256,'micro_batch':1,
                  'lr':1e-6,'lr_schedule':'constant','optimizer':'adam','betas':[0.9,0.98],'weight_decay':0.1,
                  'clip_low':0.2,'clip_high':0.28,'kl_coefficient':0,'entropy_coefficient':0,
                  'temperature':0.8,'top_p':1.0,'top_k':-1,'response_tokens':1024,'context_tokens':32768,
                  'max_turns':15,'screenshots':1,'reasoning_history':'full','judge':'gpt-4.1/action_history'},
      'systems':{'gpus':8,'gpu_type':'H200','tp':2,'dp':4,'browser_concurrency':64,'cuda_cache_limit_gib':48,
                 'full_activation_recomputation':True,'cpu_preflight_required':True,'gpu_startup_validation_required':True},
      'first_endpoint':{'training_iteration':20,'full300_eval_iterations':[10,20],
                        'endpoint_contingent_on_measured_throughput':True,'continuation_beyond_approved_budget_requires_new_approval':True},
      'evaluation':{'benchmark':'Online-Mind2Web','tasks':300,'browser':'local','temperature':0,
                    'judge':'gpt-4.1/action_history','max_turns':30,'response_tokens':4096,
                    'save_all_rollouts_and_verdicts':True,'separate_eval_project':'openwebrl-evals'},
      'proposal':{'jobs':1,'gpus':8,'gpu_type':'H200','hours':24,'gpu_hours':192,'cpus':64,'memory_gib':960,
                  'includes':'training, milestone evaluations, startup checks and all retries; stop early when target20/eval20 verified',
                  'terminal_judge_estimate_usd':[100,200],'compute_approval_pending':True},
      'comparison':{'control':'Historical original2102 outcome-only baseline, matched iteration evaluations',
                    'changed_scientific_knob':'training task pool only',
                    'caveat':'Historical comparison is exploratory; live-web/time/topology differences remain. Track completed optimizer updates and browser cost as well as iterations.',
                    'report_by_source':['proposed groups','accepted mixed groups','invalid rate','reward','time per accepted group']},
      'launch_readiness_checks':['Frozen union membership/schema/hash verified','No archived outcomes or ARMlabels injected into prompts',
                                'Pending: frozen baseline runtime with TP2/DP4 and recovery safeguards',
                                'Pending: native arguments, both milestone output paths, W&B projects and judge-budget approval verification',
                                'Pending: Slurm admission and GPU startup in the approved allocation']}
    freeze(CONTROL/'training-plan.json',(json.dumps(plan,indent=2)+'\n').encode())
    print(json.dumps({'control':str(CONTROL),'tasks':4102,'cross_pool_overlap':0,'dataset_sha256':sha(p),'compute_approved':False,'submitted':False}))
    return plan

if __name__=='__main__':
    prepare()
