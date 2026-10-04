"""Explicitly synthetic labels/rewards for GPU plumbing, never experiment data."""
import json
import math
import os
from pathlib import Path
import urllib.request


def sample_templates(args,rollout_id,output):
    """Five actual current-actor responses with aligned native token logprobs."""
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.hf_checkpoint)
    prompt=tokenizer.apply_chat_template([dict(role='user',content='Write one small integer.')],
        tokenize=True,add_generation_prompt=True)
    templates=[]
    for index in range(5):
        payload=dict(input_ids=prompt,return_logprob=True,sampling_params=dict(max_new_tokens=8,
            temperature=.8,top_p=.9,sampling_seed=42+5*rollout_id+index))
        request=urllib.request.Request(f'http://{args.sglang_router_ip}:{args.sglang_router_port}/generate',
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=90) as response: result=json.load(response)
        pairs=result.get('meta_info',{}).get('output_token_logprobs',[])
        if not pairs or any(len(x)<2 or not isinstance(x[1],int) or x[0] is None
                or not math.isfinite(x[0]) or x[0]>1e-4 for x in pairs):
            raise ValueError('Missing or invalid current-actor token/log-probability pairs')
        templates.append(dict(tokens=prompt+[x[1] for x in pairs],response=result['text'],
            response_length=len(pairs),rollout_log_probs=[x[0] for x in pairs]))
    Path(output,'actor-templates.json').write_text(json.dumps(dict(rollout_id=rollout_id,
        actor_responses_are_real=True,rewards_and_labels_are_synthetic=True,templates=templates)))
    return templates


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation or os.environ.get('ARM_GPU_DIAGNOSTIC') != '1':
        raise ValueError('Synthetic fixture requires the isolated GPU diagnostic')
    from slime.utils.types import Sample
    from slime.rollout.base_types import RolloutFnTrainOutput
    from openwebrl.arm_turn_bonus_cycles import config
    current=config()
    if 'gpu-diagnostic' not in current['run_id']:
        raise ValueError('Synthetic data must not enter a real experiment')
    templates=sample_templates(args,rollout_id,current['output'])
    data_source.get_samples(24)
    rows=[]
    for group in range(24):
        for trajectory in range(5):
            parent=group*5+trajectory
            template=templates[trajectory]
            for turn in range(10):
                selected=0 if trajectory==group%5 else 1
                label={} if turn else dict(eligible=True,policy_id=current['policy_id'],
                    executed_index=0,selected_index=selected,unit_bonus=.8 if selected==0 else -.2,
                    reason='admitted',diagnostic_synthetic=True)
                rows.append(Sample(index=(parent<<16)|turn,group_index=group,
                    **template,loss_mask=[1]*template['response_length'],
                    status=Sample.Status.COMPLETED,reward=float(trajectory<2),
                    metadata=dict(task_id=f'synthetic-gpu-test-{group}',trajectory_id=parent,turn_index=turn,
                        num_turns_in_trajectory=10,arm_turn_bonus=label,diagnostic_synthetic=True)))
    Path(current['output'],'synthetic-fixture.json').write_text(json.dumps(dict(
        diagnostic_only=True,rows=len(rows),synthetic_labels=120,
        limitation='Tests GPU optimizer and native control flow, not live ARM calibration or policy quality')))
    return RolloutFnTrainOutput(samples=rows,metrics={'diagnostic/synthetic_fixture':1})
