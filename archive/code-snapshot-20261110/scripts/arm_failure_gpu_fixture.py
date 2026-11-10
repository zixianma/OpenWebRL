"""Isolated diagnostic: real model calls, explicitly constructed outcomes/panels.

This is NOT task-success evidence or an experimental training dataset.
"""
import asyncio
import base64
from copy import deepcopy
import json
import math
import os
from pathlib import Path


def diagnostic_task(config):
    from openwebrl.arm_failure_coverage import sampled_at
    # Deterministically choose an identity that exercises both nested samples.
    for number in range(1000):
        task = f'diagnostic-failure-{number}'
        identities = [(config['seed'], config['sampling_policy_id'], task, str(1000+t), turn)
                      for t in range(5) for turn in range(2)]
        if any(sampled_at(i, .2) for i in identities) and any(
                sampled_at(i, .4) and not sampled_at(i, .2) for i in identities):
            return task
    raise ValueError('No diagnostic identity exercises both sampling branches')


async def generate_fixture(args, rollout_id, data_source):
    import torch
    from transformers import AutoProcessor
    from playwright.async_api import async_playwright
    from slime.utils.types import Sample
    from slime.rollout.base_types import RolloutFnTrainOutput
    from openwebrl import arm_turn_bonus as bonus
    from openwebrl import arm_failure_additive as additive
    from openwebrl import arm_failure_coverage as coverage
    from openwebrl.generate_browser import _run_inference_step, _encode_with_processor
    from openwebrl.arm_inference import request_selection_result, parse_selection, split_response

    current = bonus.state(args)
    config = current['config']
    if (os.getenv('ARM_FAILURE_GPU_DIAGNOSTIC') != '1' or not config.get('diagnostic_only')
            or 'gpu-diagnostic' not in config['run_id'] or rollout_id != 0):
        raise ValueError('Fixture is restricted to one isolated diagnostic collection')
    root = Path(config['output'])
    # Exercise a local browser and retain screenshots/action/verdict for replay.
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=['--no-sandbox'])
        try:
            page = await browser.new_page(viewport={'width':448, 'height':224})
            await page.set_content('<html><body><h2>Diagnostic buttons</h2>' + ''.join(
                f'<button onclick="document.body.dataset.selected={i}">Button {i}</button>'
                for i in range(1,6)) + '</body></html>')
            screenshot = await page.screenshot(path=str(root/'browser-before.png'))
            await page.get_by_role('button', name='Button 1', exact=True).click()
            selected = await page.locator('body').get_attribute('data-selected')
            await page.screenshot(path=str(root/'browser-after.png'))
            if selected != '1': raise ValueError('Local browser action did not execute')
            bonus.write_json(root/'browser-replay.json', dict(diagnostic_only=True,
                action={'button':1}, observed_selected=selected, local_assertion_passed=True,
                screenshots=['browser-before.png','browser-after.png'],
                judge='local deterministic assertion; not an OM2W judge verdict'))
        finally:
            await browser.close()

    processor = AutoProcessor.from_pretrained(args.hf_checkpoint)
    tokenizer = processor.tokenizer
    tool = dict(type='function', function=dict(name='click', description='Click numbered button',
        parameters=dict(type='object', properties={'x':{'type':'integer'}}, required=['x'])))
    task = 'Click Button 1. x is the button number. Use a click tool call.'
    schema = '<tools>\n'+json.dumps(tool)+'\n</tools>\n'
    prompt = tokenizer.apply_chat_template([
        dict(role='system', content='You operate a browser. '+schema),
        dict(role='user', content='<|vision_start|><|image_pad|><|vision_end|>\n'+task)],
        tokenize=False, add_generation_prompt=True)
    images = ['data:image/png;base64,'+base64.b64encode(screenshot).decode()]
    prompt_ids, _, _, multimodal = _encode_with_processor(processor, tokenizer, prompt, images)
    prompt_ids = list(map(int, prompt_ids))
    if not multimodal or not all(torch.isfinite(t).all() for t in multimodal.values()):
        raise ValueError('Missing/nonfinite multimodal training tensors')
    params = dict(max_new_tokens=128, temperature=.8, top_p=.9)
    url = f'http://{args.sglang_router_ip}:{args.sglang_router_port}/generate'
    actual = []
    for index in range(5):
        response = await _run_inference_step(url, prompt, dict(params,sampling_seed=42+index),
                                            images, timeout_secs=120)
        text, ids, logps, finish = response
        if not ids or len(ids)!=len(logps) or any(not math.isfinite(x) for x in logps):
            raise ValueError('Actor returned missing or nonfinite token/log-probability pairs')
        actual.append(response)
    # Even a duplicate panel is useful for an end-to-end model API smoke test;
    # save gate rejection separately and never call that panel eligible.
    try:
        natural_actions = bonus.eligible_actions(prompt, actual)
        natural_gate = dict(eligible=True, actions=natural_actions)
    except ValueError as exc:
        natural_gate = dict(eligible=False, reason=str(exc))
    selection = await request_selection_result(config['selector_endpoint'], dict(mode='selection',
        task=task, url='about:blank', history=[], candidates=[split_response(r[0]) for r in actual],
        screenshot=base64.b64encode(screenshot).decode()), 120, None)
    actual_index = parse_selection(selection['raw'],5)
    bonus.write_json(root/'actual-model-panel.json', dict(diagnostic_only=True, prompt=prompt,
        candidates=[dict(text=r[0],token_ids=r[1],logprobs=r[2],finish=r[3]) for r in actual],
        gate=natural_gate, selected_index=actual_index, raw_selection=selection['raw']))

    # Construct valid, distinct actions to guarantee coverage of the auxiliary
    # path. ARM labels remain real; outcomes and candidate text are synthetic.
    controlled = []
    for i in range(1,6):
        response = '<think>Inspect button '+str(i)+'.</think><tool_call>'+json.dumps(
            dict(name='click',arguments={'x':i}))+'</tool_call><|im_end|>\n'
        ids = tokenizer.encode(response, add_special_tokens=False)
        controlled.append((response, ids, [0.]*len(ids), 'stop'))
    bonus.eligible_actions(prompt,controlled)
    task_id = diagnostic_task(config)
    group = []
    for trajectory in range(5):
        parent = 1000+trajectory
        reservoir = coverage.TurnReservoir(config,parent)
        rows, records = [], []
        for turn in range(2):
            sample_index = (parent<<16)|turn
            identity = (config['seed'],config['sampling_policy_id'],task_id,str(parent),turn)
            record = dict(policy_id=config['policy_id'], trajectory_id=str(parent),task_id=task_id,
                turn=turn, context=dict(parent_sample_index=sample_index,group_index=48),
                executed_index=0,eligible=False,candidate_requests=0,candidate_response_tokens=0,
                selector_requests=0,sampled=coverage.sampled_at(identity,.2),reason='not_sampled')
            captured = dict(prompt=prompt,params=params,images=images,
                observation=dict(screenshot=screenshot,active_tab_url='about:blank'),history=[],
                task=task, original=controlled[0])
            reservoir.capture(identity,record,captured,120)
            if record['sampled']:
                sequence = iter(controlled[1:])
                async def controlled_infer(*unused, **kwargs): return next(sequence)
                await bonus.ShadowSelector(config,parent)._label(
                    controlled_infer,url,captured,identity,record,120)
                if not record.get('eligible'): raise ValueError('Controlled live ARM panel was rejected: '+str(record))
            current['records'].append(record)
            rows.append(Sample(index=sample_index,group_index=48,tokens=prompt_ids+controlled[0][1],
                response=controlled[0][0],response_length=len(controlled[0][1]),
                rollout_log_probs=controlled[0][2],loss_mask=[1]*len(controlled[0][1]),
                multimodal_train_inputs=multimodal,status=Sample.Status.FAILED,reward=0.,
                metadata=dict(task_id=task_id,trajectory_id=parent,turn_index=turn,num_turns_in_trajectory=2,
                    arm_turn_bonus=deepcopy(record),diagnostic_synthetic=True,
                    terminate_reason='max_steps_exhausted',
                    reward=dict(combined=0,judge_prompt_variant='action_history',diagnostic_synthetic=True))))
            records.append(record)
        reservoir.persist()
        for sample,record in zip(rows,records):
            if record.get('failure_coverage_state'):
                sample.metadata['arm_failure_coverage_state']=record['failure_coverage_state']
        group.append(rows)
    if additive.filter_groups(args,group).keep or len(current.get('failure_additive_pool',{}))!=1:
        raise ValueError('Diagnostic must retain one auxiliary group and zero ordinary slots')
    # Per-panel candidate calls use deterministic seed -> candidate index.
    seed_to_output={}
    for trajectory in group:
        for sample in trajectory:
            identity=(config['seed'],config['sampling_policy_id'],task_id,str(sample.metadata['trajectory_id']),sample.metadata['turn_index'])
            for i in range(1,5):seed_to_output[bonus.stable_seed(*identity,i)]=controlled[i]
    async def deferred_infer(url,prompt,params,images,timeout_secs=None):
        return seed_to_output[params['sampling_seed']]
    await coverage.finalize(args,rollout_id,infer=deferred_infer)
    report=current['failure_coverage_report']
    if (report['failure_coverage']['labels']<1 or report['new_selector_requests']<1
            or report['reused']<1):
        raise ValueError('Coverage diagnostic did not exercise both reused and deferred live ARM labels')

    data_source.get_samples(48)
    outcome_rows=[]
    for g in range(48):
        for trajectory in range(5):
            text,ids,logps,finish=actual[trajectory]
            for turn in range(2):
                label={} if turn else dict(policy_id=config['policy_id'],eligible=True,
                    executed_index=0,selected_index=1,unit_bonus=-.2,reason='diagnostic_synthetic',
                    diagnostic_synthetic=True)
                outcome_rows.append(Sample(index=((g*5+trajectory)<<16)|turn,group_index=g,
                    tokens=prompt_ids+ids,response=text,response_length=len(ids),loss_mask=[1]*len(ids),
                    rollout_log_probs=logps,multimodal_train_inputs=multimodal,
                    status=Sample.Status.COMPLETED,reward=float(trajectory<2),
                    metadata=dict(task_id=f'diagnostic-mixed-{g}',trajectory_id=g*5+trajectory,
                        turn_index=turn,num_turns_in_trajectory=2,arm_turn_bonus=label,diagnostic_synthetic=True)))
    bonus.write_json(root/'fixture-provenance.json',dict(diagnostic_only=True,
        mixed_rows=len(outcome_rows),mixed_groups=48,failure_groups=1,
        mixed_outcomes_and_labels='constructed, not judged',
        auxiliary_candidates='constructed valid distinct actions; actual ARM selections',
        actual_actor_responses=len(actual),actual_arm_selection=True,multimodal=True,
        natural_panel_gate=natural_gate,
        limitation='No natural failure yield, benchmark success, or eight-GPU scalability claim'))
    return RolloutFnTrainOutput(samples=outcome_rows,metrics={'diagnostic/synthetic_fixture':1})


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation: raise ValueError('Diagnostic fixture cannot supply benchmark evaluation')
    return asyncio.run(generate_fixture(args,rollout_id,data_source))
