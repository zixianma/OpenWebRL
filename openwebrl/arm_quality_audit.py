"""Offline, fixed-state actor/SelectionARM comparison; zero browser or Adam steps."""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import random

from openwebrl.arm_inference import candidate_seed, parse_selection, request_selection_result, split_response


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def validate_inputs(config):
    path = Path(config['states'])
    raw = path.read_bytes()
    if digest(raw) != config['states_sha256']:
        raise ValueError('Fixed state panel changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 64 or len({r['id'] for r in rows}) != 64 or len({r['task_id'] for r in rows}) != 64:
        raise ValueError('Expected 64 distinct states and tasks')
    for row in rows:
        if digest(row['actor_prompt'].encode()) != row['prompt_sha256']:
            raise ValueError('Actor prompt changed')
        for im in row['actor_images'] + [dict(path=row['screenshot'], sha256=row['screenshot_sha256'])]:
            if digest(Path(im['path']).read_bytes()) != im['sha256']:
                raise ValueError('Archived image changed')
    return rows


def permutation(state_id, purpose, seed=20260921):
    order = list(range(5))
    random.Random(digest(f'{seed}:{purpose}:{state_id}'.encode())).shuffle(order)
    return order


async def collect(args, config):
    from openwebrl.generate_browser import _run_inference_step
    from slime.rollout.base_types import RolloutFnEvalOutput
    rows = validate_inputs(config)
    output = Path(config['output'])
    records = output/'records'
    records.mkdir(parents=True, exist_ok=False)
    gate = asyncio.Semaphore(4)
    url = f'http://{args.sglang_router_ip}:{args.sglang_router_port}/generate'
    sampling = dict(temperature=.8, top_p=1., top_k=-1, max_new_tokens=1024,
                    skip_special_tokens=False)
    # Mirror the resolved native generation stop controls, without a browser.
    from slime.rollout.sglang_rollout import GenerateState
    native = GenerateState(args).sampling_params
    for key in ('stop', 'stop_token_ids', 'no_stop_trim', 'spaces_between_special_tokens'):
        if key in native:
            sampling[key] = native[key]

    async def one(row):
        async with gate:
            images = ['data:image/png;base64,' + base64.b64encode(Path(i['path']).read_bytes()).decode()
                      for i in row['actor_images']]
            seeds = [candidate_seed(20260921, row['id'], row['turn'], i) for i in range(5)]
            outputs = await asyncio.gather(*[
                _run_inference_step(url, row['actor_prompt'], dict(sampling, sampling_seed=seed), images,
                                    timeout_secs=240) for seed in seeds])
            candidates = [split_response(o[0]) for o in outputs]
            order = permutation(row['id'], 'selector')
            record = dict(id=row['id'], task_id=row['task_id'], actor=config['actor'],
                          checkpoint=config['checkpoint'], candidates=candidates,
                          responses=[o[0] for o in outputs], candidate_seeds=seeds,
                          response_tokens=[len(o[1]) for o in outputs], finish_types=[o[3] for o in outputs],
                          prompt_sha256=row['prompt_sha256'], screenshot_sha256=row['screenshot_sha256'],
                          permutation=order, selected_index=None, selector_error=None)
            # Save candidates even if the selector later fails. No silent fallback.
            atomic_json(records/(row['id']+'.json'), record)
            payload = dict(mode='selection', task=row['task'], url=row['url'], history=row['history'],
                           candidates=[candidates[i] for i in order],
                           screenshot=base64.b64encode(Path(row['screenshot']).read_bytes()).decode())
            try:
                result = await request_selection_result(config['selector_endpoint'], payload, 180)
                index = parse_selection(result['raw'], 5)
                record.update(selected_index=order[index], selector_raw=result['raw'])
            except Exception as exc:
                record['selector_error'] = type(exc).__name__
                atomic_json(records/(row['id']+'.json'), record)
                raise
            atomic_json(records/(row['id']+'.json'), record)
            done = len(list(records.glob('*.json')))
            print(f'[ARMQuality] actor={config["actor"]} saved={done}/64', flush=True)
            return record

    # Test the exact path on two predeclared states before opening concurrency.
    first = [await one(row) for row in rows[:2]]
    all_records = first + await asyncio.gather(*(one(row) for row in rows[2:]))
    atomic_json(output/'collection-complete.json', dict(complete=True, actor=config['actor'],
        states=len(all_records), candidates=sum(len(r['candidates']) for r in all_records),
        output_tokens=sum(sum(r['response_tokens']) for r in all_records),
        optimizer_updates=0, browsers=0))
    return RolloutFnEvalOutput(data={}, metrics={'arm_quality/states': len(all_records),
        'arm_quality/output_tokens': sum(sum(r['response_tokens']) for r in all_records)})


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if not evaluation or args.num_rollout != 0:
        raise ValueError('Quality audit requires evaluation-only mode')
    from slime.utils.async_utils import run
    config = json.loads(Path(os.environ['OPENWEBRL_ARM_QUALITY_CONFIG']).read_text())
    return run(collect(args, config))
