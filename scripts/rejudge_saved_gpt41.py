#!/usr/bin/env python3
"""Rejudge a trusted local rollout archive with the preserved GPT-4.1 protocol.

Reads only pickle metadata, skipping tensor storage. No GPUs or browsers.
Default is a CPU-only input audit; --execute makes authorized judge API calls.
"""
import argparse
import asyncio
import collections
import contextvars
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace
import zipfile


def skip_tensor(*args, **kwargs):
    return None


class MetadataOnly(pickle.Unpickler):
    def persistent_load(self, pid):
        return None

    def find_class(self, module, name):
        if module.startswith('torch'):
            return skip_tensor
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        raise ValueError(f'Unsupported metadata global: {module}.{name}')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def summarize(records, planned):
    valid = [r for r in records if not r['invalid']]
    successes = sum(r['reward'] == 1 for r in valid)
    return {'completed': len(records), 'planned': planned, 'successes': successes,
            'valid': len(valid), 'invalid': len(records) - len(valid),
            'success_rate_all_planned': successes / planned,
            'success_rate_valid': successes / len(valid) if valid else 0,
            'original_invalid': sum(r['original_invalid'] for r in records),
            'judge_timeouts': sum(r.get('judge_timeout', False) for r in records)}


async def main(a):
    from dotenv import dotenv_values
    for key, value in dotenv_values(a.env_file).items():
        if value and key.startswith(('OPENAI_', 'JUDGE_', 'AZURE_', 'WANDB_')):
            os.environ.setdefault(key, value)
    sys.path.insert(0, str(a.source.resolve()))
    from openwebrl import reward_browser as rb
    from slime.utils.types import Sample
    assert Path(rb.__file__).resolve().is_relative_to(a.source.resolve())
    archive = a.input / 'runtime/rollout_recovery/eval_0.pt'
    original = {}
    for f in (a.input / 'completed_tasks').glob('*.json'):
        record = json.loads(f.read_text())
        assert record['task_id'] not in original, 'Duplicate original task'
        original[record['task_id']] = record
    groups = collections.defaultdict(list)
    if a.rollout_dir:
        import torch
        for path in sorted(a.rollout_dir.glob('*.pt')):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            for value in payload['turns']:
                sample = Sample.from_dict(value)
                sample.multimodal_train_inputs = None
                groups[payload['task_id']].append(sample)
        archive = a.rollout_dir
    else:
        with zipfile.ZipFile(archive) as z:
            member = next(n for n in z.namelist() if n.endswith('/data.pkl'))
            with z.open(member) as stream:
                data = MetadataOnly(stream).load()
        for value in data['samples']:
            sample = Sample.from_dict(value)
            sample.multimodal_train_inputs = None
            groups[sample.metadata['task_id']].append(sample)
    if not original:
        original = {task_id: {'remove_sample': False, 'judge_invalid': False,
                              'transport_reward': None} for task_id in groups}
    assert set(groups) == set(original) and groups, 'Archive and task records must have the same nonempty task set'
    args = SimpleNamespace(judge_api_model='gpt-4.1', judge_prompt_variant='action_history',
                           judge_api_mode='served', judge_timeout_secs=120.,
                           judge_max_attached_imgs=3, context_num_screenshots=1,
                           browser_response_format_mode='browser_env',
                           turn_history_reasoning_mode='full', debug_trace_prob=0.)
    if not os.environ.get('JUDGE_API_BASE'):
        os.environ['JUDGE_API_BASE'] = os.environ.get('OPENAI_API_BASE') or 'https://api.openai.com/v1'
    a.output.mkdir(parents=True, exist_ok=True)
    result_dir = a.output / 'completed_tasks'
    result_dir.mkdir(exist_ok=True)
    manifest = {'input': str(a.input.resolve()), 'archive_bytes': archive.stat().st_size,
                'source': str(a.source.resolve()), 'judge': vars(args),
                'reward_module_sha256': hashlib.sha256(Path(rb.__file__).read_bytes()).hexdigest(),
                'task_ids': sorted(groups), 'wandb_project': 'openwebrl-evals',
                'wandb_run_id': a.wandb_id, 'actor_temperature': 0., 'checkpoint_after': 80,
                'rejudged_saved_trajectories': True, 'planned_tasks': len(groups),
                'new_gpu_hours': 0, 'new_browser_sessions': 0}
    manifest_path = a.output / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Resume manifest mismatch'
    write_json(manifest_path, manifest)
    print(json.dumps({'stage': 'inputs_validated', 'tasks': len(groups), 'execute': a.execute}), flush=True)
    if not a.execute:
        return
    import wandb
    client = rb._get_openai_client('served')
    task_context = contextvars.ContextVar('task_id')
    usage = []

    async def create(**kwargs):
        started = time.time()
        response = await client.chat.completions.create(**kwargs)
        u = response.usage.model_dump() if response.usage else {}
        record = {'task_id': task_context.get(), 'model': response.model, 'usage': u,
                  'seconds': time.time() - started}
        usage.append(record)
        with (a.output / 'api_usage.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        return response

    rb._get_openai_client = lambda method='served': SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    run = wandb.init(entity='zixianma', project='openwebrl-evals', id=a.wandb_id,
                     name=a.wandb_id, resume='allow', config=manifest, dir=str(a.output),
                     settings=wandb.Settings(x_disable_stats=True))
    records = [json.loads(f.read_text()) for f in result_dir.glob('*.json')]
    done = {r['task_id'] for r in records}
    assert len(done) == len(records) and done <= set(groups)
    gate = asyncio.Semaphore(a.concurrency)

    async def process(task_id, turns):
        if task_id in done:
            return
        async with gate:
            task_context.set(task_id)
            turns.sort(key=lambda s: s.metadata.get('turn_index', -1))
            old = original[task_id]
            invalid = (old['remove_sample'] or old.get('judge_invalid', False)
                       or any(s.remove_sample or s.status == Sample.Status.ABORTED for s in turns))
            last = next(s for s in turns if s.metadata.get('is_last_turn'))
            if invalid:
                score, metadata = None, {'judge_text': 'Original invalid attempt; not rejudged.'}
            else:
                scores = await rb.reward_func(args, turns)
                for s, score in zip(turns, scores, strict=True):
                    s.reward = score
                score, metadata = last.reward, last.metadata['reward']
            record = {'task_id': task_id, 'original_invalid': bool(invalid),
                      'invalid': bool(invalid or any(s.remove_sample for s in turns)),
                      'reward': score, 'original_reward': old['transport_reward'],
                      'judge_timeout': metadata.get('judge_timeout', False),
                      'judge_result': metadata, 'turns': len(turns)}
            write_json(result_dir / (hashlib.sha256(task_id.encode()).hexdigest() + '.json'), record)
            records.append(record)
            if len(records) % 10 == 0 or len(records) == len(groups):
                metrics = summarize(records, len(groups))
                write_json(a.output / 'metrics.json', metrics)
                run.log({f'eval/{k}': v for k, v in metrics.items()})
                print(json.dumps(metrics), flush=True)

    try:
        await asyncio.gather(*(process(task_id, turns) for task_id, turns in groups.items()))
        metrics = summarize(records, len(groups))
        assert metrics['completed'] == len(groups)
        paired = [r for r in records if not r['invalid']]
        metrics['gpt41_only_successes'] = sum(r['reward'] == 1 and r['original_reward'] != 1 for r in paired)
        metrics['o4_only_successes'] = sum(r['reward'] != 1 and r['original_reward'] == 1 for r in paired)
        metrics['prompt_tokens_this_process'] = sum(u['usage'].get('prompt_tokens', 0) for u in usage)
        metrics['completion_tokens_this_process'] = sum(u['usage'].get('completion_tokens', 0) for u in usage)
        write_json(a.output / 'metrics.json', metrics)
        run.log({f'eval/{k}': v for k, v in metrics.items()})
        run.summary.update({f'eval/{k}': v for k, v in metrics.items()})
        run.finish()
        write_json(a.output / 'status.json', {'complete': True, 'exit_code': 0})
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        await client.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--rollout-dir', type=Path,
                   help='Per-task .pt rollout directory produced by the evaluator')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--env-file', type=Path, default=Path(__file__).resolve().parents[1] / '.env')
    p.add_argument('--wandb-id', required=True)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--execute', action='store_true')
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(p.parse_args()))
