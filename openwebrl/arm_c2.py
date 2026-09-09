"""Full-pool ARM winner collection and successful-trajectory SFT data building."""
import argparse
import asyncio
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time
import unicodedata
from urllib.parse import urlsplit

from openwebrl.artifact_io import run_artifact_io


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def task_key(task_id):
    return sha(str(task_id).encode())[:20]


def normalized(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def hostname(url):
    return (urlsplit(url).hostname or '').lower().removeprefix('www.')


def prepare(source, evaluation, output):
    import pandas as pd
    source, evaluation, output = Path(source), Path(evaluation), Path(output)
    df = pd.read_parquet(source)
    evaluation_rows = [json.loads(s) for s in evaluation.read_text().splitlines() if s.strip()]
    eval_ids, eval_intents, by_host = set(), set(), defaultdict(list)
    for row in evaluation_rows:
        meta = row.get('metadata', {})
        eval_ids.add(meta.get('task_id', row.get('id')))
        intent = normalized(meta.get('intent', row.get('ques', '')))
        eval_intents.add(intent)
        by_host[hostname(meta.get('start_url', row.get('web', '')))].append(intent)
    seen, groups, excluded = set(), defaultdict(list), []
    for index, meta in enumerate(df['metadata']):
        task_id, intent, url = str(meta['task_id']), str(meta['task']), str(meta['start_url'])
        norm, host = normalized(intent), hostname(url)
        reason = None
        if task_id in eval_ids or norm in eval_intents:
            reason = 'exact_evaluation_overlap'
        for other in by_host[host]:
            a, b = set(norm.split()), set(other.split())
            if (len(a & b) / max(1, len(a | b)) >= .6
                    or difflib.SequenceMatcher(None, norm, other, autojunk=False).ratio() >= .85):
                reason = 'same_host_near_evaluation_overlap'; break
        if (host, norm) in seen:
            reason = 'duplicate_host_instruction'
        if reason:
            excluded.append(dict(source_index=index, task_id=task_id, reason=reason)); continue
        seen.add((host, norm))
        groups[host].append(dict(task_id=task_id, metadata=dict(task_id=task_id, intent=intent, start_url=url,
                            source='webgym_filtered_popular_2102_cleaned', source_index=index, split='c2_full_train')))
    order_key = lambda text: sha(('arm-c2-full-v1:42:' + text).encode())
    hosts = sorted(groups, key=order_key)
    for rows in groups.values(): rows.sort(key=lambda r: order_key(r['task_id']))
    tasks = []
    for index in range(max(map(len, groups.values()), default=0)):
        tasks.extend(groups[h][index] for h in hosts if index < len(groups[h]))
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate training task IDs')
    output.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks)
    target = output / 'tasks.jsonl'
    if target.exists() and target.read_text() != content:
        raise ValueError('Refuse to replace a different frozen task inventory')
    target.write_text(content)
    audit = dict(source=str(source.resolve()), source_sha256=sha(source.read_bytes()), source_rows=len(df),
                 eval_source=str(evaluation.resolve()), eval_sha256=sha(evaluation.read_bytes()),
                 task_count=len(tasks), excluded=excluded, unique_hosts=len(hosts), task_file_sha256=sha(target.read_bytes()),
                 split='All deduplicated tasks are training tasks. Earlier unexecuted draft holdouts are superseded by explicit user request.',
                 task_order='Seed 42 hash ordering within host and over hosts, then round-robin over all hosts.',
                 checkpoint_selection='Fixed epoch-2 checkpoint; no selection using Online-Mind2Web.',
                 limitations='Heuristic instruction overlap screen, not proof of semantic or historical training disjointness.')
    write_json(output / 'data-audit.json', audit)
    return audit


class TurnExporter:
    """Write pre-action data separately from the eventual trajectory label."""
    def __init__(self, root):
        self.root = Path(root)
        self.attempts = {}
        self._image_locks = {}
        self._image_locks_guard = threading.Lock()
        (self.root / 'images').mkdir(parents=True, exist_ok=True)

    def begin(self, task_id):
        task_root = self.root / 'attempts' / task_key(task_id)
        task_root.mkdir(parents=True, exist_ok=True)
        attempt = task_root / f'attempt-{len(list(task_root.glob("attempt-*"))) + 1:04d}'
        attempt.mkdir()
        self.attempts[task_id] = attempt
        write_json(attempt / 'attempt.json', dict(task_id=task_id, started_utc=datetime.now(timezone.utc).isoformat()))
        return attempt

    def __call__(self, *, record, prompt, images, outputs):
        attempt = self.attempts[record['task_id']]
        paths = []
        for data in images:
            raw = base64.b64decode(data.split(',', 1)[-1], validate=True)
            digest = sha(raw)
            image = self.root / 'images' / (digest + '.img')
            with self._image_locks_guard:
                image_lock = self._image_locks.setdefault(digest, threading.Lock())
            with image_lock:
                if image.exists():
                    if sha(image.read_bytes()) != digest: raise ValueError('Image content-address collision')
                else:
                    image.write_bytes(raw)
            paths.append(dict(path=str(image.resolve()), sha256=digest))
        candidates = [dict(response=o[0], response_token_ids=list(o[1]), finish_type=o[3]) for o in outputs]
        value = dict(record, format_version=1, attempt=str(attempt.resolve()), prompt=prompt,
                     images=paths, raw_candidates=candidates,
                     response_token_provenance='SGLang token IDs after the existing generator appends canonical im_end/newline; no retokenized response target.')
        path = attempt / f'turn-{record["turn"]:04d}.json'
        if path.exists(): raise ValueError('Refuse to overwrite a captured turn')
        write_json(path, value)

    def finish(self, task_id, samples, outcome):
        attempt = self.attempts[task_id]
        turns = []
        for sample in samples:
            meta = sample.metadata or {}
            if 'turn_index' not in meta: continue
            path = attempt / f'turn-{meta["turn_index"]:04d}.json'
            if not path.exists(): raise ValueError('Missing exact pre-action record')
            captured = json.loads(path.read_text())
            tokens = sample.tokens.tolist() if hasattr(sample.tokens, 'tolist') else list(sample.tokens)
            n = sample.response_length
            chosen = captured['raw_candidates'][captured['selected_index']]
            if n <= 0 or tokens[-n:] != chosen['response_token_ids'] or sample.response != chosen['response']:
                raise ValueError('Selected response/tokens differ from the executed turn')
            if sample.prompt != captured['prompt']:
                raise ValueError('Pre-action prompt differs from executed turn')
            turns.append(dict(turn=meta['turn_index'], source=str(path.resolve()), source_sha256=sha(path.read_bytes()),
                              prompt_token_ids=tokens[:-n], image_grid_thw=meta['image_grid_thw'],
                              executed='step_tool_responses' in meta,
                              tool_responses=meta.get('step_tool_responses', []), status=str(sample.status)))
        write_json(attempt / 'execution.json', dict(task_id=task_id, turns=turns))
        value = dict(outcome, attempt=str(attempt.resolve()), turn_count=len(turns), completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(attempt / 'outcome.json', value)
        write_json(self.root / 'results' / (task_key(task_id) + '.json'), value)
        return value


def eligible_turn(captured, execution, max_tokens=32768):
    selected = captured.get('selected_index')
    candidates = captured.get('raw_candidates', [])
    if type(selected) is not int or not 0 <= selected < len(candidates): return 'invalid_selected_index'
    if captured.get('fallback'): return 'selector_fallback'
    scores = captured.get('scores')
    if captured['mode'] == 'scalar' and (len(scores or []) != len(candidates) or not all(math.isfinite(s) for s in scores)):
        return 'invalid_scores'
    chosen = candidates[selected]
    if chosen['finish_type'] != 'stop': return 'truncated_or_aborted_target'
    if not execution['executed']: return 'not_executed'
    if not chosen['response_token_ids'] or not execution['prompt_token_ids']: return 'empty_tokens'
    if len(execution['prompt_token_ids']) + len(chosen['response_token_ids']) > max_tokens: return 'oversize'
    if sha(captured['prompt'].encode()) != captured['prompt_sha256']: raise ValueError('Prompt hash mismatch')
    for image in captured['images']:
        if sha(Path(image['path']).read_bytes()) != image['sha256']: raise ValueError('Image hash mismatch')
    return None


def build_dataset(root, require_complete=True):
    root = Path(root)
    config = json.loads((root / 'frozen-config.json').read_text())
    tasks = [json.loads(s) for s in Path(config['task_file']).read_text().splitlines()]
    ids = {t['task_id'] for t in tasks}
    results = [json.loads(p.read_text()) for p in sorted((root / 'results').glob('*.json'))]
    if len({r['task_id'] for r in results}) != len(results) or not {r['task_id'] for r in results} <= ids:
        raise ValueError('Unexpected or duplicate collection outcomes')
    if require_complete and len(results) != len(tasks): raise ValueError('Full-pool collection is incomplete')
    reasons, records, successful_tasks = Counter(), [], set()
    for result in results:
        if not result.get('valid') or result.get('reward') != 1:
            reasons['trajectory_unsuccessful_or_unavailable'] += 1; continue
        execution = json.loads((Path(result['attempt']) / 'execution.json').read_text())
        if execution['task_id'] != result['task_id']: raise ValueError('Cross-task execution join')
        for turn in execution['turns']:
            source = Path(turn['source'])
            if sha(source.read_bytes()) != turn['source_sha256']: raise ValueError('Captured turn changed')
            captured = json.loads(source.read_text())
            if captured['task_id'] != result['task_id'] or captured['turn'] != turn['turn'] or captured['attempt'] != result['attempt']:
                raise ValueError('Cross-task or cross-attempt training join')
            reason = eligible_turn(captured, turn, config['training']['max_tokens'])
            if reason:
                reasons[reason] += 1; continue
            records.append(dict(task_id=result['task_id'], turn=turn['turn'], source=str(source),
                                source_sha256=turn['source_sha256'], prompt_token_ids=turn['prompt_token_ids'],
                                image_grid_thw=turn['image_grid_thw'], selected_index=captured['selected_index']))
            successful_tasks.add(result['task_id'])
    dataset = root / ('training.jsonl' if require_complete else 'training-preview.jsonl')
    content = ''.join(json.dumps(r) + '\n' for r in records)
    if require_complete and dataset.exists() and dataset.read_text() != content:
        raise ValueError('Refuse to replace a different complete training dataset')
    dataset.write_text(content)
    summary = dict(complete_collection=len(results) == len(tasks), scheduled=len(tasks), attempted=len(results),
                   successful_tasks_with_usable_turns=len(successful_tasks), retained_turns=len(records), exclusions=dict(reasons),
                   dataset=str(dataset), dataset_sha256=sha(dataset.read_bytes()), filter='C2: valid successful trajectory + usable executed winner; no confidence cutoff')
    write_json(root / ('dataset-audit.json' if require_complete else 'dataset-preview-audit.json'), summary)
    return summary


def load_training_example(row, processor):
    """Reproduce the actual multimodal prefix; supervise only current response IDs."""
    from PIL import Image
    import torch
    from slime.utils.processing_utils import build_processor_kwargs
    source = Path(row['source'])
    if sha(source.read_bytes()) != row['source_sha256']: raise ValueError('Training source hash mismatch')
    captured = json.loads(source.read_text())
    images = []
    for item in captured['images']:
        path = Path(item['path'])
        if sha(path.read_bytes()) != item['sha256']: raise ValueError('Training image hash mismatch')
        with Image.open(path) as image: images.append(image.convert('RGB'))
    features = processor(text=[captured['prompt']], **build_processor_kwargs({'images': images}))
    ids = features['input_ids'][0]
    if hasattr(ids, 'tolist'): ids = ids.tolist()
    if ids != row['prompt_token_ids']: raise ValueError('Processor-expanded prompt token mismatch')
    if features['image_grid_thw'].tolist() != row['image_grid_thw']: raise ValueError('Image grid mismatch')
    target = captured['raw_candidates'][row['selected_index']]['response_token_ids']
    input_ids = torch.tensor([ids + target], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[0, len(ids):] = torch.tensor(target, dtype=torch.long)
    tensors = {k: v for k, v in features.items() if k not in ('input_ids', 'attention_mask') and isinstance(v, torch.Tensor)}
    return dict(input_ids=input_ids, labels=labels, attention_mask=torch.ones_like(input_ids), **tensors)


async def collect(config_path):
    import httpx
    from dotenv import load_dotenv
    from openwebrl import generate_browser as generation
    from openwebrl import run_evaluate as evaluation
    from openwebrl.arm_inference import ActionSelector
    from openwebrl.arm_eval import summarize
    from slime.utils.http_utils import init_http_client
    from slime.utils.types import Sample
    q = json.loads(Path(config_path).read_text())
    root = Path(q['output'])
    load_dotenv('.env', override=False)
    os.environ.setdefault('JUDGE_API_BASE', 'https://api.openai.com/v1')
    os.environ.update(SLIME_BROWSER_ENV_MODE='local_process',
                     SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(root / 'browser_logs'),
                     SLIME_BROWSER_LOCAL_PROCESS_PORT_START='19200', SLIME_BROWSER_LOCAL_PROCESS_PORT_END='19399',
                     SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=str(q['parallel']))
    generation._BROWSER_HOST_BLACKLIST_PATH = str(root / 'navigation_failures.txt')
    manifest = q['teacher_manifest']
    args = evaluation.EvalArgs(sglang_router_ip='127.0.0.1', sglang_router_port=19100,
        hf_checkpoint=manifest['actor'], max_steps=30, context_num_screenshots=1,
        judge_api_model='o4-mini', judge_api_mode='served', judge_timeout_secs=180,
        browser_response_format_mode='browser_env', turn_history_reasoning_mode='full',
        browser_include_tool_response=1, inference_step_timeout_secs=180, task_timeout_secs=1800,
        path_to_save_generated_samples=str(root / 'samples'))
    sampling = dict(temperature=.7, top_p=.9, max_new_tokens=1024)
    exporter = TurnExporter(root)
    args.browser_action_selector = ActionSelector(q['teacher'], 'http://127.0.0.1:19101', root / 'selections', seed=q['seed'], exporter=exporter)
    args.rollout_temperature, args.rollout_max_response_len = .7, 1024
    args.browser_async_artifact_io = True
    init_http_client(args)
    reward_func = evaluation._load_reward_func('online_mind2web')
    tasks = evaluation.load_tasks_from_jsonl(q['task_file'])
    if len(tasks) != q['task_count'] or sha(Path(q['task_file']).read_bytes()) != q['task_file_sha256']:
        raise ValueError('Collection task inventory changed')
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        info = await client.get('http://127.0.0.1:19100/get_model_info'); info.raise_for_status()
        if os.path.realpath(info.json()['model_path']) != os.path.realpath(manifest['actor']): raise ValueError('Wrong actor')
        info = await client.get('http://127.0.0.1:19101/health'); info.raise_for_status()
        if info.json() != manifest['selector_health']: raise ValueError('Wrong teacher')
    (root / 'results').mkdir(exist_ok=True)
    deadline = datetime.fromisoformat(q['stop_utc']).timestamp()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT): loop.add_signal_handler(sig, stop.set)
    remaining = [task for task in tasks if not (root / 'results' / (task_key(task['task_id']) + '.json')).exists()]
    completed_records = {r['task_id']: r for r in
                         (json.loads(p.read_text()) for p in (root / 'results').glob('*.json'))}
    summary_lock = asyncio.Lock()
    pending = asyncio.Queue()
    for task in remaining: pending.put_nowait(task)
    async def one(task):
        task_id = task['task_id']
        attempt = await run_artifact_io(exporter.begin, task_id)
        print('START', task_id, str(attempt), flush=True)
        samples = []
        async def trajectory():
            nonlocal samples
            original = Sample(index=task['index'], prompt=task['prompt'], metadata=dict(task_id=task_id,
                              start_url=task['start_url'], intent=task['prompt']))
            samples = await generation.generate_turn_sample(args, original, sampling)
            reward = await reward_func(args, samples)
            if isinstance(reward, list): reward = reward[-1] if reward else None
            last = samples[-1]
            valid = reward is not None and 'ABORTED' not in str(last.status)
            valid = valid and not last.metadata.get('judge_invalid') and not last.metadata.get('judge_timeout')
            return dict(task_id=task_id, reward=reward, valid=bool(valid), status=str(last.status),
                        terminate_reason=last.metadata.get('terminate_reason'), total_steps=last.metadata.get('total_steps'))
        try:
            outcome = await asyncio.wait_for(trajectory(), timeout=args.task_timeout_secs)
        except asyncio.CancelledError:
            await run_artifact_io(write_json, attempt / 'interrupted.json',
                                 dict(task_id=task_id, reason='allocation_or_controller_stop',
                                      utc=datetime.now(timezone.utc).isoformat()))
            raise
        except Exception as exc:
            outcome = dict(task_id=task_id, reward=None, valid=False, error_type=type(exc).__name__, error=str(exc))
        result = await run_artifact_io(exporter.finish, task_id, samples, outcome)
        print('DONE', task_id, 'reward', result.get('reward'), 'valid', result['valid'], 'turns', result['turn_count'], flush=True)
        async with summary_lock:
            completed_records[task_id] = result
            await run_artifact_io(write_json, root / 'collection-summary.json',
                                  summarize(list(completed_records.values()), len(tasks)))
    # Validate a small prefix of the same collection before dispatching the
    # full queue. Its completed outcomes remain part of the one-attempt dataset.
    if not (root / 'export-smoke-passed.json').exists():
        smoke_tasks = [pending.get_nowait() for _ in range(min(8, pending.qsize()))]
        smoke_work = asyncio.gather(*(one(task) for task in smoke_tasks))
        while not smoke_work.done():
            if stop.is_set() or time.time() >= deadline - 30:
                smoke_work.cancel()
                break
            await asyncio.sleep(2)
        try:
            await smoke_work
        except asyncio.CancelledError:
            return
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(manifest['actor'], local_files_only=True)
        checked = 0
        for result_path in sorted((root / 'results').glob('*.json')):
            outcome = json.loads(result_path.read_text())
            execution = json.loads((Path(outcome['attempt']) / 'execution.json').read_text())
            for turn in execution['turns']:
                captured = json.loads(Path(turn['source']).read_text())
                if not turn['executed'] or captured['raw_candidates'][captured['selected_index']]['finish_type'] != 'stop':
                    continue
                row = dict(turn, selected_index=captured['selected_index'])
                load_training_example(row, processor)
                checked += 1
                if checked >= 8: break
            if checked >= 8: break
        if not checked:
            raise ValueError('First collection batch produced no executed turn to validate; inspect before expanding')
        write_json(root / 'export-smoke-passed.json', dict(checked_turns=checked, processor_prefix_and_image_grid_match=True,
                    successful_trajectory_filter='Not applied to this engineering check; applied by dataset builder'))
        del processor
    async def worker():
        while not stop.is_set() and time.time() < deadline - 180 and not pending.empty():
            task = pending.get_nowait()
            await one(task)
    workers = [asyncio.create_task(worker()) for _ in range(q['parallel'])]
    joined = asyncio.gather(*workers)
    while not joined.done():
        if stop.is_set() or time.time() >= deadline - 30:
            for worker_task in workers: worker_task.cancel()
            break
        await asyncio.sleep(2)
    try:
        await joined
    except asyncio.CancelledError:
        pass
    finally:
        for worker_task in workers:
            if not worker_task.done(): worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        records = [json.loads(p.read_text()) for p in (root / 'results').glob('*.json')]
        write_json(root / 'collection-summary.json', summarize(records, len(tasks)))
        if records:
            print('DATASET_AUDIT', json.dumps(build_dataset(root, require_complete=len(records) == len(tasks))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', default='openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet')
    prep.add_argument('--evaluation', default='openwebrl/data/eval/online-mind2web.jsonl')
    prep.add_argument('--output', required=True)
    coll = sub.add_parser('collect'); coll.add_argument('--config', required=True)
    build = sub.add_parser('build'); build.add_argument('--output', required=True); build.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': print(json.dumps(prepare(args.source, args.evaluation, args.output), indent=2))
    elif args.command == 'build': print(json.dumps(build_dataset(args.output, not args.allow_partial), indent=2))
    else:
        from openwebrl import run_evaluate  # Import before asyncio creates its loop.
        asyncio.run(collect(args.config))


if __name__ == '__main__':
    main()
