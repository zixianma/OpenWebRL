"""v12: one approval-bound WebArena cell with the OpenWebRL-4B-SFT actor (no_search or + Piotr SelectionARM).

Same lifecycle as v11's run_native_webarena_cell.py (lease claim, native site login, supervised child, released
evaluator, result.json), but the agent is OpenWebRL's own loop (openwebrl.webprm_owrl_v12 explains the plumbing).
Runs with the OpenWebRL venv Python inside the site's network namespace; the frozen OpenWebRL snapshot supplies the
agent stack, this package supplies the webprm_* harness modules.
"""
from __future__ import annotations

# --- import layout (before any openwebrl import): agent stack from the frozen snapshot, harness from this package ---
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/'
                'webarena-subset-review-20261021/v12-tools/owrl-snapshot')
NLTK_SITE = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/nltk-runtime/venv/lib/python3.12/site-packages'
PACKAGE_FIRST = ('webprm_', 'jev_', 'kev_eval', 'decision_selection', 'selection_budget')


class _HarnessFirst(importlib.abc.MetaPathFinder):
    """openwebrl.<webprm_*, jev_*, ...> from this package; every other openwebrl module from the snapshot."""

    def find_spec(self, name, path=None, target=None):
        parts = name.split('.')
        if len(parts) == 2 and parts[0] == 'openwebrl' and parts[1].startswith(PACKAGE_FIRST):
            return importlib.machinery.PathFinder.find_spec(name, [str(ROOT / 'openwebrl')])
        return None


def install_import_layout():
    # The launcher's environment prepends the NLTK runtime (for the evaluator) and this package; keep the venv's own
    # packages ahead of the NLTK runtime's regex/tqdm/click and put the snapshot first.
    sys.path[:] = [p for p in sys.path if p not in (NLTK_SITE, str(ROOT), str(SNAPSHOT))]
    sys.path[:0] = [str(SNAPSHOT), str(ROOT)]
    sys.path.append(NLTK_SITE)
    if not any(isinstance(f, _HarnessFirst) for f in sys.meta_path):
        sys.meta_path.insert(0, _HarnessFirst())
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers')
    os.environ.setdefault('SLIME_ASYNC_USE_STDLIB_LOOP', '1')
    os.environ.setdefault('SLIME_HF_VISION_ATTN_IMPL', 'sdpa')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                      WANDB_PROJECT='openwebrl-evals', TOKENIZERS_PARALLELISM='false')


install_import_layout()

import argparse  # noqa: E402
import asyncio  # noqa: E402
from copy import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from openwebrl.webprm_lifecycle import EPISODE_WATCHDOG, mark_episode_phase, run_supervised_episode, save_outcome_and_close  # noqa: E402
from openwebrl.webprm_native_release import reference  # noqa: E402
from openwebrl.webprm_webarena import TASK_FILE, EvaluationNotReady, digest, evaluation_requirements, verify_source  # noqa: E402
from openwebrl.webprm_webarena_episode import TaskEnvelope  # noqa: E402
from openwebrl import webprm_owrl_v12 as V12  # noqa: E402

PROTOCOL = 'native_table5_v12_owrl'
TASK_IDS = (0, 2, 4, 8, 9, 10, 13, 14, 17, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 54, 55, 56, 58, 59, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 148, 149, 150, 151, 152, 153, 154, 156, 157, 160, 162, 163, 164)  # v9/v11 128-task cohort
METHODS = ('no_search', 'piotr_selectionarm')
ACTOR = V12.V12_ACTOR


def write(path, value):
    from openwebrl.jev_eval import write_json
    write_json(path, value)
    return reference(path)


def cell_arm(actor, selector):
    if actor != ACTOR or selector not in METHODS:
        raise EvaluationNotReady('Actor or selector outside the v12 cohort')
    return actor + '__' + selector


def broker_request(config, task, selector):
    """v11 broker_request with the v12 cell name (only the native fuzzy judge crosses the broker)."""
    if not config.get('broker_socket'):
        raise EvaluationNotReady('An owned host-namespace model broker is required')
    from openwebrl.webprm_broker import make_broker_client
    client = make_broker_client(config['broker_socket'], cell_id=task.execution_id + '/' + cell_arm(config['actor_model'], selector),
                                identity_sha256=config['identity_sha256'], deadline=config['deadline'])

    def request(call_config, payload, journal, **kwargs):
        mark_episode_phase(call_config, 'judge_request' if kwargs.get('purpose') == 'native_fuzzy' else 'actor_request',
                           purpose=kwargs.get('purpose'), model=payload['model'])
        return client(call_config, payload, journal, **kwargs)
    request._webprm_broker_client = getattr(client, '_webprm_broker_client', False)
    request._webprm_budget_owner = getattr(client, '_webprm_budget_owner', None)
    return request


def verify_snapshot(config):
    sums = SNAPSHOT / 'SNAPSHOT_SHA256SUMS'
    if hashlib.sha256(sums.read_bytes()).hexdigest() != config['owrl']['snapshot_sums_sha256']:
        raise EvaluationNotReady('Frozen OpenWebRL snapshot manifest changed')
    for line in sums.read_text().splitlines():
        value, name = line.split('  ', 1)
        if hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() != value:
            raise EvaluationNotReady('Frozen OpenWebRL snapshot file changed: ' + name)


class EpisodeLoop:
    """One asyncio loop on a dedicated thread: the agent episode runs there; the evaluator calls in synchronously."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name='owrl-episode-loop', daemon=True)
        self.thread.start()

    def run(self, coroutine, timeout=None):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=30)


async def owrl_episode(*, task, method, config, folder, storage_state, mark):
    """OpenWebRL generate_turn_sample with crossbench's MeteredActor(CrossbenchSelector), in-process WebEnv."""
    from openwebrl import generate_browser as generation
    from openwebrl import crossbench_selector_eval as core
    from openwebrl.run_evaluate import EvalArgs
    from slime.utils import http_utils
    from slime.utils.types import Sample
    owrl = config['owrl']
    http_utils._http_client = V12.uds_async_client(owrl['actor_socket'])
    released_post = http_utils.post

    async def marked_post(url, payload, *args, **kwargs):
        mark('actor_request')
        return await released_post(url, payload, *args, **kwargs)
    generation.post = marked_post                     # MeteredActor resolves generation.post at call time
    env = V12.InProcessWebArenaEnv(storage_state=storage_state, mark=mark, record_dir=folder / 'owrl-browser')
    intent, start_url = task.actor_view['intent'], task.actor_view['start_url']
    resources = {}

    async def create_env(task_id, env_config, task_metadata=None, response_format_mode_name=None):
        task_data, tool_list, policy = generation._load_local_resources(
            task_id, env_config, dict(start_url=start_url, intent=intent),
            response_format_mode_name=response_format_mode_name)
        task_data = dict(task_id=task_id, start_url=start_url, intent=intent, sites=list(task.actor_view['sites']))
        resources.update(env_config={k: v for k, v in env_config.items() if k not in ('sandbox', 'browser_use')},
                         policy_sha256=hashlib.sha256(policy.encode()).hexdigest(), tool_list=tool_list)
        await env.initialize(task_id, task_data, env_config, tool_list=tool_list, policy=policy)
        return env, task_data
    generation._create_env = create_env
    args = EvalArgs(**V12.eval_args_kwargs(folder))
    args.rollout_temperature = V12.SAMPLING['temperature']
    args.rollout_max_response_len = V12.SAMPLING['max_new_tokens']
    args.browser_history_full_turns = 0
    http_utils.init_http_client(args)
    if http_utils._http_client is None or not isinstance(http_utils._http_client._transport, __import__('httpx').AsyncHTTPTransport):
        raise EvaluationNotReady('Actor transport is not the UNIX-socket client')
    kind = V12.SELECTOR_KIND[method]
    selector = core.CrossbenchSelector(kind, folder / 'selections', seed=V12.SEED, candidates=V12.CANDIDATES[method],
                                       endpoint=V12.SELECT_URL if kind == 'selection' else None,
                                       request_fn=V12.uds_selection_request(owrl['select_socket'], mark))
    system_policy = (SNAPSHOT / 'openwebrl/env/prompts/system_prompt_browser_env.md').read_text()
    metered = core.MeteredActor(selector, folder, system_policy=system_policy)
    args.browser_action_selector = metered
    sample = Sample(index=0, prompt=intent, metadata=dict(task_id=task.execution_id, start_url=start_url, intent=intent))
    sampling = dict(V12.SAMPLING, sampling_seed=V12.SEED)
    started = time.monotonic()
    samples = await asyncio.wait_for(generation.generate_turn_sample(args, sample, sampling),
                                     timeout=V12.TASK_TIMEOUT_SECONDS + 60)
    stats = dict(actor_requests=metered.requests, actor_input_tokens=metered.input_tokens,
                 actor_output_tokens=metered.output_tokens, actor_client_seconds=metered.client_seconds,
                 episode_seconds=time.monotonic() - started)
    return env, samples, stats, resources


def _run_episode(task_record, arm, config):
    task = TaskEnvelope.from_release(task_record['released'], config['endpoints'])
    method = config['selector']
    folder = Path(config['output']) / task.execution_id / arm
    result = dict(task_id=task.execution_id, arm=arm, actor_model=config['actor_model'], selector=method,
                  identity_sha256=config['identity_sha256'], attempted=False, valid=False, success=0,
                  started_at=time.time(), protocol=PROTOCOL, paper_exact_protocol=False,
                  owrl_protocol_sha256=V12.protocol_sha256())
    loop = env = None
    try:
        if config.get('authorization_status') != 'approved' or not config.get('api_usd_cap'):
            raise EvaluationNotReady('Exact positive v12 authorization required')
        if str(config.get('job_id')) != os.environ.get('SLURM_JOB_ID'):
            raise EvaluationNotReady('Worker allocation identity changed')
        for path, expected in config['code_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise EvaluationNotReady('Frozen worker dependency changed')
        if config['owrl'].get('protocol_sha256') != V12.protocol_sha256():
            raise EvaluationNotReady('v12 OpenWebRL protocol differs from the frozen plan')
        verify_snapshot(config)
        from openwebrl.webprm_webarena_worker import claim_lease, authenticated_state, episode_evaluator
        request = broker_request(config, task, method)
        instance, reset_id = claim_lease(task, method, config)
        config = dict(config, reset_id=reset_id)
        from openwebrl.webprm_runtime import prepare_browser_runtime
        runtime_root = prepare_browser_runtime()
        os.environ['HF_HOME'] = str(Path(runtime_root) / 'hf')
        result['attempted'] = True
        mark_episode_phase(config, 'native_authentication')
        from openwebrl.webprm_webarena_sites import site_authentication
        state, auth = site_authentication(task, config, instance, V12.CHROMIUM, folder,
                                          reddit_authenticate=authenticated_state)
        helpers = None
        if evaluation_requirements(task.native_config())['helper_functions_required']:
            from openwebrl.webprm_webarena_helper_runtime import owned_native_helpers
            helpers = owned_native_helpers(task, config, instance)
        evaluator = episode_evaluator(task, config, request, folder, deployment_helpers=helpers)
        mark = lambda phase: mark_episode_phase(config, phase)
        mark('owrl_startup')
        loop = EpisodeLoop()
        env, samples, stats, resources = loop.run(owrl_episode(task=task, method=method, config=config, folder=folder,
                                                               storage_state=state, mark=mark))
        ending, agent_valid, reason = V12.outcome_of(samples)
        selections = sorted((folder / 'selections').glob('*.jsonl'))
        steps = [json.loads(line) for path in selections for line in path.read_text().splitlines() if line]
        result.update(owrl_ending=ending, owrl_terminate_reason=reason, owrl_stats=stats,
                      owrl_resources=dict(resources, tool_list=None), turns=len(samples) if samples else 0,
                      executed_actions=len(env.steps), selector_calls=sum(s['kind'] == 'selection' for s in steps),
                      selector_fallbacks=sum(bool(s.get('fallback')) for s in steps),
                      selector_picks=[s['selected_index'] for s in steps],
                      decision_seconds=sum(s['timing']['decision_seconds'] for s in steps),
                      generation_seconds=sum(s['timing']['generation_seconds'] for s in steps),
                      selection_seconds=sum(s['timing']['selection_seconds'] for s in steps),
                      authentication=json.loads((folder / 'authentication.json').read_text())
                      if (folder / 'authentication.json').exists() else None,
                      authentication_policy=auth, reset_id=reset_id, original_instance=instance)
        write(folder / 'owrl-steps.json', dict(start_url=env.start_url, steps=env.steps, answer=env.answer,
                                                ending=ending, terminate_reason=reason))
        write(folder / 'rollout.json', [dict(index=s['index'], url=s['url'], calls=[a.get('name') for a in s['actions']])
                                        for s in env.steps])
        if not agent_valid:
            raise EvaluationNotReady(f'OpenWebRL infrastructure ending {ending}: {reason}'[:2000])
        done = ending == 'done'
        trajectory = V12.project_trajectory(env.start_url, env.steps, env.answer, done=done)
        write(folder / 'native-evaluator-trajectory.json', trajectory)
        mark_episode_phase(config, 'native_evaluation')
        page = V12.SyncPage(env, loop.loop)
        page.set_default_timeout(V12.EVALUATION_TIMEOUT_MS)
        verdict = evaluator.evaluate(task.native_config(), trajectory, page, termination='stop' if done else 'turn_limit')
        write(folder / 'native-evaluation.json', verdict)
        valid = bool(isinstance(verdict, dict) and verdict.get('valid'))
        result.update(valid=valid, success=int(valid and verdict['success']),
                      native_success=int(valid and verdict.get('native_success', verdict['success'])),
                      native_evaluation=verdict, terminal=dict(answer=trajectory[-1]['answer'],
                                                               source=trajectory[-1]['owrl_terminal']),
                      termination='stop' if done else 'turn_limit', evaluator_page_calls=page.calls,
                      method_interpretation='v12_owrl_' + V12.SELECTOR_KIND[method])
        if not valid:
            result.update(error_type='NativeEpisodeInvalid', error='Native verdict invalid')
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error)[:4000])
    finally:
        closure = dict(closed=False, errors=[])
        if loop is not None:
            try:
                if env is not None:
                    loop.run(env.close(), timeout=60)
                closure['closed'] = True
            except Exception as error:
                closure['errors'].append(dict(stage='browser', error_type=type(error).__name__))
            loop.stop()
        result['closure'] = closure
        if closure['errors']:
            result['valid'] = False
            result['success'] = 0
        save_outcome_and_close(result, folder, None, config)
    return result


def run_episode(task_record, selector, config):
    if selector not in METHODS or config['actor_model'] != ACTOR or task_record['released']['task_id'] not in TASK_IDS:
        raise EvaluationNotReady('Cell outside the fixed v12 cohort')
    if config.get('max_turns') != V12.MAX_STEPS:
        raise EvaluationNotReady('v12 turn cap changed')
    arm = cell_arm(config['actor_model'], selector)
    return run_supervised_episode(task_record, arm, dict(config, selector=selector, cell_name=arm), _run_episode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--actor', required=True); parser.add_argument('--selector', choices=METHODS, required=True)
    parser.add_argument('--lease', type=Path, required=True); args = parser.parse_args()
    plan = json.loads(args.plan.read_text()); identity = digest(plan)
    if not args.execute:
        raise EvaluationNotReady('v12 cell worker is execution-only')
    if (args.task_id not in TASK_IDS or plan.get('task_ids') != [args.task_id] or plan.get('actors') != [ACTOR]
            or args.actor != ACTOR or plan.get('selectors') != [args.selector]):
        raise EvaluationNotReady('Worker CLI differs from exact assigned plan')
    if verify_source(plan['arbiter_source']) != plan['source_manifest']:
        raise EvaluationNotReady('Native task/evaluator source changed')
    rows = json.loads((Path(plan['arbiter_source']) / TASK_FILE).read_text())
    raw = next(x for x in rows if x['task_id'] == args.task_id)
    if digest([raw]) != plan['tasks_sha256']:
        raise EvaluationNotReady('Released task identity changed')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    write(output / 'manifest.json', dict(identity=plan, identity_sha256=identity))
    task = TaskEnvelope.from_release(raw, plan['endpoints'])
    config = dict(plan, output=str(output), actor_model=args.actor, identity_sha256=identity,
                  episode_watchdog=dict(EPISODE_WATCHDOG), lease_path=str(args.lease.resolve()),
                  lease_sha256=hashlib.sha256(args.lease.read_bytes()).hexdigest())
    result = run_episode(dict(task_id=task.execution_id, released=raw), args.selector, config)
    print(json.dumps({k: result[k] for k in ['task_id', 'arm', 'attempted', 'valid', 'success']}))


if __name__ == '__main__':
    main()
