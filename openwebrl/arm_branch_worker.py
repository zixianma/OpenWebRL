"""Fresh fixed-SFT prefixes, outcome-blind replay barriers and continuations.

One process owns one episode. All network work needs a registered allocation
and the shared all-attempt ledger; importing this module starts no work.
"""
from __future__ import annotations
import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from urllib.parse import urlparse

from openwebrl.arm_branch_protocol import (
    SAMPLING, choose_distinct, fixed_candidates, compare_observations, observation_evidence, seed,
)
from openwebrl.controlled_sft_eval import (
    ScopedBudget, HTTPTransports, generation_bindings, normalized_task,
)
from openwebrl.controlled_sft_worker import (
    AttemptJournal, ControlledSFTPolicy, ControlledClient, CanonicalOM2WReward,
    judge_terminal, validate_actor_environment_config,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers treat existence as readiness. Publish only complete, fsynced JSON;
    # link preserves exclusive creation without overwriting earlier receipts.
    staged = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.staged')
    with staged.open('x') as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.link(staged, path)
    # Keep the staged inode as a private write receipt; it shares the data blocks.
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def save_observation(journal, observation, name):
    result = deepcopy(observation)
    result['screenshot'] = journal.blob(name + '.png', result['screenshot'])
    return result


def load_image(observation):
    receipt = observation['screenshot']
    raw = Path(receipt['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise ValueError('Preserved screenshot changed')
    return raw


class Captured(RuntimeError):
    pass


class ReplayRejected(RuntimeError):
    pass


class BranchPolicy(ControlledSFTPolicy):
    def __init__(self, *, mode, directory, anchor=None, candidate=None, repeat=None,
                 state_directory=None, anchor_depth=None, candidate_count=3, candidate_draws=9,
                 retain_duplicate_candidates=False, **kwargs):
        super().__init__('L01', **kwargs)
        self.mode, self.directory = mode, Path(directory)
        if not 1 <= candidate_count <= candidate_draws:
            raise ValueError('Invalid candidate panel dimensions')
        self.candidate_count, self.candidate_draws = candidate_count, candidate_draws
        self.retain_duplicate_candidates = retain_duplicate_candidates
        if retain_duplicate_candidates and candidate_count != candidate_draws:
            raise ValueError('Fixed candidate panel cannot resample or filter proposals')
        self.anchor, self.candidate, self.repeat = anchor, candidate, repeat
        self.state_directory = Path(state_directory) if state_directory else None
        self.depth = anchor['depth'] if anchor else anchor_depth
        self.prefix, self.captured, self.released = [], False, False
        self.rejected, self.awaiting_candidate = None, False
        self.replay_checks = []

    def _reserve(self, kind, count, **context):
        # Replayed responses never call the actor; do not fabricate generation cost.
        if kind == 'local_sft_generations' and self.mode == 'branch' and self.decisions <= self.depth:
            return None
        return super()._reserve(kind, count, **context)

    async def _decide(self, **kwargs):
        turn, observation = kwargs['turn'], kwargs['observation']
        evidence = observation_evidence(observation)
        before = save_observation(self.journal, observation, f'before-{turn:03d}')
        if self.mode == 'branch' and turn <= self.depth:
            expected = self.anchor['prefix'][turn] if turn < self.depth else self.anchor
            try:
                check = compare_observations(expected['evidence'], evidence,
                    load_image(expected['before_observation']), observation['screenshot'])
                self.replay_checks.append(dict(turn=turn, **check))
            except ValueError as exc:
                self.rejected = str(exc)
                self.journal('reconstruction_rejected', dict(turn=turn, reason=str(exc)))
                raise ReplayRejected(str(exc)) from exc
            if turn == self.depth:
                ready = dict(candidate=self.candidate, repeat=self.repeat,
                    attempt_directory=str(self.directory), checks=self.replay_checks,
                    before_observation=before, anchor_sha256=self.anchor['artifact_sha256'])
                write(self.directory/'ready.json', ready)
                deadline = time.monotonic() + 600
                while not (self.state_directory/'release.json').exists():
                    if (self.state_directory/'reject.json').exists():
                        self.rejected = 'State group rejected before any candidate dispatch'
                        raise ReplayRejected(self.rejected)
                    if time.monotonic() > deadline:
                        self.rejected = 'Reconstruction barrier timeout'
                        raise ReplayRejected(self.rejected)
                    await asyncio.sleep(.5)
                release = json.loads((self.state_directory/'release.json').read_text())
                if release['anchor_sha256'] != self.anchor['artifact_sha256']:
                    raise ValueError('Barrier released a different anchor')
                if str(self.directory) not in release['attempt_directories']:
                    raise ValueError('Attempt is not in the accepted branch cohort')
                # Every branch gets the same remaining operation and wall-time budget.
                self.started, self.released, self.awaiting_candidate = self.clock(), True, True

        original_infer = kwargs['infer']
        candidates = None

        async def inference(url, text, sampling, images, timeout_secs=None):
            nonlocal candidates
            if self.mode == 'branch' and turn <= self.depth:
                output = (self.anchor['prefix'][turn]['output'] if turn < self.depth
                          else self.anchor['candidates'][self.candidate]['output'])
                self.journal('forced_response', dict(turn=turn, output=output,
                    role='prefix' if turn < self.depth else 'candidate'))
                return tuple(output)
            if self.mode == 'discover' and turn == self.depth:
                # Fixed proposal count, first distinct parsed bundles. No
                # outcome, teacher or probability-based screening of candidates.
                super(BranchPolicy, self)._reserve('local_sft_generations', self.candidate_draws - 1)
                async def proposal(index):
                    output = await original_infer(url, text,
                        dict(sampling, sampling_seed=seed(self.task_id, 'candidate', index)),
                        images, timeout_secs=timeout_secs)
                    self.journal('candidate_draw', dict(index=index, output=output))
                    return output
                outputs = await asyncio.gather(*(proposal(i) for i in range(self.candidate_draws)), return_exceptions=True)
                failures = [x for x in outputs if isinstance(x, BaseException)]
                if failures:
                    raise failures[0]
                candidates = outputs
                return tuple(outputs[0])
            return await original_infer(url, text, dict(sampling,
                sampling_seed=seed(self.task_id, self.repeat if self.released else 'prefix', turn)),
                images, timeout_secs=timeout_secs)

        kwargs['infer'] = inference
        output, metadata = await super()._decide(**kwargs)
        if self.mode == 'discover':
            entry = dict(evidence=evidence, before_observation=before, output=output)
            if turn < self.depth:
                self.prefix.append(entry)
            else:
                try:
                    choose = fixed_candidates if self.retain_duplicate_candidates else choose_distinct
                    candidates = choose(candidates, self.candidate_count)
                except ValueError as exc:
                    self.rejected = 'Malformed fixed candidate panel' if self.retain_duplicate_candidates else 'Insufficient distinct parseable candidates'
                    self.journal('candidate_panel_rejected', dict(reason=str(exc)))
                    raise ReplayRejected(self.rejected) from exc
                anchor = dict(task=kwargs['task'], task_id=self.task_id, depth=self.depth,
                    history=kwargs['history'], prefix=self.prefix, candidates=candidates,
                    evidence=evidence, before_observation=before)
                write(self.directory/'anchor.json', anchor)
                self.captured = True
                raise Captured('Anchor saved before any proposed candidate was executed')
        return output, dict(metadata, branch_mode=self.mode, candidate=self.candidate, repeat=self.repeat)


@contextmanager
def branch_environment(generation, final_state, policy, budget):
    original = generation._create_env

    class RecordingClient:
        def __init__(self, client): self.client = client
        def __getattr__(self, name): return getattr(self.client, name)
        async def step(self, actions):
            if policy.awaiting_candidate:
                # Live pages can change while siblings reach the barrier. Check
                # again immediately before dispatch; retain a failed check as
                # an invalid accepted attempt, never silently substitute it.
                from openwebrl.controlled_browser_server import fresh_observation
                current = await fresh_observation(self.client)
                saved = save_observation(policy.journal, current, 'pre-dispatch')
                compare_observations(policy.anchor['evidence'], observation_evidence(current),
                    load_image(policy.anchor['before_observation']), current['screenshot'])
                policy.journal('candidate_pre_dispatch_verified', dict(observation=saved))
            result = await self.client.step(actions)
            if policy.awaiting_candidate:
                policy.awaiting_candidate = False
                observation = save_observation(policy.journal, result[0], 'immediate')
                write(policy.directory/'immediate.json', dict(candidate=policy.candidate,
                    repeat=policy.repeat, observation=observation, tool_feedback=result[4],
                    terminated=bool(result[2]), truncated=bool(result[3])))
            return result

    async def create(*args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs['env_config']
        validate_actor_environment_config(config)
        reservation = budget.reserve('browser_sessions', 1)
        # Own the child before initialization: reset may reject an unstable
        # observation before the native factory returns its client handle.
        from openwebrl.env import local_process_env
        transport_create = local_process_env.create_local_process_env

        def own(env):
            wrapped = ControlledClient(RecordingClient(env), final_state, budget=budget, reservation=reservation)
            final_state['client'] = wrapped
            return wrapped

        async def owned_create(local_config):
            return own(await transport_create(local_config))

        local_process_env.create_local_process_env = owned_create
        try:
            env, task = await original(*args, **kwargs)
        finally:
            local_process_env.create_local_process_env = transport_create
        return (env if isinstance(env, ControlledClient) else own(env)), task

    generation._create_env = create
    try:
        yield
    finally:
        generation._create_env = original


async def run(claim, state, actor_url):
    import httpx
    from openwebrl import generate_browser as generation, run_evaluate as evaluation
    from openwebrl.eval import reward_online_mind2web as reward
    config = deepcopy(state.plan['worker_config'])
    for key in ('prompt', 'tool_list'):
        if sha(config[key+'_path']) != config[key+'_sha256']:
            raise ValueError('Frozen prompt/tool file changed')
    task = normalized_task(claim['task'], claim['task_id'])
    config['episode_task'] = task
    directory = Path(claim['artifact_directory'])
    journal, budget = AttemptJournal(directory/'events'), ScopedBudget(state, claim)
    item = claim['item']
    state_dir = state.root/'states'/item['state']
    anchor = None
    if item['mode'] == 'branch':
        reference = json.loads((state_dir/'anchor-reference.json').read_text())
        if sha(reference['path']) != reference['sha256']:
            raise ValueError('Frozen anchor artifact changed')
        anchor = json.loads(Path(reference['path']).read_text())
        anchor['artifact_sha256'] = reference['sha256']
    policy = BranchPolicy(mode=item['mode'], directory=directory, anchor=anchor,
        state_directory=state_dir, candidate=item.get('candidate'), repeat=item.get('repeat'),
        candidate_count=state.plan['candidate_count'], candidate_draws=state.plan['candidate_draws'],
        retain_duplicate_candidates=state.plan.get('retain_duplicate_candidates', False),
        anchor_depth=item['depth'], task_id=claim['task_id'], budget=budget, journal=journal,
        selector_call=None, prompt_path=config['prompt_path'], prompt_sha256=config['prompt_sha256'])
    manifest = dict(config['browser_manifest'], artifact_directory=str(directory/'browser'))
    write(directory/'browser-manifest.json', manifest)
    os.environ.update(OPENWEBRL_CONTROLLED_BROWSER_MANIFEST=str(directory/'browser-manifest.json'),
        OPENWEBRL_CONTROLLED_BROWSER_MANIFEST_SHA256=sha(directory/'browser-manifest.json'),
        SLIME_BROWSER_ENV_MODE='local_process', SLIME_BROWSER_ROLLOUT_CONCURRENCY='1',
        SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING='0', SLIME_BROWSER_APPEND_THINKING_PREFILL='1',
        SLIME_BROWSER_THINKING_OPEN_TAG='<think>', SLIME_BROWSER_THINKING_CLOSE_TAG='</think>')
    address = urlparse(actor_url)
    args = evaluation.EvalArgs(sglang_router_ip=address.hostname, sglang_router_port=address.port,
        hf_checkpoint=config['actor_checkpoint'], max_steps=60, max_consecutive_parse_failures=3,
        context_num_screenshots=1, judge_api_model='o4-mini-2025-04-16', judge_api_mode='served',
        judge_timeout_secs=120, browser_response_format_mode='browser_env', turn_history_reasoning_mode='full',
        browser_include_tool_response=1, inference_step_timeout_secs=180, task_timeout_secs=3600,
        rollout_temperature=1., rollout_top_p=.95, rollout_top_k=-1, rollout_max_response_len=4096,
        rollout_max_context_len=32768, path_to_save_generated_samples=str(directory/'rollouts'))
    args.browser_action_selector = policy
    sample = reward.Sample(index=task['index'], prompt=task['intent'],
        metadata=dict(task, _browser_task_file=config['tasks_path']))
    final_state, started = {}, time.monotonic()
    async with httpx.AsyncClient(trust_env=False, timeout=180) as client:
        transport = HTTPTransports(client, actor_base_url=actor_url, kev_base_url=actor_url,
            openai_key=os.getenv('JUDGE_API_KEY') or os.getenv('OPENAI_API_KEY'), jev_key=None,
            journal=journal, normalize_actor=generation._ensure_im_end_w_new_line, on_failure=budget.halt)
        try:
            with generation_bindings(generation, config=config, policy=policy, actor_call=transport.actor_call):
                with branch_environment(generation, final_state, policy, budget):
                    samples = await generation.generate_turn_sample(args, sample, dict(SAMPLING))
            outcome = dict(valid=False, score=None, reason='anchor_not_captured', judge_http_attempts=0)
            if policy.released:
                judge = CanonicalOM2WReward(reward, source_sha256=config['canonical_reward_sha256'],
                    chat_create=transport.judge, budget=budget, journal=journal, task_id=claim['task_id'])
                scored = samples if any('turn_index' in s.metadata for s in samples) else samples[-1]
                outcome = await judge_terminal(args=args, samples=scored, final_state=final_state,
                    policy=policy, canonical_reward=judge, journal=journal)
            elif policy.captured:
                outcome.update(valid=True, reason='anchor_captured')
            elif policy.rejected:
                outcome['reason'] = policy.rejected
            result = dict(task_id=claim['task_id'], condition='L01', attempt_id=claim['attempt_id'],
                state=item['state'], mode=item['mode'], candidate=item.get('candidate'), repeat=item.get('repeat'),
                valid=outcome['valid'], reward=outcome['score'], reason=outcome['reason'],
                captured=policy.captured, released=policy.released, browser_closed=final_state.get('browser_closed', False),
                terminal_capture_seconds=final_state.get('terminal_capture_seconds'),
                terminal_capture_error_type=final_state.get('capture_error_type'),
                elapsed_seconds=time.monotonic()-started, actor_decisions=policy.decisions,
                browser_operations=(final_state.get('observation') or {}).get('controlled', {}).get('action_attempts'),
                judge_http_attempts=outcome['judge_http_attempts'],
                halt_required=bool(policy.provider_invalid or policy.contract_invalid or policy.budget_invalid))
            write(directory/'result.json', result)
            return dict(result, result_path=str(directory/'result.json'), result_sha256=sha(directory/'result.json'))
        finally:
            if final_state.get('client') is not None:
                await final_state['client'].exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--item', required=True)
    parser.add_argument('--actor-url', required=True)
    args = parser.parse_args()
    from openwebrl.arm_branch_state import BranchState
    state = BranchState(args.root)
    claim = state.claim_item(args.item, str(os.getpid()))
    result = asyncio.run(run(claim, state, args.actor_url))
    state.finish(claim['attempt_id'], result)
    if result['halt_required']:
        state.halt('Branch worker requires diagnosis', result=result)
        raise RuntimeError('Branch worker halted')


if __name__ == '__main__':
    main()
