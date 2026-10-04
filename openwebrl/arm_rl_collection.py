"""Opt-in collection adapters for ARM RL mechanics pilots.

Instantiate per trajectory. Local labels remain separate from executed Samples.
The controller must drain() before changing actor weights or releasing services.
"""
import asyncio
import base64
from copy import deepcopy
import hashlib
import random

from openwebrl.arm_inference import ActionSelector, parse_selection, request_selection_result, split_response
from openwebrl.arm_rl import stable_seed


class LocalPreferenceSelector:
    def __init__(self, *, policy_id, trajectory_id, endpoint, sink, seed=42,
                 scored_fraction=.2, max_pending=4, timeout=120, request=request_selection_result):
        if not policy_id or not trajectory_id or not 0 <= scored_fraction <= 1 or max_pending < 1:
            raise ValueError('Invalid local-selector provenance/budget')
        self.policy_id, self.trajectory_id = policy_id, trajectory_id
        self.endpoint, self.sink, self.seed = endpoint, sink, seed
        self.scored_fraction, self.max_pending, self.timeout = scored_fraction, max_pending, timeout
        self.request, self.pending, self.audit = request, set(), []
        self.training_context = None

    def set_training_context(self, context):
        self.training_context = context

    async def __call__(self, *, infer, url, input_text, sampling_params, images,
                       observation, history, task, task_id, turn, timeout):
        # This request is identical to ordinary actor execution: ARM cannot choose it.
        original = await infer(url, input_text, sampling_params, images, timeout_secs=timeout)
        identity = (self.seed, self.policy_id, task_id, self.trajectory_id, turn)
        scored = stable_seed(*identity, 'score') / 2**31 < self.scored_fraction
        if scored and len(self.pending) < self.max_pending:
            # Freeze before the caller can execute the original action and mutate state.
            snapshot = deepcopy(dict(input_text=input_text, sampling_params=sampling_params,
                images=images, observation=observation, history=history, task=task,
                task_id=task_id, turn=turn, original=original, training_context=self.training_context))
            worker = asyncio.create_task(self._label(infer, url, snapshot, identity, timeout))
            self.pending.add(worker)
            worker.add_done_callback(self.pending.discard)
        else:
            self.audit.append(dict(turn=turn, reason='queue_full' if scored else 'not_sampled'))
        return original, dict(mode='local_preference', executed_index=0,
                              policy_id=self.policy_id, trajectory_id=self.trajectory_id)

    async def _label(self, infer, url, state, identity, timeout):
        workers = []
        try:
            # One deadline covers candidate generation, ARM and persistence.
            async with asyncio.timeout(self.timeout):
                for i in range(1, 5):
                    params = dict(state['sampling_params'], sampling_seed=stable_seed(*identity, i))
                    workers.append(asyncio.create_task(infer(url, state['input_text'], params,
                        state['images'], timeout_secs=timeout)))
                outputs = [state['original'], *await asyncio.gather(*workers)]
                order = list(range(5))
                random.Random(stable_seed(*identity, 'permutation')).shuffle(order)
                candidates = [split_response(o[0]) for o in outputs]
                payload = dict(mode='selection', task=state['task'],
                    url=state['observation'].get('active_tab_url', ''),
                    history=[split_response(r) for r in state['history']],
                    candidates=[candidates[i] for i in order],
                    screenshot=base64.b64encode(state['observation']['screenshot']).decode())
                result = await self.request(self.endpoint, payload, self.timeout, None)
                winner = order[parse_selection(result.get('raw', ''), 5)]
                record = dict(policy_id=self.policy_id, trajectory_id=self.trajectory_id,
                    task_id=state['task_id'], turn=state['turn'], executed_index=0,
                    selected_index=winner, permutation=order, raw_label=result.get('raw'),
                    prompt=state['input_text'], images=state['images'], outputs=outputs,
                    prompt_sha256=hashlib.sha256(state['input_text'].encode()).hexdigest(),
                    screenshot_sha256=hashlib.sha256(state['observation']['screenshot']).hexdigest(),
                    terminal_rewards=None, old_logps_require_teacher_forcing=True)
                record['training_context'] = state['training_context']
                # Sink must preserve each candidate's token IDs/logps and image provenance.
                await self.sink(record)
        except Exception as exc:
            self.audit.append(dict(turn=state['turn'], reason='auxiliary_dropped', error=type(exc).__name__))
        finally:
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)

    async def drain(self):
        if self.pending:
            await asyncio.gather(*tuple(self.pending))

    async def cancel(self):
        tasks = tuple(self.pending)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def rescue_selector(*, policy_id, trajectory_id, endpoint, output, exporter=None, seed=42):
    if not policy_id or not trajectory_id:
        raise ValueError('Rescue proposals require policy and trajectory provenance')
    return ActionSelector('selection', endpoint, output,
        seed=stable_seed(seed, policy_id, trajectory_id), candidates=5,
        exporter=exporter, candidate_representation='full')
