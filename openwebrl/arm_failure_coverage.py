"""Bounded deferred labels on valid failed groups, before any optimizer update.

Ordinary shadow labels remain unchanged. The q=0.4 treatment captures a nested
Bernoulli sample of exact pre-action states and labels only groups admitted by
the historical q=0.2 rule. The legacy four-turn pilot is retained separately.
"""
import asyncio
import base64
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from openwebrl.arm_failure_recipe import coverage_budget, validate_recipe, failure_fraction, deferred_enabled


def sampled_at(identity, fraction):
    from openwebrl.arm_turn_bonus import stable_seed
    return stable_seed(*identity, 'score') / 2**31 < fraction


def turn_key(identity):
    return hashlib.sha256(json.dumps([*identity, 'failure-four-turns-v1'],
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def sampling_identity(sample, config):
    """Use the captured request identity, independently of browser task aliases."""
    identity = sample.metadata.get('arm_sampling_identity')
    expected = [config['seed'], config.get('sampling_policy_id', config['policy_id']),
                str(sample.metadata['trajectory_id']), sample.metadata['turn_index']]
    if (not isinstance(identity, list) or len(identity) != 5
            or not isinstance(identity[2], str) or not identity[2]
            or [identity[0], identity[1], identity[3], identity[4]] != expected):
        raise ValueError('Missing or mismatched original ARM sampling identity')
    return identity


class TurnReservoir:
    def __init__(self, config, trajectory_id):
        validate_recipe(config)
        self.config = config
        self.trajectory_id = str(trajectory_id)
        self.budget = coverage_budget(config)
        self.fraction = failure_fraction(config)
        if not deferred_enabled(config):
            raise ValueError('State capture requires the coverage treatment')
        self.items = {}
        self.seen = set()
        self.persisted = False

    def capture(self, identity, record, state, timeout):
        if not state['original'][1]:
            return  # Native empty-response sentinels cannot supply trainable turns.
        if self.persisted or identity[-1] in self.seen:
            raise ValueError('Repeated turn or capture after persistence')
        self.seen.add(identity[-1])
        if len(self.seen) > 15:
            raise ValueError('Coverage pilot preserves the 15-turn horizon')
        if self.fraction == .4 and not sampled_at(identity, self.fraction):
            return
        key = turn_key(identity)
        if self.budget and len(self.items) == self.budget and key > max(self.items):
            return
        self.items[key] = dict(identity=list(identity), record=record,
            state=deepcopy(state), timeout=timeout)
        for dropped in sorted(self.items)[self.budget or 15:]:
            self.items.pop(dropped)

    def persist(self):
        if self.persisted:
            return
        from openwebrl.arm_turn_bonus import write_json
        root = Path(self.config['output'])/'failure_states'
        for key, item in sorted(self.items.items()):
            state = item['state']
            screenshot = state['observation']['screenshot']
            if not isinstance(screenshot, bytes) or not all(isinstance(i, str) for i in state['images']):
                raise ValueError('Expected original screenshot bytes and serialized actor images')
            state['observation']['screenshot'] = base64.b64encode(screenshot).decode()
            body = dict(schema=1, policy_id=self.config['policy_id'],
                checkpoint=self.config['checkpoint'], rollout_id=self.config['rollout_id'],
                trajectory_id=self.trajectory_id, identity=item['identity'],
                sample_index=item['record']['context']['parent_sample_index'],
                group_index=item['record']['context']['group_index'],
                timeout=item['timeout'], state=state)
            if len(json.dumps(body).encode()) > 16*1024**2:
                raise ValueError('Captured state exceeds the 16-MiB per-state bound')
            path = root/f'{key}.json'
            write_json(path, body)
            item['record']['failure_coverage_state'] = dict(path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                turn=item['identity'][-1], trajectory_turns=len(self.seen))
        self.items.clear()
        self.persisted = True


def retain_valid_group(args, samples, current):
    from openwebrl.arm_failure_bonus import invalid_group_reason, _rows
    from openwebrl.arm_failure_additive import retain
    from openwebrl.arm_turn_bonus import unit_bonus
    if reason := invalid_group_reason(args, samples, current):
        rejected = current.setdefault('failure_coverage_rejections', {})
        rejected[reason] = rejected.get(reason, 0)+1
        return
    # Preserve an independent copy of the historical admission reservoir for
    # same-collection diagnostics. No label sign is used to select either pool.
    rows = [s for trajectory in samples for s in _rows(trajectory)]
    if any(unit_bonus(s, current['config']['policy_id']) for s in rows):
        control = current.setdefault('failure_coverage_control', dict(config=current['config']))
        retain(control, samples)
    retain(current, samples)


def pool_summary(pool, policy_id):
    from openwebrl.arm_turn_bonus import unit_bonus
    from openwebrl.arm_failure_bonus import _rows
    rows = [s for group in pool.values() for trajectory in group for s in _rows(trajectory)]
    units = [unit_bonus(s, policy_id) for s in rows]
    return dict(groups=len(pool), turns=len(rows), labels=sum(x != 0 for x in units),
        positive=sum(x > 0 for x in units), negative=sum(x < 0 for x in units),
        coefficient=len(pool)/48, label_fraction=sum(x != 0 for x in units)/len(rows) if rows else 0)


def load_state(sample, config):
    ref = sample.metadata['arm_failure_coverage_state']
    path = Path(ref['path'])
    if path.parent.resolve() != (Path(config['output'])/'failure_states').resolve():
        raise ValueError('Deferred state belongs to a different collection')
    if path.stat().st_size > 16*1024**2:
        raise ValueError('Oversized deferred state')
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != ref['sha256']:
        raise ValueError('Deferred state changed after collection')
    body = json.loads(blob)
    expected = dict(schema=1, policy_id=config['policy_id'], checkpoint=config['checkpoint'],
        rollout_id=config['rollout_id'], sample_index=sample.index, group_index=sample.group_index,
        trajectory_id=str(sample.metadata['trajectory_id']))
    if any(body.get(k) != v for k, v in expected.items()):
        raise ValueError('Deferred state/actor/sample provenance mismatch')
    identity = sampling_identity(sample, config)
    if (body['identity'] != identity or path.stem != turn_key(identity)
            or ref['turn'] != identity[-1] or body['state']['original'][0] != sample.response):
        raise ValueError('Deferred original response or turn identity changed')
    body['state']['observation']['screenshot'] = base64.b64decode(
        body['state']['observation']['screenshot'], validate=True)
    return body


async def finalize(args, rollout_id, *, infer=None):
    from openwebrl.arm_turn_bonus import state, ShadowSelector, write_json
    from openwebrl.arm_failure_bonus import _rows
    current = state(args)
    config = current['config']
    budget = coverage_budget(config)
    if not deferred_enabled(config):
        return
    validate_recipe(config)
    if current.get('failure_coverage_complete') or config['rollout_id'] != rollout_id:
        raise ValueError('Repeated or stale deferred labeling round')
    if infer is None:
        from openwebrl.generate_browser import _run_inference_step
        infer = _run_inference_step
    pool = current.get('failure_additive_pool', {})
    before = pool_summary(pool, config['policy_id'])
    control = pool_summary(current.get('failure_coverage_control', {}).get('failure_additive_pool', {}),
                           config['policy_id'])
    selected = []
    for group in pool.values():
        for trajectory in group:
            trajectory = _rows(trajectory)
            chosen = [s for s in trajectory if s.metadata.get('arm_failure_coverage_state')]
            if failure_fraction(config) == .4:
                expected = {s.index for s in trajectory if sampled_at(sampling_identity(s, config), .4)}
                if {s.index for s in chosen} != expected:
                    raise ValueError('Missing Bernoulli-sampled pre-action states')
            elif len(chosen) != min(budget, len(trajectory)):
                raise ValueError('Missing uniformly sampled pre-action states')
            selected.extend(chosen)
    maximum = 600 if failure_fraction(config) == .4 else 160
    if len(selected) > maximum:
        raise ValueError('Failure label budget exceeds bounded group/turn population')
    # Validate every state before changing metadata or making an actor request.
    captured = [(s, load_state(s, config)) for s in selected]
    selected_ids = {s.index for s in selected}
    for group in pool.values():
        for trajectory in group:
            for s in _rows(trajectory):
                if s.index not in selected_ids:
                    s.metadata['arm_turn_bonus'] = dict(policy_id=config['policy_id'],
                        executed_index=0, eligible=False, sampled=False, reason='outside_failure_budget')
                s.metadata['arm_all_failure_group'] = True
    log_by_id = {(r['trajectory_id'], r['turn']):r for r in current['records']}
    sem = asyncio.Semaphore(4)
    results = []
    url = f'http://{args.sglang_router_ip}:{args.sglang_router_port}/generate'
    async def label(sample, captured):
        async with sem:
            old = log_by_id.get((str(sample.metadata['trajectory_id']), sample.metadata['turn_index']))
            # Reuse all already-attempted labels, including rejected/negative
            # ones. Retrying only poor labels would change the distribution.
            reused = bool(old and old.get('candidate_requests', 0))
            if reused and (old.get('policy_id') != config['policy_id'] or old.get('context') !=
                    dict(parent_sample_index=sample.index, group_index=sample.group_index)):
                raise ValueError('Already-attempted label has stale policy/sample provenance')
            record = deepcopy(old) if reused else dict(
                policy_id=config['policy_id'], executed_index=0, eligible=False, sampled=True,
                candidate_gate=config.get('candidate_gate','distinct5'),
                credit_assignment=config.get('credit_assignment','response_index'),
                context=dict(parent_sample_index=sample.index, group_index=sample.group_index),
                trajectory_id=str(sample.metadata['trajectory_id']), task_id=str(sample.metadata['task_id']),
                turn=sample.metadata['turn_index'], candidate_requests=0, candidate_response_tokens=0,
                selector_requests=0, reason='pending')
            if not reused:
                selector = ShadowSelector(config, sample.metadata['trajectory_id'])
                await selector._label(infer, url, captured['state'], captured['identity'], record,
                                      captured['timeout'])
            record.update(failure_coverage=True, reused_original_label=reused,
                          state_reference=sample.metadata['arm_failure_coverage_state'])
            if record.get('reason') in ('pending','cancelled'):
                raise ValueError('Deferred label was not completed')
            path = Path(config['output'])/'failure_labels'/f'{sample.index}.json'
            write_json(path, record)
            keys = ('policy_id','executed_index','eligible','sampled','reason','unit_bonus',
                'selected_index','candidate_gate','credit_assignment','bonus_kind','action_class_ids',
                'distinct_actions','executed_action_multiplicity','chance_baseline','selected_executed_action')
            sample.metadata['arm_turn_bonus'] = {k:record[k] for k in keys if k in record}
            sample.metadata['arm_turn_bonus']['record_file'] = str(path)
            results.append(record)
    started = time.monotonic()
    # A canceled/failed phase never reaches reward serialization or training.
    async with asyncio.timeout(1800):
        workers = [asyncio.create_task(label(s, body)) for s, body in captured]
        try:
            if workers:
                await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done(): task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
    after = pool_summary(pool, config['policy_id'])
    report = dict(complete=True, policy_id=config['policy_id'], checkpoint=config['checkpoint'],
        rollout_id=rollout_id, ordinary_beta=.5, failure_beta=.5, ordinary_q=.2,
        groups_seen=current.get('failure_additive_eligible',0),
        historical_admission_control=control, same_pool_twenty_percent=before,
        four_turn_coverage=after if budget else None, failure_coverage=after,
        failure_q=failure_fraction(config) if not budget else None,
        admission='historical q=0.2 label admission' if not budget else 'validity before labels',
        selected_turns=len(selected),
        reused=sum(r['reused_original_label'] for r in results),
        new_candidate_requests=sum(r['candidate_requests'] for r in results if not r['reused_original_label']),
        new_selector_requests=sum(r['selector_requests'] for r in results if not r['reused_original_label']),
        new_candidate_tokens=sum(r['candidate_response_tokens'] for r in results if not r['reused_original_label']),
        reasons=dict(Counter(r['reason'] for r in results)),
        retained_state_bytes=sum(Path(s.metadata['arm_failure_coverage_state']['path']).stat().st_size
                                 for s in selected),
        elapsed_seconds=time.monotonic()-started,
        rejected_groups=current.get('failure_coverage_rejections',{}),
        interpretation='Coverage/cost audit, not evidence of task-success gain')
    write_json(Path(config['output'])/'failure-coverage.json', report)
    current.update(failure_coverage_complete=True, failure_coverage_report=report)
