"""Persistent-state episode/retry and pre-action-selection protocol.

This direct-engine adapter validates evaluation semantics. Its dispatch count
is NOT a browser-step count, and its timings are NOT model serving benchmarks.
Policies receive public observations only; evaluator verdicts stay internal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Callable


@dataclass(frozen=True)
class Usage:
    requests: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    estimated_flops: float | None = None

    def __post_init__(self):
        for name in ('requests', 'input_tokens', 'output_tokens', 'cached_input_tokens'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError('Cached input exceeds total input')
        if self.estimated_flops is not None and (
                not math.isfinite(self.estimated_flops) or self.estimated_flops < 0):
            raise ValueError('FLOPs must be finite/nonnegative, or unknown')


@dataclass(frozen=True)
class Action:
    action_id: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Proposals:
    actions: tuple[Action, ...]
    usage: Usage | None = None


@dataclass(frozen=True)
class Selection:
    index: int
    usage: Usage | None = None


@dataclass(frozen=True)
class Protocol:
    episodes: int = 1
    candidates: int = 1
    steps_per_episode: int = 30
    action_budget: int = 30
    oracle_stop_after_success: bool = False

    def __post_init__(self):
        for name in ('episodes', 'candidates', 'steps_per_episode', 'action_budget'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.oracle_stop_after_success) is not bool:
            raise ValueError('oracle_stop_after_success must be boolean')


def _usage_summary(items):
    known = [x for x in items if x is not None]
    total = {key: sum(getattr(x, key) for x in known)
             for key in ('requests', 'input_tokens', 'output_tokens', 'cached_input_tokens')}
    total.update(metered_calls=len(known), unmetered_calls=len(items)-len(known),
                 estimated_flops=(sum(x.estimated_flops for x in known)
                    if len(known) == len(items) and all(x.estimated_flops is not None for x in known) else None))
    total['coverage'] = 'complete' if len(known) == len(items) else 'partial_or_unknown'
    return total


def run_campaign(sandbox, case_id: str, propose: Callable, *, protocol=Protocol(),
                 select: Callable | None = None, campaign_id: str | None = None):
    """Run one fresh method/case campaign, preserving the world across episodes.

    propose(observation, context) returns Proposals. An empty action tuple ends
    the episode while retaining its usage receipt. None is an unmetered stop.
    select(observation, tuple[Action, ...], context) returns Selection.
    A proposal generation is not a sandbox dispatch. Only the selected action
    executes. Each method gets a separate fresh campaign from the same fixture.
    Model/token usage is supplied by adapters; absent receipts remain unknown.
    This version enforces actions/episodes, not a hard model-compute cap.
    """
    if protocol.candidates > 1 and select is None:
        raise ValueError('Multiple proposals require a selector')
    campaign = sandbox.start_campaign(case_id, campaign_id=campaign_id)
    campaign = campaign['campaign_id'] if isinstance(campaign, dict) else campaign
    records, episodes, actor_usage, selector_usage = [], [], [], []
    dispatches = 0
    start = time.monotonic()
    for episode_index in range(protocol.episodes):
        if dispatches >= protocol.action_budget:
            break
        episode = sandbox.start_episode(campaign)
        episode = episode['episode_id'] if isinstance(episode, dict) else episode
        termination = 'episode_step_cap'
        for step_index in range(protocol.steps_per_episode):
            if dispatches >= protocol.action_budget:
                termination = 'campaign_action_cap'
                break
            observation = sandbox.observe(campaign)
            context = dict(episode_index=episode_index, step_index=step_index,
                           requested_candidates=protocol.candidates,
                           remaining_actions=protocol.action_budget-dispatches)
            t0 = time.monotonic()
            batch = propose(observation, context)
            proposal_seconds = time.monotonic()-t0
            if batch is None:
                actor_usage.append(None)
                termination = 'unmetered_policy_stop'
                break
            if not isinstance(batch, Proposals):
                raise ValueError('Expected Proposals with explicit usage or unknown coverage')
            actor_usage.append(batch.usage)
            if not batch.actions:
                termination = 'policy_stop'
                break
            if len(batch.actions) != protocol.candidates:
                raise ValueError('Proposal count differs from protocol')
            if not all(isinstance(a, Action) for a in batch.actions):
                raise ValueError('Expected structured Action proposals')
            t0 = time.monotonic()
            choice = select(observation, batch.actions, context) if select else Selection(0, Usage(0, 0, 0, estimated_flops=0))
            selector_seconds = time.monotonic()-t0
            if not isinstance(choice, Selection) or type(choice.index) is not int or not 0 <= choice.index < len(batch.actions):
                raise ValueError('Selector index outside candidate panel')
            selector_usage.append(choice.usage)
            sandbox.record_candidates(campaign, episode,
                                      [asdict(a) for a in batch.actions], selected_index=choice.index)
            action = batch.actions[choice.index]
            t0 = time.monotonic()
            receipt = sandbox.execute(campaign, episode, action.action_id, action.parameters)
            action_seconds = time.monotonic()-t0
            dispatches += 1
            records.append(dict(episode_index=episode_index, step_index=step_index,
                                proposals=[asdict(a) for a in batch.actions], selected_index=choice.index,
                                actor_usage=asdict(batch.usage) if batch.usage else None,
                                selector_usage=asdict(choice.usage) if choice.usage else None,
                                proposal_seconds=proposal_seconds, selector_seconds=selector_seconds,
                                action_seconds=action_seconds, action_receipt=receipt))
        verdict = sandbox.verdict(campaign)
        episodes.append(dict(episode_index=episode_index, episode_id=episode,
                             termination=termination, verdict=verdict))
        if protocol.oracle_stop_after_success and verdict['goal_success']:
            break
    final = sandbox.verdict(campaign)
    any_success = any(e['verdict']['goal_success'] for e in episodes)
    return dict(schema_version=1, transport='direct-state-engine', campaign_id=campaign,
                case_id=case_id, protocol=asdict(protocol), episodes=episodes, decisions=records,
                outcomes=dict(final_goal_success=final['goal_success'],
                              any_episode_goal_success=any_success,
                              any_harm=final['any_harm'],
                              harm_free_final_success=final['safe_success'],
                              harm_free_any_episode_success=any_success and not final['any_harm']),
                sandbox_action_dispatches=dispatches, elapsed_seconds=time.monotonic()-start,
                usage=dict(actor=_usage_summary(actor_usage), selector=_usage_summary(selector_usage)),
                limitations=['Direct engine dispatches are not physical browser steps.',
                             'Policies and usage receipts are supplied by callers; no actor adapter is bundled.',
                             'Compute totals with missing receipts remain incomplete; hard compute caps are not implemented.',
                             'Free success-based stopping, when enabled, is an oracle sensitivity.'])
