"""Common-candidate WebArena execution; no model/browser starts at import.

This is an explicit viewer reconstruction, not recovered historical Table 5 code.
NoSearch requests one action; first-of-five remains a separate diagnostic.
Only original Reddit has a validated hosting/authentication lease adapter.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
from urllib.parse import quote, urlsplit

from .jev_eval import write_json as _atomic_json
from .webprm_arbiter_harness import (load_arbiter_harness, load_native_observation_processor,
                                    native_execution_code)
from .webprm_lifecycle import mark_episode_phase, run_supervised_episode, save_outcome_and_close
from .webprm_source import ARBITER_REVISION, _extract, load_arbiter, load_shepherd, source_manifest
from .webprm_webarena import (EvaluationNotReady, digest, evaluation_requirements, load_native_evaluator,
    NATIVE_EVALUATOR_PROTOCOL, ABSTENTION_EVALUATOR_PROTOCOL, evaluator_protocol)
from .webprm_webarena_episode import (EpisodeHooks, NativeTrajectory, TaskEnvelope,
                                     open_authenticated_browser, score_and_close)
from .webprm_worker import (ACTOR, CHROMIUM, arbiter_auxiliary, completion, native_tournament,
                            input_token_upper_bound, prepare_browser, single_response_text)

PRIMARY_SELECTORS = ('no_search', 'gpt4omini', 'shepherd8b', 'arbiter7b', 'piotr_selectionarm')
DIAGNOSTIC_SELECTORS = ('first5', 'random5')
SELECTORS = PRIMARY_SELECTORS + DIAGNOSTIC_SELECTORS
ACTOR_SNAPSHOTS = (ACTOR, 'gpt-4o-2024-05-13')
PROTOCOL = 'webarena_table5_viewer_reconstruction_v2'
STRICT_TURN_LIMIT = 'strict_actual_answer_v1'
NA_TURN_LIMIT = 'released_viewer_na_fallback_v1'
LEGACY_NO_SEARCH_PROMPT = 'single_action_reconstruction_v1'
ENVELOPE_NO_SEARCH_PROMPT = 'single_blocks_envelope_v2'
CAMPAIGN_SCIENCE = dict(protocol='fixed_table5_scientific_bindings_v1',seed=42,max_turns=30,
    fuzzy_enabled=True,fuzzy_model='gpt-4o-2024-05-13',fuzzy_protocol='webarena_fuzzy_gpt4o_20240513_v1',
    fuzzy_model_change_approved=True,fuzzy_max_input_tokens=8192,fuzzy_max_calls=1)


def no_search_prompt_version(config):
    value = config.get('no_search_prompt_version', LEGACY_NO_SEARCH_PROMPT)
    if value not in (LEGACY_NO_SEARCH_PROMPT, ENVELOPE_NO_SEARCH_PROMPT):
        raise EvaluationNotReady('Unknown explicit NoSearch prompt version')
    return value


def turn_limit_protocol(config):
    value = config.get('turn_limit_protocol', STRICT_TURN_LIMIT)
    if value not in (STRICT_TURN_LIMIT, NA_TURN_LIMIT):
        raise EvaluationNotReady('Unknown explicit turn-limit protocol')
    return value


def write_json(path, value):
    """Check both atomic-write paths, then sync the replaced directory entry."""
    from scripts.check_webprm_reddit_reset import safe_path
    path = Path(path)
    for candidate in (path, path.with_suffix(path.suffix + '.tmp')):
        safe_path(candidate)
        if candidate.is_symlink():
            raise ValueError('Refusing an evidence-file symlink')
    _atomic_json(path, value)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def cell_name(actor, selector):
    if actor not in ACTOR_SNAPSHOTS or selector not in SELECTORS:
        raise EvaluationNotReady('Actor or selector has no validated execution adapter')
    return actor + '__' + selector


def validate_execution_config(config):
    """Reject unsafe paths and unapproved/non-finite budgets before supervision."""
    from scripts.check_webprm_reddit_reset import safe_path
    if config.get('authorization_status') != 'approved':
        raise EvaluationNotReady('An explicitly approved execution plan is required')
    if str(config.get('job_id')) != os.environ.get('SLURM_JOB_ID'):
        raise EvaluationNotReady('Execution must stay inside its approved existing allocation')
    if any(key in config for key in ('protocol', 'protocol_contract', 'turn_limit_protocol',
                                      'site_runtime', 'native_helpers', 'evaluator_observations')):
        validate_protocol_contract(config)
    cap, deadline = config.get('api_usd_cap'), config.get('deadline')
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
        raise EvaluationNotReady('Explicit finite positive approved API cap required')
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.time():
        raise EvaluationNotReady('Explicit unexpired finite allocation deadline required')
    if config.get('max_turns') != 30 or type(config.get('seed')) is not int:
        raise EvaluationNotReady('Common 30-turn protocol and integer seed required')
    paid_models = {config.get('actor_model')}
    if config.get('selector') == 'gpt4omini':
        paid_models.add(ACTOR)
    if config.get('fuzzy_enabled'):
        from .webprm_webarena_fuzzy import validate_fuzzy_configuration
        paid_models.add(validate_fuzzy_configuration(config)['model'])
    for model in paid_models:
        rates = config.get('prices_usd_per_token', {}).get(model)
        if (not isinstance(rates, (list, tuple)) or len(rates) != 2 or
                any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates)):
            raise EvaluationNotReady('Explicit finite positive model prices required')
    for name in ['output', 'budget_path', 'lease_path', 'broker_socket']:
        if not config.get(name):
            raise EvaluationNotReady('Missing explicit runtime path: ' + name)
        safe_path(config[name])
    critic = {'arbiter7b': 'arbiter', 'shepherd8b': 'shepherd', 'piotr_selectionarm': 'piotr'}.get(config.get('selector'))
    if critic and any(not isinstance(config.get(critic + suffix), str) or not config[critic + suffix]
                      for suffix in ('_model', '_endpoint')):
        raise EvaluationNotReady('Explicit frozen critic model and broker route required')
    if config.get('selector') == 'piotr_selectionarm':
        from .webprm_webarena_piotr import validate_piotr_selectionarm
        if config.get('piotr_manifest') != validate_piotr_selectionarm(config):
            raise EvaluationNotReady('Frozen Piotr source/checkpoint manifest required')
    # v6 uses these fixed existing Chromium cache locations. Resolve before its
    # mkdir call so a preexisting cache symlink cannot cross the no-write boundary.
    safe_path('/tmp/openwebrl-webprm-browser-cache/chromium-1117/chrome-linux')


def validate_frozen_sources(config):
    required = {'arbiter_harness', 'arbiter_observation', 'viewer_reward', 'arbiter_parser', 'shepherd'}
    manifests = config.get('source_manifests', {})
    if set(manifests) != required:
        raise EvaluationNotReady('Complete native actor/observation/selector source freeze required')
    for name, expected in manifests.items():
        root = config['shepherd_source'] if name == 'shepherd' else config['arbiter_source']
        if source_manifest(root, expected['files'], expected['revision']) != expected:
            raise EvaluationNotReady('Frozen native source changed: ' + name)


def require_supported(task, selector, config=None):
    if selector not in SELECTORS:
        raise EvaluationNotReady('Selector is not part of the declared Table5/Piotr comparison')
    requirements = evaluation_requirements(task.native_config())
    if requirements['paid_fuzzy_judge_may_be_required']:
        if not config or config.get('fuzzy_enabled') is not True:
            raise EvaluationNotReady('Fixed task requires disabled native GPT-4-1106-preview; explicit available and approved judge protocol required')
        from .webprm_webarena_fuzzy import validate_fuzzy_configuration
        validate_fuzzy_configuration(config)
    if requirements['helper_functions_required']:
        from .webprm_webarena_helper_runtime import validate_helper_plan
        validate_helper_plan(task, config or {})


def load_viewer_reward(root):
    """Exact pinned online viewer builder; explicitly not the static dataset template."""
    root = Path(root)
    files = ['viewer/src/prompts/reward_p.py', 'viewer/src/prompts/prompt_builder.py']
    manifest = source_manifest(root, files, ARBITER_REVISION)
    namespace = {}
    _extract(root / files[0], namespace, names=set(), constants=True)
    namespace.update(reward_context_template=namespace['context_rft_v3'],
                     reward_criteria=namespace['evaluation_summarized_v3'], reward_role=namespace['role'])
    _extract(root / files[1], namespace, names={'RewardPromptBuilder'})
    return SimpleNamespace(manifest=manifest, build=namespace['RewardPromptBuilder'].build_reward_prompt)


def single_action_system_prompt(original, *, version=LEGACY_NO_SEARCH_PROMPT):
    """Disclosed NoSearch reconstruction with an explicit envelope clarification.

    V1 preserves the original examples. Opt-in V2 wraps the positive example
    in the required BLOCKS array and clarifies the matching instruction.
    Current-state reasoning, action grammar, user prompt, state difference and
    terminal detection remain shared. Neither version is paper-exact evidence.
    """
    no_search_prompt_version({'no_search_prompt_version': version})
    replacements = [
        ('You are an autonomous web navigation agent. Generate 5 diverse, PARALLEL action candidates for the current page state.\n'
         'CRITICAL: These are 5 independent choices, NOT a sequential plan. Each candidate must be executable from the CURRENT state.\n\n',
         'You are an autonomous web navigation agent. Generate exactly ONE action for the current page state.\n'
         'CRITICAL: Return one action, NOT a sequential plan. The action must be executable from the CURRENT state.\n\n'),
        ('{"BLOCKS":[{"thought":"...","action":"...","confidence":0.0-1.0}, ... (5 items)]}',
         '{"BLOCKS":[{"thought":"...","action":"...","confidence":0.0-1.0}]}'),
    ]
    prompt = original
    for previous, replacement in replacements:
        if prompt.count(previous) != 1:
            raise EvaluationNotReady('Pinned native single-action adaptation anchor changed')
        prompt = prompt.replace(previous, replacement, 1)
    begin, end = '# DIVERSITY REQUIREMENT CRITICAL\n', '# ACTION RULES\n'
    if prompt.count(begin) != 1 or prompt.count(end) != 1:
        raise EvaluationNotReady('Pinned native diversity-section anchors changed')
    prefix, rest = prompt.split(begin, 1)
    _, suffix = rest.split(end, 1)
    if version == ENVELOPE_NO_SEARCH_PROMPT:
        good_header = '# EXAMPLE (GOOD - with clear justification)\n'
        bad_header = '# ANTI-EXAMPLES (BAD - DO NOT IMITATE)\n'
        if suffix.count(good_header) != 1 or suffix.count(bad_header) != 1:
            raise EvaluationNotReady('Pinned native NoSearch example anchors changed')
        before, example = suffix.split(good_header, 1)
        positive, after = example.split(bad_header, 1)
        lines = positive.splitlines(keepends=True)
        object_lines = [i for i, line in enumerate(lines) if line.startswith('{"thought":')]
        if len(object_lines) != 1:
            raise EvaluationNotReady('Pinned native NoSearch positive example changed')
        index = object_lines[0]
        block = json.loads(lines[index])
        if set(block) != {'thought', 'action', 'confidence'}:
            raise EvaluationNotReady('Pinned native NoSearch example fields changed')
        lines[index] = json.dumps({'BLOCKS': [block]}, separators=(',', ':')) + '\n'
        suffix = before + good_header + ''.join(lines) + bad_header + after
        instruction = ('Return one JSON object with the key BLOCKS whose value is a one-element array.\n'
            'Put thought, action, and confidence inside that array element; never return a bare action object.\n'
            'The single BLOCK represents one action from the CURRENT state.\n')
    else:
        instruction = 'Return exactly one BLOCK representing one action from the CURRENT state.\n'
    return prefix + ('# SINGLE ACTION REQUIREMENT\n'
        + instruction +
        'Do NOT chain actions or return alternative candidates.\n'
        'The action must be executable RIGHT NOW from the current page.\n\n') + end + suffix


def protocol_definition(source, *, turn_limit=STRICT_TURN_LIMIT, piotr_representation=None,
                        no_search_prompt=LEGACY_NO_SEARCH_PROMPT,
                        scoring_protocol=NATIVE_EVALUATOR_PROTOCOL, helper_configuration=None,
                        site_configuration=None, observation_configuration=None, campaign_science=None):
    evaluator_protocol(scoring_protocol)
    if scoring_protocol == ABSTENTION_EVALUATOR_PROTOCOL and no_search_prompt != ENVELOPE_NO_SEARCH_PROMPT:
        raise EvaluationNotReady('Prospective abstention guard requires the explicit v4 NoSearch protocol')
    turn_limit_protocol({'turn_limit_protocol': turn_limit})
    no_search_prompt_version({'no_search_prompt_version': no_search_prompt})
    if no_search_prompt == ENVELOPE_NO_SEARCH_PROMPT and turn_limit != NA_TURN_LIMIT:
        raise EvaluationNotReady('Clarified NoSearch prompt requires the released turn-limit fallback')
    from .webprm_webarena_piotr import REPRESENTATION, REPRESENTATION_V2, representation
    selected_representation = representation({'piotr_representation': REPRESENTATION if piotr_representation is None else piotr_representation})
    if piotr_representation is not None and not isinstance(piotr_representation, str):
        raise EvaluationNotReady('Piotr representation must be an explicit string')
    if selected_representation == REPRESENTATION_V2 and turn_limit != NA_TURN_LIMIT:
        raise EvaluationNotReady('Descriptor-v2 requires the explicit corrected v3 protocol')
    native = load_arbiter_harness(source).policy_system_prompt
    definition = dict(identity=PROTOCOL, paper_exact_protocol=False,
        primary_selectors=list(PRIMARY_SELECTORS), diagnostic_selectors=list(DIAGNOSTIC_SELECTORS),
        guided_requested_raw_slots=5, no_search_requested_raw_slots=1,
        policy_request=dict(temperature=1., top_p=.9, n=1, max_tokens=2048),
        pairwise_request=dict(temperature=0., top_p=1., max_tokens=2048),
        gpt_selector_model=ACTOR,
        five_action_prompt_sha256=hashlib.sha256(native.encode()).hexdigest(),
        single_action_prompt_sha256=hashlib.sha256(single_action_system_prompt(native, version=no_search_prompt).encode()).hexdigest(),
        no_search_origin='pinned_viewer_prompt_with_disclosed_one_action_reconstruction',
        pairwise_origin='same_pinned_online_viewer_prompt_and_native_queue_tournament',
        invalid_slots='preserve_original_slots_and_errors_no_resampling_select_admissible_remainder',
        empty_pool='consume_turn_without_action_or_selector_call',
        paper_confirmed='two_policy_families_five_guided_candidates_knockout_and_four_published_rows',
        reconstruction='dated_model_snapshots_prompts_sampling_terminal_and_task_membership_not_proven_paper_exact')
    if turn_limit == NA_TURN_LIMIT:
        definition.update(identity='webarena_table5_viewer_reconstruction_v3',
            turn_limit_protocol=turn_limit,
            turn_limit_behavior='normal_exhaustion_executes_protocol_origin_NA_before_native_evaluation',
            terminal_edge_correction='actual_answers_preserved_including_empty_and_last_turn')
    if selected_representation == REPRESENTATION_V2:
        definition.update(piotr_representation=selected_representation,
            piotr_target_policy='all_five_native_slots_with_preaction_availability_status_no_coordinate_invention',
            piotr_unknown_navigation='operation_only_destination_state_not_observed')
    if no_search_prompt == ENVELOPE_NO_SEARCH_PROMPT:
        definition.update(identity='webarena_table5_viewer_reconstruction_v4',
            no_search_prompt_version=no_search_prompt,
            no_search_origin='one_action_reconstruction_with_explicit_BLOCKS_envelope_and_consistent_positive_example')
    if scoring_protocol == ABSTENTION_EVALUATOR_PROTOCOL:
        definition.update(evaluator_protocol=scoring_protocol,
            evaluator_amendment='bare_native_normalized_na_cannot_satisfy_ordinary_fuzzy_string_references_without_explicit_na',
            native_verdict_preservation='unchanged_native_calls_raw_verdict_score_and_component_scores_retained',
            unaffected_evaluators='native_unachievable_reference_ua_url_and_dom_paths_unchanged')
    if helper_configuration is not None:
        from .webprm_webarena_helper_runtime import HELPER_CONFIGURATION
        if helper_configuration != HELPER_CONFIGURATION:
            raise EvaluationNotReady('Unknown explicit native helper protocol configuration')
        definition.update(native_helpers=deepcopy(helper_configuration),
            helper_binding='claimed_original_site_endpoint_reset_and_task_identity',
            helper_semantics='pinned_native_bodies_with_exact_expression_dispatch_and_explicit_owned_deployment')
    if site_configuration is not None:
        from .webprm_webarena_sites import validate_site_configuration
        validate_site_configuration(site_configuration)
        definition.update(site_runtime=deepcopy(site_configuration),
            site_ownership='original_provenance_live_controller_service_namespace_and_fresh_private_state_claim',
            map_authentication='released_unauthenticated_tasks_receive_storage_state_None')
    if observation_configuration is not None:
        from .webprm_evaluator_transcript import validate_transcript_configuration
        validate_transcript_configuration(observation_configuration)
        definition.update(evaluator_observations=deepcopy(observation_configuration),
            evaluator_replay_scope='native_computation_from_observed_page_inputs_and_shopping_helper_return_boundary',
            evaluator_replay_excludes='independent_website_truth_provider_semantics_and_shopping_REST_query_verification')
    if campaign_science is not None:
        if campaign_science != CAMPAIGN_SCIENCE:
            raise EvaluationNotReady('Unknown explicit full-campaign scientific binding')
        definition['campaign_science']=deepcopy(campaign_science)
    return definition



def validate_protocol_contract(config):
    if config.get('viewer_parity_diagnostic') is not None:
        from .webprm_viewer_parity import validate as validate_parity
        validate_parity(config)
    if config.get('reproduction_diagnostic') is not None:
        from .webprm_reproduction_rollout import validate
        validate(config)
    definition = protocol_definition(config['arbiter_source'], turn_limit=turn_limit_protocol(config),
                                     piotr_representation=config.get('piotr_representation'),
                                     no_search_prompt=no_search_prompt_version(config),
                                     scoring_protocol=config.get('evaluator_protocol', NATIVE_EVALUATOR_PROTOCOL),
                                     helper_configuration=config.get('native_helpers'),
                                     site_configuration=config.get('site_runtime'),
                                     observation_configuration=config.get('evaluator_observations'),
                                     campaign_science=config.get('campaign_science'))
    if config.get('campaign_science') is not None and any(config.get(key)!=value
            for key,value in CAMPAIGN_SCIENCE.items() if key!='protocol'):
        raise EvaluationNotReady('Actual seed, turn or native judge settings differ from the campaign protocol')
    if config.get('protocol') != definition['identity'] or config.get('protocol_contract') != definition:
        raise EvaluationNotReady('Frozen viewer protocol/turn-limit contract differs')
    return definition


def episode_evaluator(task, config, request, folder, *, deployment_helpers=None):
    """Inject a metered judge only when the frozen cell explicitly enables it."""
    requirements = evaluation_requirements(task.native_config())
    if requirements['helper_functions_required']:
        from .webprm_webarena_helper_runtime import validate_helper_plan, HELPER_RUNTIME_PROTOCOL
        validate_helper_plan(task, config)
        if (deployment_helpers is None
                or deployment_helpers.manifest.get('runtime_protocol') != HELPER_RUNTIME_PROTOCOL
                or deployment_helpers.manifest.get('state_provenance_sha256') != config.get('reset_id')
                or deployment_helpers.manifest.get('configured_task_sha256') != task.configured_config_sha256):
            raise EvaluationNotReady('Task helper evaluator requires its claimed original-site binding')
    elif deployment_helpers is not None:
        raise EvaluationNotReady('Helper adapter supplied for a task without helper expressions')
    scoring = evaluator_protocol(config.get('evaluator_protocol', NATIVE_EVALUATOR_PROTOCOL))
    recorder = None
    if config.get('evaluator_observations') is not None:
        from .webprm_evaluator_transcript import EvaluationRecorder
        recorder = EvaluationRecorder(Path(folder) / 'native-evaluator-transcript.json',
            config['evaluator_observations'])
    if not requirements['paid_fuzzy_judge_may_be_required']:
        return load_native_evaluator(config['arbiter_source'], scoring_protocol=scoring,
            deployment_helpers=deployment_helpers, observation_recorder=recorder)
    from .webprm_webarena_fuzzy import (BrokerBudgetedTransport, NativeFuzzyJudge,
                                      validate_fuzzy_configuration)
    selected = validate_fuzzy_configuration(config)
    judge = NativeFuzzyJudge(config['arbiter_source'], enabled=True,
        transport=BrokerBudgetedTransport(request, config, Path(folder) / 'requests.jsonl'),
        journal_path=str(Path(folder) / 'native-fuzzy.jsonl'),
        count_input_tokens=lambda payload: input_token_upper_bound(payload['messages']),
        max_input_tokens=config['fuzzy_max_input_tokens'], max_calls=config['fuzzy_max_calls'],
        deadline=config['deadline'], model=selected['model'], protocol=selected['protocol'],
        model_change_approved=selected['model_change_approved'])
    return load_native_evaluator(config['arbiter_source'], fuzzy_judge=judge, scoring_protocol=scoring,
        deployment_helpers=deployment_helpers, observation_recorder=recorder)


class ViewerActor:
    def __init__(self, config, journal, *, request=completion):
        self.parity = self.parity_helpers = None
        if config.get('viewer_parity_diagnostic') is not None:
            from .webprm_viewer_parity import validate as validate_parity, load_viewer_helpers
            self.parity = validate_parity(config)
            self.parity_helpers = load_viewer_helpers(config['arbiter_source'])
        self.parity_observations = []
        if config.get('reproduction_diagnostic') is not None:
            from .webprm_reproduction_rollout import adapt_request
            request = adapt_request(config, request)
        self.config, self.journal, self.request = config, journal, request
        self.harness = load_arbiter_harness(config['arbiter_source'])
        self.raw_slots = 1 if config.get('selector') == 'no_search' else 5
        version = no_search_prompt_version(config)
        self.policy_system_prompt = (single_action_system_prompt(self.harness.policy_system_prompt, version=version)
            if self.raw_slots == 1 else self.harness.policy_system_prompt)
        self.previous_tree = self.previous_terminal = None

    def propose(self, view, observation, history):
        # view is TaskEnvelope.actor_view: evaluation annotations cannot enter.
        context = dict(intent=view['intent'], start_url=view['start_url'],
                       current_url=observation['url'], text_observation=observation['text'],
                       trajectory='\n'.join(f"Thought {i+1}: {row['thought']}\nAction {i+1}: {row['action']}" +
                           (f"\nExecution error: {row['error']}" if row.get('error') else '')
                           for i, row in enumerate(history)))
        if self.parity:
            if self.parity['enrich_history']:
                from .webprm_viewer_parity import history_from_saved_observations
                context['trajectory'] = history_from_saved_observations(
                    self.parity_helpers, history, self.parity_observations)
                # Keep the same factual execution-error feedback in every arm.
                context['trajectory'] += ''.join(
                    f"\nExecution error at step {i}: {row['error']}"
                    for i, row in enumerate(history) if row.get('error'))
            elif len(history) != len(self.parity_observations):
                raise ValueError('One parity observation per actual decision required')
            self.parity_observations.append(deepcopy(observation))
        state_diff, terminal = arbiter_auxiliary(self.harness, self.config, self.journal, context,
            self.previous_tree, self.previous_terminal, observation.get('last_action_error'),
            actor_model=self.config['actor_model'], request=self.request)
        self.previous_tree, self.previous_terminal = observation['text'], terminal
        context['state_diff'] = state_diff
        rejected, raw_response, raw_blocks = [], None, []
        detector_override = bool(terminal and terminal['action'])
        if detector_override:
            candidates = [dict(action=terminal['action'], thought=terminal['thought'], raw_slot=None)]
            self.harness.parse_action(terminal['action'])
        else:
            messages = [dict(role='system', content=self.policy_system_prompt),
                dict(role='user', content=self.harness.render_policy_prompt(
                    objective=view['intent'], observation=observation['text'], url=observation['url'],
                    trajectory=context['trajectory'], state_diff=state_diff, domain_knowledge=None))]
            response = self.request(self.config, dict(model=self.config['actor_model'], messages=messages,
                temperature=1., top_p=.9, n=1, max_tokens=2048,
                response_format={'type': 'json_object'}), self.journal, purpose='policy')
            raw_response = single_response_text(response)
            try:
                raw_blocks = json.loads(raw_response)['BLOCKS']
                if not isinstance(raw_blocks, list):
                    raise ValueError('Native BLOCKS is not a list')
            except (ValueError, TypeError, KeyError):
                raw_blocks = []
            candidates = []
            # Fixed requested slots. Keep native per-slot parser semantics,
            # original order and every rejection; never resample a bad slot.
            for slot in range(self.raw_slots):
                try:
                    if slot >= len(raw_blocks):
                        raise ValueError('Requested raw slot was missing or response malformed')
                    block = raw_blocks[slot]
                    if not isinstance(block, dict) or any(
                            name in block and not isinstance(block[name], str) for name in ('thought', 'action')):
                        raise ValueError('Raw slot must contain string thought/action fields')
                    blocks = self.harness.parse_policy_response(json.dumps({'BLOCKS': [raw_blocks[slot]]})).candidates
                    if len(blocks) != 1 or not isinstance(blocks[0].thought, str) or not isinstance(blocks[0].action, str):
                        raise ValueError('Native parser rejected this raw slot')
                    self.harness.parse_action(blocks[0].action)
                    if self.parity and self.parity['filter_policy_candidates'] and not self.parity_helpers.validate_candidate(
                            blocks[0], self.parity_helpers.extract_elements(observation['text'])):
                        raise ValueError('Pinned viewer candidate validator rejected this original slot')
                    candidates.append(dict(action=blocks[0].action, thought=blocks[0].thought, raw_slot=slot))
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    rejected.append(dict(raw_slot=slot, error_type=type(error).__name__, error=str(error)[:1000]))
            if len(raw_blocks) > self.raw_slots:
                rejected.extend(dict(raw_slot=index, error_type='ExtraRawSlot', error='Outside fixed requested-slot budget')
                                for index in range(self.raw_slots, len(raw_blocks)))
        # `terminal` is the historical selector-bypass flag for a detector
        # override. run_loop decides actual termination from the parsed action,
        # so a native detector click remains an ordinary browser transition.
        result = dict(candidates=candidates, context=context, terminal=detector_override,
                    native_terminal=terminal, candidates_sha256=digest(candidates), raw_response=raw_response,
                    raw_blocks=raw_blocks, raw_slot_count=len(raw_blocks),
                    exact_five_raw_slots=None if detector_override or self.raw_slots != 5 else len(raw_blocks) == 5,
                    exact_requested_raw_slots=None if detector_override else len(raw_blocks) == self.raw_slots,
                    requested_raw_slots=0 if detector_override else self.raw_slots, rejected_slots=rejected,
                    actor_prompt_sha256=hashlib.sha256(self.policy_system_prompt.encode()).hexdigest())
        if self.parity:
            result['viewer_parity_diagnostic'] = deepcopy(self.parity)
        return result


class ViewerSelector:
    def __init__(self, name, config, journal, *, request=completion):
        if name not in SELECTORS:
            raise EvaluationNotReady('Unsupported selector')
        self.name, self.config, self.journal, self.request = name, config, journal, request
        self.checklist = None
        self.shepherd = load_shepherd(config['shepherd_source']) if name == 'shepherd8b' else None
        self.arbiter = load_arbiter(config['arbiter_source']) if name in {'arbiter7b', 'gpt4omini'} else None
        self.reward = load_viewer_reward(config['arbiter_source']) if self.arbiter else None
        self.piotr = None
        if name == 'piotr_selectionarm':
            from .webprm_webarena_piotr import PiotrSelectionARMSelector
            self.piotr = PiotrSelectionARMSelector(config, journal, request=request)

    def historical_action(self, raw, *, viewport, element_bounds):
        if not self.piotr:
            raise EvaluationNotReady('Historical grounded display exists only for the Piotr selector')
        return self.piotr.historical_action(raw, viewport=viewport, element_bounds=element_bounds)

    def select(self, proposal, turn, *, evidence=None):
        pool = deepcopy(proposal['candidates'])
        if proposal['terminal'] or self.name in {'first5', 'no_search'}:
            return 0, []
        if self.name == 'random5':
            return random.Random(self.config['seed'] + turn).randrange(len(pool)), []
        if self.piotr:
            return self.piotr.select(deepcopy(proposal), turn, evidence=evidence)
        context = proposal['context']
        if self.arbiter:
            def compare(left, right):
                # Match RewardGenerator: online viewer supplies current URL as start URL.
                system, user = self.reward.build(intent=context['intent'], observation=context['text_observation'],
                    current_url=context['current_url'], start_url=context['current_url'],
                    trajectory=context['trajectory'], state_diff=context['state_diff'], domain_knowledge=None,
                    candidate_a_thought=left['thought'], candidate_a_action=left['action'],
                    candidate_b_thought=right['thought'], candidate_b_action=right['action'])
                local = self.name == 'arbiter7b'
                response = self.request(self.config, dict(model=self.config['arbiter_model'] if local else ACTOR,
                    messages=[dict(role='system', content=system), dict(role='user', content=user)],
                    temperature=0., top_p=1., max_tokens=2048), self.journal, local=local,
                    endpoint=self.config['arbiter_endpoint'] if local else None, purpose='selector_pair')
                return self.arbiter.parse_answer(single_response_text(response))
            indexed = [dict(candidate, index=index) for index, candidate in enumerate(pool)]
            winner, trace = native_tournament(indexed, compare)
            return winner['index'], trace
        if self.checklist is None:
            messages = self.shepherd.get_messages(context, 'checklist_generation', 'web_shepherd')
            response = self.request(self.config, dict(model=self.config['shepherd_model'], messages=messages,
                temperature=0., max_tokens=1024), self.journal, local=True,
                endpoint=self.config['shepherd_endpoint'], purpose='selector_checklist')
            text = single_response_text(response)
            self.checklist = text.split('<answer>')[-1].split('</answer>')[0].strip() if '<answer>' in text else text.split('[CHECKLISTS]')[-1].strip()
            if not self.checklist:
                raise EvaluationNotReady('Empty generated Shepherd checklist')
        trace = []
        for index, candidate in enumerate(pool):
            messages = self.shepherd.get_messages(dict(context, checklist=self.checklist, **candidate),
                'judge_progress', 'web_shepherd', text_obs=True, image_obs=False)
            response = self.request(self.config, dict(model=self.config['shepherd_model'], messages=messages,
                temperature=0., max_tokens=2048, logprobs=True, top_logprobs=10), self.journal,
                local=True, endpoint=self.config['shepherd_endpoint'], purpose='selector_pointwise')
            text = single_response_text(response)
            tokens = response['choices'][0].get('logprobs', {}).get('content')
            if not tokens or any('top_logprobs' not in token for token in tokens) or (
                    '<answer>' not in text and 'CHECKLIST EVALUATION:' not in text):
                raise EvaluationNotReady('Shepherd label probabilities unavailable')
            probabilities = self.shepherd.get_judge_probs(tokens)
            if not probabilities or any(sum(row.values()) == 0 for row in probabilities):
                raise EvaluationNotReady('Shepherd probability fallback rejected')
            score = self.shepherd.get_score([probabilities], 'prob')[0]
            trace.append(dict(index=index, score=score, probabilities=probabilities))
        return max(range(len(pool)), key=lambda index: (trace[index]['score'], -index)), trace


def run_loop(task, selector_name, config, env, actor, selector, observe, evaluate, folder, *, dispatch_evidence=None):
    """Execute only selected actions; reusable under a network-blocked synthetic env."""
    require_supported(task, selector_name, config)
    limit_protocol = turn_limit_protocol(config)
    folder = Path(folder)
    mark_episode_phase(config, 'browser_reset')
    obs, info = env.reset(seed=config['seed'])
    mark_episode_phase(config, 'native_observation')
    observation = observe(env, obs)
    # Snapshot beside the native bounds, before potentially slow actor inference.
    # Never pair earlier element rectangles with an image taken after selection.
    preaction_screenshot = env.page.screenshot(type='png') if selector_name == 'piotr_selectionarm' else None
    book = NativeTrajectory(observation, {'reset_id': config['reset_id']})
    history, rollout = [], []
    for turn in range(config.get('max_turns', 30)):
        if time.time() >= config['deadline']:
            raise TimeoutError('Episode deadline reached')
        screenshot_path = None
        if selector_name == 'piotr_selectionarm':
            screenshot = preaction_screenshot
            if not isinstance(screenshot, bytes) or not screenshot.startswith(b'\x89PNG\r\n\x1a\n'):
                raise EvaluationNotReady('Piotr selector requires an actual preaction PNG')
            from scripts.check_webprm_reddit_reset import safe_path
            screenshot_path = safe_path(folder / f'preaction-{turn:02d}.png')
            with screenshot_path.open('xb') as handle:
                handle.write(screenshot)
                handle.flush()
                os.fsync(handle.fileno())
        proposal = actor.propose(task.actor_view, deepcopy(observation), deepcopy(history))
        # A malformed selector response or unsupported Piotr input must retain
        # the exact actor output and original raw slots, even before selection.
        write_json(folder / f'proposal-{turn:02d}.json', proposal)
        candidates = deepcopy(proposal['candidates'])
        if selector_name == 'piotr_selectionarm' and not proposal['terminal'] and (
                proposal.get('exact_five_raw_slots') is not True or len(candidates) != 5):
            raise EvaluationNotReady('Piotr requires five original admissible slots; proposal retained without padding or resampling')
        if not candidates:
            error = 'No admissible native action in the fixed requested slots; no action executed or resampled'
            rollout.append(dict(turn=turn, proposal=proposal, selected_index=None, selection_trace=[],
                                execution_error=error, native_dispatch=None))
            history.append(dict(action='[no action executed]', thought='', error=error))
            if selector_name == 'piotr_selectionarm':
                history[-1]['selectionarm_action'] = json.dumps(dict(native_action='[no action executed]',
                    operation='none'), sort_keys=True, separators=(',', ':'))
            observation = dict(observation, last_action_error=error)
            write_json(folder / 'rollout.json', rollout)
            write_json(folder / 'trajectory-progress.json', book.entries)
            continue
        before = digest(proposal)
        selection_evidence = None
        if selector_name == 'piotr_selectionarm' and not proposal['terminal']:
            screenshot = preaction_screenshot
            evidence = dict(screenshot=screenshot, viewport=deepcopy(observation.get('viewport')),
                element_bounds=deepcopy(observation.get('element_bounds')), history=deepcopy(history))
            selection_evidence = dict(screenshot_path=str(screenshot_path),
                screenshot_sha256=hashlib.sha256(screenshot).hexdigest(),
                viewport=evidence['viewport'], element_bounds=evidence['element_bounds'], history_sha256=digest(history))
            chosen, trace = selector.select(deepcopy(proposal), turn, evidence=evidence)
        else:
            chosen, trace = selector.select(deepcopy(proposal), turn)
        if type(chosen) is not int or not 0 <= chosen < len(candidates) or digest(proposal) != before:
            raise EvaluationNotReady('Selector did not return an original candidate index')
        if selector_name == 'piotr_selectionarm' and not proposal['terminal'] and (
                not isinstance(trace, dict) or not isinstance(trace.get('selected_grounded_action'), str)
                or not trace['selected_grounded_action']):
            raise EvaluationNotReady('Piotr selector omitted its frozen historical action display')
        selected = candidates[chosen]
        parsed = actor.harness.parse_action(selected['action'])
        record = dict(turn=turn, proposal=proposal, selected_index=chosen, selection_trace=trace,
                      selection_evidence=selection_evidence)
        rollout.append(record)
        write_json(folder / 'rollout.json', rollout)
        mark_episode_phase(config, 'browser_step', turn=turn)
        previous_messages = len(obs.get('chat_messages', []))
        if dispatch_evidence is not None:
            dispatch_evidence.clear()
        obs, _, terminated, truncated, info = env.step(selected['action'])
        record['native_dispatch'] = deepcopy(dispatch_evidence) if dispatch_evidence is not None else None
        error = obs.get('last_action_error')
        record['execution_error'] = error
        history.append(dict(selected, error=error))
        if selector_name == 'piotr_selectionarm' and not proposal['terminal']:
            history[-1]['selectionarm_action'] = trace['selected_grounded_action']
        elif selector_name == 'piotr_selectionarm' and parsed['kind'] != 'send_message_to_user':
            # A terminal-detector fallback executed a native action without a Piotr
            # call; ground it from this turn's pre-action observation (job 356527).
            history[-1]['selectionarm_action'] = selector.historical_action(selected['action'],
                viewport=observation.get('viewport'), element_bounds=observation.get('element_bounds'))
        if parsed['kind'] == 'send_message_to_user':
            answer = parsed['args'][0]
            actual = obs.get('chat_messages', [])[previous_messages:]
            if error or not any(row.get('role') == 'assistant' and row.get('message') == answer for row in actual):
                raise EvaluationNotReady('Terminal action did not execute with its actual literal answer')
            book.record_executed_terminal({'action_type': 'send_msg_to_user', 'answer': answer,
                                           'raw_action': selected['action']})
        else:
            mark_episode_phase(config, 'native_observation', turn=turn)
            observation = observe(env, obs)
            preaction_screenshot = env.page.screenshot(type='png') if selector_name == 'piotr_selectionarm' else None
            book.record_executed_action(dict(action_type=parsed['kind'], raw_action=selected['action'],
                args=list(parsed['args']), kwargs=parsed['kwargs']), observation,
                {'execution_error': error, 'native_dispatch': record['native_dispatch']})
            if terminated or truncated:
                book.finish_at_state('turn_limit' if truncated else 'stop')
        write_json(folder / 'rollout.json', rollout)
        write_json(folder / 'trajectory-progress.json', book.entries)
        if book.termination:
            break
    fallback = None
    if book.termination is None:
        if limit_protocol == NA_TURN_LIMIT:
            # Only normal exhaustion reaches here. Early environment stop,
            # exceptions and actor-selected terminals leave through other paths.
            # Recoverable per-action errors remain actor-visible and do not
            # change normal exhaustion into an infrastructure failure.
            if time.time() >= config['deadline']:
                raise TimeoutError('Episode deadline reached before turn-limit fallback')
            action = 'send_msg_to_user [N/A]'
            fallback = dict(schema_version=1, protocol=limit_protocol,
                origin='released_viewer_normal_turn_limit', termination_reason='turn_limit',
                action=action, answer='N/A', actor_selected=False, actor_turns=len(rollout),
                pre_fallback_trajectory_sha256=digest(book.entries),
                source_revision=ARBITER_REVISION,
                source_file='viewer/src/server/service/executors/step_loop_executor.py',
                status='dispatch_pending')
            write_json(folder / 'protocol-fallback.json', fallback)
            mark_episode_phase(config, 'browser_step', turn=len(rollout), protocol_fallback=True)
            previous_messages = deepcopy(obs.get('chat_messages', []))
            fallback['chat_messages_before'] = previous_messages
            if dispatch_evidence is not None:
                dispatch_evidence.clear()
            try:
                final_obs, _, _, _, _ = env.step(action)
                actual = final_obs.get('chat_messages', [])
                if (final_obs.get('last_action_error') or
                        actual[:len(previous_messages)] != previous_messages or
                        not any(row.get('role') == 'assistant' and row.get('message') == 'N/A'
                                for row in actual[len(previous_messages):])):
                    raise EvaluationNotReady('Protocol fallback did not append its actual literal answer')
                fallback.update(status='executed', native_dispatch=deepcopy(dispatch_evidence),
                    appended_messages=deepcopy(actual[len(previous_messages):]),
                    chat_messages_after=deepcopy(actual))
                book.record_executed_terminal(dict(action_type='send_msg_to_user', answer='N/A',
                    raw_action=action, origin=fallback['origin'], protocol=limit_protocol))
                write_json(folder / 'trajectory-progress.json', book.entries)
            except Exception as error:
                fallback.update(status='failed', error_type=type(error).__name__, error=str(error),
                    native_dispatch=deepcopy(dispatch_evidence))
                write_json(folder / 'protocol-fallback.json', fallback)
                raise
            write_json(folder / 'protocol-fallback.json', fallback)
        else:
            book.finish_at_state('turn_limit')
    result = dict(task_id=task.execution_id, released_task_id=task.released_task_id,
                  arm=config['cell_name'], selector=selector_name, actor_model=config['actor_model'],
                  identity_sha256=config['identity_sha256'], attempted=True, turns=len(rollout),
                  reset_id=config['reset_id'], valid=False, success=0,
                  turn_limit_protocol=limit_protocol, protocol_fallback=fallback)
    def durable_before_close(native):
        result.update(native, success=int(native['success']), arm=config['cell_name'])
        write_json(folder / 'result.json', result)
        mark_episode_phase(config, 'cleanup', cleanup_started_at=time.time())
    mark_episode_phase(config, 'native_evaluation')
    native = score_and_close(task, book, env, folder / 'native-evaluation',
        hooks=EpisodeHooks(snapshot=lambda page: dict(url=page.url, screenshot=page.screenshot(type='png')),
                           evaluate=evaluate), before_close=durable_before_close)
    result.update(native, success=int(native['success']), arm=config['cell_name'])
    write_json(folder / 'result.json', result)
    mark_episode_phase(config, 'completed', cleanup_started_at=time.time())
    return result


def broker_request(config, task, selector):
    """Only the host broker can cross from the isolated website to model routes."""
    if not config.get('broker_socket'):
        raise EvaluationNotReady('An owned host-namespace model broker is required')
    from .webprm_broker import make_broker_client
    client = make_broker_client(config['broker_socket'],
        cell_id=task.execution_id + '/' + cell_name(config['actor_model'], selector),
        identity_sha256=config['identity_sha256'], deadline=config['deadline'])
    def request(call_config, payload, journal, **kwargs):
        # The host broker cannot update this spawned child's phase token.
        phase = ('judge_request' if kwargs.get('purpose') == 'native_fuzzy'
                 else 'critic_request' if kwargs.get('local') else 'actor_request')
        mark_episode_phase(call_config, phase,
                           purpose=kwargs.get('purpose'), model=payload['model'])
        return client(call_config, payload, journal, **kwargs)
    request._webprm_broker_client = getattr(client, '_webprm_broker_client', False)
    request._webprm_budget_owner = getattr(client, '_webprm_budget_owner', None)
    return request


def claim_lease(task, selector, config):
    """A canonical service-instance claim prevents cross-cell reset reuse."""
    from scripts.check_webprm_reddit_reset import safe_path, write_new
    from .webprm_webarena_sites import site_adapter, validate_site_instance, canonical_runtime_path
    adapter = site_adapter(task, config)
    path = safe_path(config['lease_path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['lease_sha256']:
        raise EvaluationNotReady('Episode lease changed')
    lease = json.loads(raw)
    expected = dict(task_id=task.execution_id, actor_model=config['actor_model'], selector=selector)
    if any(lease.get(key) != value for key, value in expected.items()):
        raise EvaluationNotReady('Episode lease assigned to a different cell')
    instance_path = safe_path(lease['instance_receipt'])
    raw_instance = instance_path.read_bytes()
    if hashlib.sha256(raw_instance).hexdigest() != lease['instance_receipt_sha256']:
        raise EvaluationNotReady('Instance receipt changed after lease assignment')
    instance = json.loads(raw_instance)
    validate_site_instance(task, config, instance)
    reset_id = instance['state_provenance_sha256']
    # Fixed original provenance requires this receipt's canonical state path.
    # Claim on the actual state, rather than a copyable lease JSON pathname.
    claim = safe_path(instance['state_path']) / '.webprm-episode-claim.json'
    claim_value = dict(**expected, reset_id=reset_id, lease_sha256=config['lease_sha256'])
    if not adapter.legacy_reddit:
        canonical_runtime_path(config['lease_path'])
        canonical_runtime_path(lease['instance_receipt'])
        claim_value.update(configured_task_sha256=task.configured_config_sha256,
                           site_runtime=config['site_runtime'])
    write_new(claim, claim_value)
    if not adapter.legacy_reddit:
        instance = dict(instance, episode_claim=claim_value)
    return instance, reset_id


def authenticated_state(source, instance, chromium, output):
    """Use released Reddit UI credentials/selectors on each fresh instance."""
    from scripts.check_webprm_reddit_reset import load_auth, write_new
    from playwright.sync_api import sync_playwright
    account, hashes = load_auth(Path(source))
    endpoint = instance['endpoint'].rstrip('/')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chromium, headless=True)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(10000)
            response = page.goto(endpoint + '/login', wait_until='domcontentloaded', timeout=15000)
            if response is None or response.status != 200:
                raise EvaluationNotReady('Native login page unavailable')
            page.get_by_label('Username').fill(account['username'])
            page.get_by_label('Password').fill(account['password'])
            page.get_by_role('button', name='Log in', exact=True).click()
            page.wait_for_url(lambda value: urlsplit(str(value)).path != '/login')
            target = endpoint + '/user/' + quote(account['username']) + '/account'
            response = page.goto(target, wait_until='domcontentloaded', timeout=15000)
            if response is None or response.status != 200 or page.url.rstrip('/') != target:
                raise EvaluationNotReady('Native authenticated account access failed')
            state = context.storage_state()
            write_new(Path(output) / 'storage-state-private.json', state, private=True)
            write_json(Path(output) / 'authentication.json', dict(authenticated=True,
                auth_source_hashes=hashes, state_provenance_sha256=instance['state_provenance_sha256']))
            return state
        except Exception as error:
            message = str(error)
            for value in account.values():
                message = message.replace(value, '[REDACTED]')
            raise EvaluationNotReady('Native authentication failed: ' + message[:2000]) from None
        finally:
            browser.close()


def run_episode(task_record, selector, config):
    validate_execution_config(dict(config, selector=selector))
    task = TaskEnvelope.from_release(task_record['released'], config['endpoints'])
    if task_record.get('task_id') != task.execution_id:
        raise EvaluationNotReady('Execution directory ID differs from released task identity')
    require_supported(task, selector, config)
    validate_frozen_sources(config)
    cell = cell_name(config['actor_model'], selector)
    from scripts.check_webprm_reddit_reset import safe_path
    folder = Path(config['output']) / task.execution_id / cell
    for name in ['result.json', 'phase.json', 'lifecycle.jsonl', 'watchdog.json', 'rollout.json']:
        path = folder / name
        for candidate in (path, path.with_suffix(path.suffix + '.tmp')):
            safe_path(candidate)
            if candidate.is_symlink():
                raise ValueError('Refusing a lifecycle artifact symlink')
    return run_supervised_episode(task_record, cell, dict(config, selector=selector, cell_name=cell), _run_episode)


def _run_episode(task_record, cell, config):
    task = TaskEnvelope.from_release(task_record['released'], config['endpoints'])
    require_supported(task, config['selector'], config)
    folder = Path(config['output']) / task.execution_id / cell
    folder.mkdir(parents=True, exist_ok=True)
    env = None
    result = dict(task_id=task.execution_id, arm=cell, valid=False, success=0, attempted=False,
                  actor_model=config['actor_model'], selector=config['selector'],
                  identity_sha256=config['identity_sha256'])
    try:
        if config.get('authorization_status') != 'approved' or config.get('api_usd_cap', 0) <= 0:
            raise EvaluationNotReady('Real actor calls require an explicit positive approved API budget')
        for path, expected in config['code_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise EvaluationNotReady('Frozen worker code changed')
        request = broker_request(config, task, config['selector'])
        instance, reset_id = claim_lease(task, config['selector'], config)
        config = dict(config, reset_id=reset_id)
        prepare_browser()
        result['attempted'] = True
        mark_episode_phase(config, 'native_authentication')
        from .webprm_webarena_sites import site_authentication
        state, authentication_proof = site_authentication(task, config, instance,
            config.get('chromium', CHROMIUM), folder, reddit_authenticate=authenticated_state)
        actor = ViewerActor(config, folder / 'requests.jsonl', request=request)
        selector = ViewerSelector(config['selector'], config, folder / 'requests.jsonl', request=request)
        bounds, dispatch_evidence = {}, {}
        def mapping(action):
            code = native_execution_code(config['arbiter_source'], actor.harness.parse_action(action),
                                         bounds, key_mapping=actor.harness.key_mapping)
            dispatch_evidence.update(action=action, code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                                     element_bounds=deepcopy(bounds))
            return code
        from browsergym.core.env import BrowserEnv
        from .webprm_runtime import live_task_type
        # BrowserGym creates env.task only during reset. Pin the same geometry
        # for construction and native observation without an extra reset.
        viewport = {'width': 1280, 'height': 720}
        env = open_authenticated_browser(task,
            hooks=EpisodeHooks(dependency=lambda _: authentication_proof),
            browser_factory=BrowserEnv, task_entrypoint=live_task_type(), action_mapping=mapping,
            storage_state=state, task_options=dict(deadline=config['deadline']),
            browser_options=dict(headless=True, slow_mo=0, viewport=deepcopy(viewport),
                pw_chromium_kwargs={'executable_path': config.get('chromium', CHROMIUM)}))
        native_observation = load_native_observation_processor(config['arbiter_source'], viewport_size=viewport)
        def observe(environment, observation):
            text, observed = native_observation.observe(environment.page)
            bounds.clear(); bounds.update(observed)
            return dict(text=text, url=environment.page.url, last_action_error=observation.get('last_action_error'),
                        element_bounds=deepcopy(bounds), viewport=dict(environment.page.viewport_size))
        helpers = None
        if evaluation_requirements(task.native_config())['helper_functions_required']:
            from .webprm_webarena_helper_runtime import owned_native_helpers
            helpers = owned_native_helpers(task, config, instance)
        evaluator = episode_evaluator(task, config, request, folder, deployment_helpers=helpers)
        write_json(folder / 'provenance.json', dict(actor=actor.harness.manifest,
            observation=native_observation.manifest,
            critic=selector.reward.manifest if selector.reward else selector.shepherd.manifest if selector.shepherd
                else selector.piotr.manifest if selector.piotr else None,
            critic_template='pinned_online_viewer' if selector.reward else config['selector'],
            original_instance=instance, reset_id=reset_id, actor_model=config['actor_model'],
            authentication=authentication_proof,
            reproduction_diagnostic=config.get('reproduction_diagnostic'),
            deployment_helpers=helpers.manifest if helpers else None,
            paper_exact_protocol=False, identity_sha256=config['identity_sha256']))
        result = run_loop(task, config['selector'], config, env, actor, selector, observe, evaluator.evaluate, folder,
                          dispatch_evidence=dispatch_evidence)
        env = None  # score_and_close already owns normal cleanup.
    except Exception as error:
        # Prefer the valid outcome persisted before a failed close, if present.
        if (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text())
        result.update(error_type=type(error).__name__, error=str(error))
    finally:
        if env is not None or result.get('cleanup_status') is None:
            save_outcome_and_close(result, folder, env, config)
    return result
