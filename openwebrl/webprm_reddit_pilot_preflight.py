"""Prospective startup contract checks through the owned, metered broker.

Importing builds/loads no model. The caller may execute this only inside the
approved allocation, after the three servers have reported their fixed IDs.
"""
import hashlib
import io
import json
import os
from pathlib import Path


NATIVE_FUZZY_PREFLIGHT = dict(
    protocol='synthetic_native_fuzzy_access_v1', cell_id='native_fuzzy_preflight',
    model='gpt-4o-2024-05-13', purpose='native_fuzzy', max_calls=1,
    expected_score=1, before_primary_admission=True, automatic_retry=False)
NATIVE_FUZZY_FIXTURE = dict(question='Which synthetic shape and color are displayed?',
    reference='blue triangle', prediction='blue triangle', expected_score=1)
GUARDED_FUZZY_PREFLIGHT = dict(
    protocol='synthetic_native_fuzzy_with_na_guard_v1', cell_id='native_fuzzy_preflight',
    model='gpt-4o-2024-05-13', purpose='native_fuzzy', max_calls=8,
    before_primary_admission=True, automatic_retry=False,
    scope='fixed_positive_negative_and_abstention_controls_not_global_judge_accuracy')
GUARDED_FUZZY_FIXTURES = (
    dict(question='Which synthetic shape and color are displayed?', reference='blue triangle', prediction='blue triangle', expected_score=1),
    dict(question='Which synthetic shape and color are displayed?', reference='blue triangle', prediction='red square', expected_score=0),
    *[dict(question='How many qualifying comments are in the synthetic fixture?', reference='0',
        prediction=prediction, expected_score=score) for prediction, score in (
            ('0', 1), ('1', 0), ('N/A', 0), ('n/a', 0), ('zero', 1), ('There are no qualifying comments.', 1))])


def _approved_fuzzy(config):
    from .webprm_webarena import EvaluationNotReady
    from .webprm_webarena_fuzzy import validate_fuzzy_configuration
    if config.get('authorization_status') != 'approved':
        raise EvaluationNotReady('Native fuzzy preflight requires explicit runtime approval')
    selected = validate_fuzzy_configuration(config)
    if selected['model'] != NATIVE_FUZZY_PREFLIGHT['model']:
        raise EvaluationNotReady('Native fuzzy preflight requires the frozen replacement judge')
    return selected


def native_fuzzy_preflight_cell(config, env_file, output):
    """Register a paid judge-only cell; the host broker alone reads credentials."""
    selected = _approved_fuzzy(config)
    return dict(config=dict(config, env_file=str(env_file)),
                journal=str(Path(output) / 'native-fuzzy-preflight-api.jsonl'),
                allowed_routes=[[selected['model'], 'native_fuzzy']])


def _save_preflight(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_native_fuzzy(config, request, output):
    """One synthetic known-answer call, never a benchmark-quality validation.

    A later allocation may repeat this gate only through the same campaign
    counters and monetary ledger. This function never retries a request.
    """
    from .webprm_webarena import EvaluationNotReady
    from .webprm_webarena_fuzzy import BrokerBudgetedTransport, NativeFuzzyJudge
    from .webprm_worker import input_token_upper_bound
    selected = _approved_fuzzy(config)
    folder = Path(output) / 'native-fuzzy-preflight'
    folder.mkdir(mode=0o700, exist_ok=False)
    fixture = dict(NATIVE_FUZZY_FIXTURE)
    scope = 'Synthetic inference access and native verdict parsing only; no benchmark-quality claim'
    _save_preflight(folder / 'input.json', dict(scope=scope, synthetic=True,
        protocol=NATIVE_FUZZY_PREFLIGHT, fixture=fixture,
        identity_sha256=config['identity_sha256'], primary_admitted=False))
    judge = None
    try:
        judge = NativeFuzzyJudge(config['arbiter_source'], enabled=True,
            transport=BrokerBudgetedTransport(request, config, folder / 'requests.jsonl'),
            journal_path=folder / 'native-fuzzy.jsonl',
            count_input_tokens=lambda payload: input_token_upper_bound(payload['messages']),
            max_input_tokens=config['fuzzy_max_input_tokens'], max_calls=1,
            deadline=config['deadline'], model=selected['model'], protocol=selected['protocol'],
            model_change_approved=selected['model_change_approved'])
        score = judge.fuzzy_match(fixture['prediction'], fixture['reference'], fixture['question'])
        if score != fixture['expected_score'] or judge.provider_calls != 1:
            raise EvaluationNotReady('Synthetic native fuzzy preflight expected verdict failed')
        proof = dict(status='passed', scope=scope, synthetic=True, primary_admitted=False,
            score=score, expected_score=fixture['expected_score'], provider_calls=judge.provider_calls,
            budget_owner='webprm_worker.completion', automatic_retry=False,
            manifest=judge.manifest,
            input_sha256=hashlib.sha256((folder / 'input.json').read_bytes()).hexdigest(),
            native_journal_sha256=hashlib.sha256((folder / 'native-fuzzy.jsonl').read_bytes()).hexdigest())
    except Exception as error:
        _save_preflight(folder / 'failed.json', dict(status='failed', scope=scope,
            primary_admitted=False, automatic_retry=False, error_type=type(error).__name__,
            provider_calls=judge.provider_calls if judge else 0))
        raise
    _save_preflight(folder / 'passed.json', proof)
    return proof


def validate_guarded_fuzzy(config, request, output):
    """Eight prespecified controls through the actual string evaluator path.

    Raw native verdicts remain durable, including any N/A false positives.
    Neither failing controls nor primary episodes are retried here.
    """
    from .webprm_webarena import (EvaluationNotReady, load_native_evaluator,
        ABSTENTION_EVALUATOR_PROTOCOL)
    from .webprm_webarena_fuzzy import BrokerBudgetedTransport, NativeFuzzyJudge
    from .webprm_worker import input_token_upper_bound
    selected = _approved_fuzzy(config)
    if config.get('evaluator_protocol') != ABSTENTION_EVALUATOR_PROTOCOL:
        raise EvaluationNotReady('Guarded preflight requires the explicit prospective scoring protocol')
    folder = Path(output) / 'native-fuzzy-preflight'
    folder.mkdir(mode=0o700, exist_ok=False)
    _save_preflight(folder / 'input.json', dict(synthetic=True,
        protocol=GUARDED_FUZZY_PREFLIGHT, fixtures=GUARDED_FUZZY_FIXTURES,
        identity_sha256=config['identity_sha256'], primary_admitted=False))
    judge = NativeFuzzyJudge(config['arbiter_source'], enabled=True,
        transport=BrokerBudgetedTransport(request, config, folder / 'requests.jsonl'),
        journal_path=folder / 'native-fuzzy.jsonl',
        count_input_tokens=lambda payload: input_token_upper_bound(payload['messages']),
        max_input_tokens=config['fuzzy_max_input_tokens'], max_calls=8,
        deadline=config['deadline'], model=selected['model'], protocol=selected['protocol'],
        model_change_approved=selected['model_change_approved'])
    evaluator = load_native_evaluator(config['arbiter_source'], fuzzy_judge=judge,
        scoring_protocol=ABSTENTION_EVALUATOR_PROTOCOL)
    results = []
    try:
        for index, fixture in enumerate(GUARDED_FUZZY_FIXTURES):
            task = dict(intent=fixture['question'], eval=dict(eval_types=['string_match'],
                reference_answers=dict(fuzzy_match=[fixture['reference']])))
            trajectory = [dict(observation=dict(text='Synthetic fixture'), info={}),
                dict(action_type='send_msg_to_user', answer=fixture['prediction'])]
            result = evaluator.evaluate(task, trajectory)
            _save_preflight(folder / f'control-{index}.json', dict(fixture=fixture, result=result))
            results.append(result)
        if judge.provider_calls != 8 or any(result['score'] != fixture['expected_score']
                for result, fixture in zip(results, GUARDED_FUZZY_FIXTURES)):
            raise EvaluationNotReady('Prespecified positive/negative guarded scoring controls failed')
    except Exception as error:
        _save_preflight(folder / 'failed.json', dict(status='failed', synthetic=True,
            primary_admitted=False, automatic_retry=False, error_type=type(error).__name__,
            provider_calls=judge.provider_calls, completed_controls=len(results)))
        raise
    proof = dict(status='passed', synthetic=True, primary_admitted=False,
        protocol=GUARDED_FUZZY_PREFLIGHT, provider_calls=judge.provider_calls,
        native_positive_count=sum(result['native_success'] for result in results),
        effective_positive_count=sum(result['success'] for result in results),
        corrected_count=sum(bool(result['scoring_amendments']) for result in results),
        manifest=judge.manifest, budget_owner='webprm_worker.completion', automatic_retry=False,
        input_sha256=hashlib.sha256((folder / 'input.json').read_bytes()).hexdigest(),
        native_journal_sha256=hashlib.sha256((folder / 'native-fuzzy.jsonl').read_bytes()).hexdigest())
    _save_preflight(folder / 'passed.json', proof)
    return proof


def validate_local_critics(config, request, output):
    from PIL import Image, ImageDraw
    from .webprm_webarena_worker import ViewerSelector
    image=Image.new('RGB',(1280,720),'white');draw=ImageDraw.Draw(image)
    bounds={}
    for i in range(1,6):
        x,y,w,h=40,40+i*90,260,60;bounds[str(i)]=[x,y,w,h]
        draw.rectangle((x,y,x+w,y+h),outline='black');draw.text((x+10,y+20),f'Choice {i}',fill='black')
    stream=io.BytesIO();image.save(stream,format='PNG');png=stream.getvalue()
    proposal=dict(terminal=False,exact_five_raw_slots=True,raw_slot_count=5,
        candidates=[dict(action=f'click [{i}]',thought=f'Select choice {i}',raw_slot=i-1) for i in range(1,6)],
        context=dict(intent='Select Choice1',start_url='about:blank',current_url='about:blank',
            text_observation='\n'.join(f'[{i}] button Choice {i}' for i in range(1,6)),trajectory='',state_diff=''))
    evidence=dict(screenshot=png,viewport={'width':1280,'height':720},element_bounds=bounds,history=[])
    results={}
    for name in ('shepherd8b','arbiter7b','piotr_selectionarm'):
        selector=ViewerSelector(name,config,Path(output)/'local-preflight-unused-client-path.jsonl',request=request)
        if name=='piotr_selectionarm':index,trace=selector.select(proposal,0,evidence=evidence)
        else:index,trace=selector.select(proposal,0)
        if type(index)is not int or not 0<=index<5:raise ValueError('Local critic did not return original candidate index')
        results[name]=dict(selected_index=index,trace=trace)
    return dict(status='passed',scope='Local output/parser/vision-request contract only; no task-success claim',
        provider_calls=0,expected_local_dispatches=11,screenshot_sha256=hashlib.sha256(png).hexdigest(),
        original_candidate_count=5,results=results)
