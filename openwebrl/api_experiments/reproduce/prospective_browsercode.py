"""Explicit, source-bound Luna effort/output-cap sweep in a new runtime only.

The archived campaign is never edited. Each generated runtime retains its base
inventory and changed original files, then freezes one 0.5x/1x/2x/4x arm over the
same public OM2W300, WebVoyager595 and DeepShop150 rosters. This is prospective
experiment preparation, not permission to execute browser or provider work.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
CAMPAIGN = 'browsercode_luna_effort'
ARM = 'luna_single_xhigh'
BASE_OUTPUT_TOKENS = 12288
CAPS = (6144, 12288, 24576, 49152)
EFFORT_ARMS = {'xhigh': ARM, 'high': 'luna_single_high'}
SCHEMA = 'browsercode-luna-single-output-sweep-v2'
RECEIPT = 'prospective-experiment.json'
ORIGINALS = 'prospective-originals'
COUNTS = {'online-mind2web': 300, 'webvoyager': 595, 'deepshop': 150}
MODEL_SHA = '804ab88145912a5ebd4056cc7c17f67432f27225880de300fc0f23b51ffb5f54'
CHANGED = ('luna_model.py', 'worker.py', 'campaign-plan.json', 'runtime-config.json', 'tasks.json', 'portable-build.json')


def need(condition, message):
    if not condition:
        raise ValueError('Prospective sweep: ' + message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def output_multiplier(cap):
    """Keep whole-number historical ratios and represent the half budget exactly."""
    quotient, remainder = divmod(cap, BASE_OUTPUT_TOKENS)
    return cap / BASE_OUTPUT_TOKENS if remainder else quotient


def task_roster():
    from tasks import load_tasks, canonical_benchmark
    rows = load_tasks(CAMPAIGN) + load_tasks('browsercode_gpt_high_wv_ds')
    for row in rows:
        row['benchmark'] = canonical_benchmark(row['benchmark'])
    need(dict(Counter(row['benchmark'] for row in rows)) == COUNTS, 'public roster counts changed')
    need(len({row['task_id'] for row in rows}) == 1045, 'public task identities must be unique')
    return rows


def adapt_model(text, effort='xhigh', max_decisions=30):
    need(type(max_decisions) is int and max_decisions in (30, 60, 120), 'choose exactly 30, 60 or 120 decisions')
    need(effort in EFFORT_ARMS, 'choose high or xhigh reasoning')
    need(hashlib.sha256(text.encode()).hexdigest() == MODEL_SHA, 'archived Luna model source changed')
    head, body = text.split('class LunaModel(OpenAIModel):\n', 1)
    if effort == 'high':
        head = head.replace('PRICING=', "MODES['luna_single_high']=(1,'high',None)\nPRICING=")
    body = body.replace('MAX_OUTPUT', 'self.output_token_cap')
    turn_guard = "need(self.turn<30,'Frozen30-decision cap')"
    need(body.count(turn_guard) == 1, 'Luna actual turn guard changed')
    body = body.replace(turn_guard, "need(self.turn<" + str(max_decisions) + ",'Prospective decision cap')")

    anchor = " def __init__(self,mode,directory,client,budget,pricing=None,harness='browsercode',http_gate=None,**kwargs):\n"
    need(body.count(anchor) == 1, 'Luna constructor anchor changed')
    declaration = (
        "  declared=json.loads((Path(__file__).resolve().parent/'campaign-plan.json').read_text());arms=declared.get('arms',[])\n"
        "  need(declared.get('prospective_protocol')=='" + SCHEMA + "' and len(arms)==1,'Explicit prospective Luna plan required');arm=arms[0]\n"
        "  need(mode=='" + EFFORT_ARMS[effort] + "' and arm.get('arm')==mode and arm.get('model')==MODEL and arm.get('proposal_count')==1 and arm.get('reasoning_effort')=='" + effort + "' and arm.get('selector_model')is None,'Prospective single effort arm differs')\n"
        "  self.output_token_cap=arm.get('output_tokens');need(type(self.output_token_cap)is int and self.output_token_cap in (6144,12288,24576,49152) and declared.get('protocol',{}).get('max_decisions')==" + str(max_decisions) + ",'Prospective output/turn cap differs')\n"
    )
    if max_decisions == 120:
        declaration += "  need(arm.get('output_tokens')==12288 and arm.get('reasoning_effort')=='high' and declared['protocol'].get('prompt_max_decisions')==120,'Single120 requires high12288 and an explicit120 prompt')\n"
    adapted = head + 'class LunaModel(OpenAIModel):\n' + body.replace(anchor, anchor + declaration)
    compile(adapted, '<prospective Luna model>', 'exec')
    return adapted


def adapt_worker(text, max_decisions=30):
    need(type(max_decisions) is int and max_decisions in (30, 60, 120), 'choose exactly 30, 60 or 120 decisions')
    replacements = {
        'max_output_tokens=12288': "max_output_tokens=next(x['output_tokens']for x in PLAN['arms']if x['arm']==mode)",
        "'max_output_tokens':12288": "'max_output_tokens':next(x['output_tokens']for x in PLAN['arms']if x['arm']==a.arm)",
        "row={'task_id':item['task_id'],": "row={'task_id':item['task_id'],'benchmark':item['benchmark'],'actor_reasoning_effort':MODES[mode][1],'actor_max_output_tokens':next(x['output_tokens']for x in PLAN['arms']if x['arm']==mode),",
    }
    replacements.update({
        "'step_cap':30": "'step_cap':PLAN['protocol']['max_decisions']",
        'step_limit=30': "step_limit=PLAN['protocol']['max_decisions']",
        "'max_decisions':30": "'max_decisions':PLAN['protocol']['max_decisions']",
    })
    for old, new in replacements.items():
        need(text.count(old) == 1, 'worker cap/tracking anchor changed: ' + old)
        text = text.replace(old, new)
    compile(text, '<prospective Luna worker>', 'exec')
    return text


def derived_plan(base, cap, tasks_sha, effort='xhigh', max_decisions=30):
    arm = deepcopy(next(row for row in base['arms'] if row['arm'] == ARM))
    arm.update(arm=EFFORT_ARMS[effort], reasoning_effort=effort, output_tokens=cap)
    protocol = deepcopy(base['protocol'])
    for key in ('same300_tasks_allarms', 'pilot20_tasks_per_arm_included_final', 'historical_high_results_judge'):
        protocol.pop(key, None)
    protocol.update(actor_max_output_tokens=cap, selector_max_output_tokens=None, max_decisions=max_decisions)
    if max_decisions == 120:
        protocol['prompt_max_decisions'] = 120
    return {'schema': 'portable-scientific-campaign-v1', 'campaign': 'luna_reasoning_token_sweep',
            'prospective_protocol': SCHEMA, 'protocol': protocol, 'arms': [arm],
            'cells': [{'harness': 'browsercode', 'policy': EFFORT_ARMS[effort]}],
            'benchmark_counts': COUNTS, 'task_manifest_sha256': tasks_sha,
            'task_order': 'original Luna OM2W roster followed by original GPT crossbench roster',
            'output_multiplier': output_multiplier(cap)}


def derived_config(base, max_decisions):
    """Preserve historical30/60 prompts; advertise120 only for the new arm."""
    config = deepcopy(base)
    config['prospective_protocol'] = SCHEMA
    if max_decisions == 120:
        old = 'Only30 model decisions are available.'
        need(config['system_templates']['browsercode'].count(old) == 1, 'original decision prompt changed')
        config['system_templates']['browsercode'] = config['system_templates']['browsercode'].replace(old, 'Only120 model decisions are available.')
        config['prompt_max_decisions'] = 120
    return config


def validate_single120(cap, effort, max_decisions):
    need(max_decisions != 120 or (type(cap) is int and cap == 12288 and effort == 'high'), 'single120 requires high reasoning and12288 output tokens')


def verify_inventory(runtime, manifest):
    need(manifest['plan_sha256'] == digest(runtime / 'campaign-plan.json'), 'plan hash changed')
    for name, expected in manifest['source_hashes'].items():
        need(digest(name) == expected, 'source changed: ' + name)


def configure(runtime, cap, effort='xhigh', max_decisions=30):
    """Apply once, after ordinary public setup and its BrowserCode-only check."""
    runtime = Path(runtime).resolve()
    need(not runtime.is_relative_to(Path('/gpfs/home/zixianma')), 'home is read-only')
    need(type(cap) is int and cap in CAPS, 'choose exactly 6144, 12288, 24576 or 49152')
    need(effort in EFFORT_ARMS, 'choose high or xhigh reasoning')
    need(type(max_decisions) is int and max_decisions in (30, 60, 120), 'choose exactly 30, 60 or 120 decisions')
    validate_single120(cap, effort, max_decisions)
    need(not (runtime / RECEIPT).exists() and not (runtime / ORIGINALS).exists(), 'already configured')
    need(not any((runtime / name).exists() for name in ('episodes', 'budget', 'orchestration')), 'runtime has execution evidence')
    manifest = read(runtime / 'source-manifest.json')
    verify_inventory(runtime, manifest)
    base_plan = read(runtime / 'campaign-plan.json')
    base_config = read(runtime / 'runtime-config.json')
    need(base_config.get('portable_campaign') == CAMPAIGN, 'requires the frozen Luna campaign')
    frozen_config = read(API / 'portable_assets/campaign_specs' / CAMPAIGN / 'runtime-config.json')
    for key, value in frozen_config.items():
        if key not in ('cuawright_source', 'frozen_support_source', 'chromium', 'python', 'support_source_hashes'):
            need(base_config.get(key) == value, 'base scientific configuration changed: ' + key)
    frozen_plan = read(API / 'portable_assets/campaign_specs' / CAMPAIGN / 'campaign-plan.json')
    need(base_plan['arms'] == frozen_plan['arms'] and base_plan['protocol'] == frozen_plan['protocol'], 'base actor protocol changed')
    model = adapt_model((runtime / 'luna_model.py').read_text(), effort, max_decisions)
    worker = adapt_worker((runtime / 'worker.py').read_text(), max_decisions)
    rows = task_roster()
    originals = runtime / ORIGINALS
    originals.mkdir()
    for name in (*CHANGED, 'source-manifest.json'):
        (originals / name).write_bytes((runtime / name).read_bytes())
    (runtime / 'luna_model.py').write_text(model)
    (runtime / 'worker.py').write_text(worker)
    write(runtime / 'tasks.json', rows)
    plan = derived_plan(base_plan, cap, digest(runtime / 'tasks.json'), effort, max_decisions)
    write(runtime / 'campaign-plan.json', plan)
    config = derived_config(base_config, max_decisions)
    write(runtime / 'runtime-config.json', config)
    build = read(runtime / 'portable-build.json')
    build['prospective_protocol'] = SCHEMA
    build['tasks_sha256'] = digest(runtime / 'tasks.json')
    build['adaptations'].append({'reason': 'Explicit Luna single effort/output-cap sweep; payload, reservation, validation and tracking use declared arm.',
                                 'output_tokens': cap, 'reasoning_effort': effort, 'files': ['luna_model.py', 'worker.py']})
    write(runtime / 'portable-build.json', build)
    receipt = {'schema': SCHEMA, 'arm': EFFORT_ARMS[effort], 'model': 'gpt-6-luna', 'reasoning_effort': effort,
               'proposal_count': 1, 'max_decisions': max_decisions, 'output_tokens': cap, 'output_multiplier': output_multiplier(cap),
               'benchmark_counts': COUNTS, 'model_pricing': config['model_pricing'],
               'judge_protocol': config['judge_protocol'], 'preparation_source_sha256': digest(__file__),
               'base_inventory_sha256': digest(originals / 'source-manifest.json'),
               'original_sha256': {name: digest(originals / name) for name in CHANGED},
               'generated_sha256': {name: digest(runtime / name) for name in CHANGED}}
    if max_decisions == 120:
        receipt['prompt_max_decisions'] = 120
    write(runtime / RECEIPT, receipt)
    for name in (*CHANGED, RECEIPT):
        manifest['source_hashes'][str(runtime / name)] = digest(runtime / name)
    for path in originals.iterdir():
        manifest['source_hashes'][str(path)] = digest(path)
    manifest['source_hashes'][str(Path(__file__).resolve())] = digest(__file__)
    manifest['plan_sha256'] = digest(runtime / 'campaign-plan.json')
    write(runtime / 'source-manifest.json', manifest)
    write(runtime / 'execution-source-manifest.json', manifest)
    validate(runtime)
    return receipt


def validate(runtime):
    """Read-only verification, called before any browser/SDK import or API call."""
    try:
        return _validate(runtime)
    except (KeyError, TypeError, OSError) as error:
        raise ValueError('Prospective sweep: malformed or unpaired override: ' + str(error)) from error


def _validate(runtime):
    runtime = Path(runtime).resolve()
    receipt = read(runtime / RECEIPT)
    need(receipt.get('schema') == SCHEMA and receipt.get('preparation_source_sha256') == digest(__file__), 'override schema/preparation source changed')
    cap = receipt.get('output_tokens')
    need(type(cap) is int and cap in CAPS, 'malformed output cap')
    max_decisions = receipt.get('max_decisions')
    need(type(max_decisions) is int and max_decisions in (30, 60, 120), 'malformed decision cap')
    effort = receipt.get('reasoning_effort')
    need(effort in EFFORT_ARMS, 'malformed reasoning effort')
    validate_single120(cap, effort, max_decisions)
    if max_decisions == 120:
        need(type(receipt.get('prompt_max_decisions')) is int and receipt['prompt_max_decisions'] == 120, 'single120 prompt metadata differs')
    originals = runtime / ORIGINALS
    base = read(originals / 'source-manifest.json')
    need(receipt['base_inventory_sha256'] == digest(originals / 'source-manifest.json'), 'base inventory changed')
    for name in CHANGED:
        need(receipt['original_sha256'][name] == digest(originals / name), 'original source changed: ' + name)
        need(base['source_hashes'].get(str(runtime / name)) == digest(originals / name), 'original inventory binding missing: ' + name)
        need(receipt['generated_sha256'][name] == digest(runtime / name), 'generated source changed: ' + name)
    for name, expected in base['source_hashes'].items():
        if name not in {str(runtime / item) for item in CHANGED}:
            need(digest(name) == expected, 'unchanged base source changed: ' + name)
    need((runtime / 'luna_model.py').read_text() == adapt_model((originals / 'luna_model.py').read_text(), effort, max_decisions), 'model overlay differs')
    need((runtime / 'worker.py').read_text() == adapt_worker((originals / 'worker.py').read_text(), max_decisions), 'worker overlay differs')
    need(read(runtime / 'tasks.json') == task_roster(), 'roster differs from pinned original public tasks')
    plan = read(runtime / 'campaign-plan.json')
    need(plan == derived_plan(read(originals / 'campaign-plan.json'), cap, digest(runtime / 'tasks.json'), effort, max_decisions), 'declared plan differs')
    config = read(runtime / 'runtime-config.json')
    expected_config = derived_config(read(originals / 'runtime-config.json'), max_decisions)
    need(config == expected_config, 'scientific configuration differs')
    need(receipt['model_pricing'] == config['model_pricing'] and receipt['judge_protocol'] == config['judge_protocol'], 'pricing/judge binding differs')
    need({key: receipt[key] for key in ('arm', 'model', 'reasoning_effort', 'proposal_count', 'max_decisions', 'output_multiplier', 'benchmark_counts')} ==
         dict(arm=EFFORT_ARMS[effort], model='gpt-6-luna', reasoning_effort=effort, proposal_count=1, max_decisions=max_decisions, output_multiplier=output_multiplier(cap), benchmark_counts=COUNTS), 'receipt identity differs')
    manifest = read(runtime / 'source-manifest.json')
    need(manifest == read(runtime / 'execution-source-manifest.json'), 'execution inventory differs')
    required = [runtime / name for name in (*CHANGED, RECEIPT)] + list(originals.iterdir()) + [Path(__file__).resolve()]
    need(all(manifest['source_hashes'].get(str(path)) == digest(path) for path in required), 'override is not paired with complete inventory')
    verify_inventory(runtime, manifest)
    return receipt
