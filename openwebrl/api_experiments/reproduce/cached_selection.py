"""Prepare fixed cached-context N3/30 or N5/60 in a new Luna runtime.

No provider/browser work or spending authorization. Historical selector context
is retained in its original package; this variant uses full native actor context.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
SCHEMA = 'browsercode-fixed-cached-selection-v1'
RECEIPT = 'cached-selection-experiment.json'
ORIGINALS = 'cached-selection-originals'
OPTIONS = {'luna_n3_cached_high30': (3, 30), 'luna_n5_cached_high60': (5, 60)}
CHANGED = ('worker.py', 'campaign-plan.json', 'runtime-config.json', 'tasks.json', 'portable-build.json')
ADDED = ('cached_model.py',)
MODEL_SHA = '5b83e9418f541f76bdd4a25e4c737e2fbec376f2ebdeefc8208baf7bbf216a71'


def need(value, message):
    if not value:
        raise ValueError('Fixed cached selection: ' + message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helper():
    spec = importlib.util.spec_from_file_location('_cached_base_sweep', HERE/'prospective_browsercode.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def parameters(arm):
    need(arm in OPTIONS, 'choose luna_n3_cached_high30 or luna_n5_cached_high60')
    return OPTIONS[arm]


def replace(text, old, new):
    need(text.count(old) == 1, 'frozen source anchor differs: ' + old)
    return text.replace(old, new)


def worker_source(original, decisions):
    expected = (API/'browsercode/workers/luna_effort_worker.py').read_text()
    expected = replace(expected, "'WEBWRIGHT_BROWSER_BACKEND':'cdp'",
                       "'WEBWRIGHT_BROWSER_BACKEND':'cdp','PYTHONPATH':os.pathsep.join([str(ROOT),CONFIG['cuawright_source'],CONFIG['frozen_support_source']]),'PYTHONDONTWRITEBYTECODE':'1'")
    need(original == expected, 'original portable Luna worker differs')
    text = replace(original, 'from luna_model import LunaModel as PairedModel,MODES',
                   'from cached_model import CachedLunaModel as PairedModel,MODES')
    for old, new in [("'step_cap':30", "'step_cap':"+str(decisions)),
                     ('step_limit=30', 'step_limit='+str(decisions)),
                     ("'max_decisions':30", "'max_decisions':"+str(decisions))]:
        text = replace(text, old, new)
    text = replace(text, "row={'task_id':item['task_id'],",
                   "row={'task_id':item['task_id'],'benchmark':item['benchmark'],")
    compile(text, '<portable fixed cached worker>', 'exec')
    return text


def plan_source(base, arm, task_hash):
    n, decisions = parameters(arm)
    protocol = deepcopy(base['protocol'])
    for name in ('same300_tasks_allarms','pilot20_tasks_per_arm_included_final','historical_high_results_judge'):
        protocol.pop(name, None)
    protocol.update(max_steps=decisions,max_decisions=decisions,prompt_max_decisions=decisions,
                    actor_max_output_tokens=12288,selector_max_output_tokens=12288,
                    actor_effort='high',selector_effort='high',
                    episode_actor_execution_deadline_seconds=900)
    spec = dict(arm=arm,model='gpt-6-luna',proposal_count=n,reasoning_effort='high',
                output_tokens=12288,selector_model='gpt-6-luna',
                selector_reasoning_effort='high',selector_output_tokens=12288)
    return dict(schema='portable-scientific-campaign-v1',campaign='fixed_cached_selection',
                cached_selection_protocol=SCHEMA,protocol=protocol,arms=[spec],
                cells=[dict(harness='browsercode',policy=arm)],benchmark_counts=helper().COUNTS,
                task_manifest_sha256=task_hash)


def config_source(base, arm):
    _, decisions = parameters(arm)
    config = deepcopy(base)
    config['cached_selection_protocol'] = SCHEMA
    old = 'Only30 model decisions are available.'
    config['system_templates']['browsercode'] = replace(config['system_templates']['browsercode'],old,
                                                       'Only'+str(decisions)+' model decisions are available.')
    config['max_decisions'] = decisions
    return config


def inventory(runtime, manifest):
    need(manifest['plan_sha256'] == digest(runtime/'campaign-plan.json'), 'plan inventory differs')
    for name, expected in manifest['source_hashes'].items():
        need(digest(name) == expected, 'source inventory differs: '+name)


def configure(runtime, arm):
    runtime = Path(runtime).resolve()
    n, decisions = parameters(arm)
    need(not runtime.is_relative_to(Path('/gpfs/home/zixianma')), 'home is read-only')
    need(not any((runtime/name).exists() for name in (RECEIPT,ORIGINALS,*ADDED,'episodes','budget','orchestration',
                  'prospective-experiment.json','selection-experiment.json','cloud-experiment.json')), 'new unconfigured runtime required')
    manifest = read(runtime/'source-manifest.json')
    inventory(runtime,manifest)
    config = read(runtime/'runtime-config.json')
    base = read(runtime/'campaign-plan.json')
    need(config.get('portable_campaign') == helper().CAMPAIGN, 'requires frozen Luna base runtime')
    fixed = read(API/'portable_assets/campaign_specs/browsercode_luna_effort/runtime-config.json')
    need(all(config.get(k) == v for k,v in fixed.items() if k not in
             ('cuawright_source','frozen_support_source','chromium','python','support_source_hashes')),
         'base scientific configuration differs')
    fixed_plan = read(API/'portable_assets/campaign_specs/browsercode_luna_effort/campaign-plan.json')
    need(base['arms'] == fixed_plan['arms'] and base['protocol'] == fixed_plan['protocol'], 'base plan differs')
    need(digest(runtime/'luna_model.py') == helper().MODEL_SHA, 'base actor source differs')
    model = API/'browsercode/models/cached_luna_model.py'
    need(digest(model) == MODEL_SHA, 'cached model source differs')
    worker = worker_source((runtime/'worker.py').read_text(),decisions)
    tasks = helper().task_roster()
    archive = runtime/ORIGINALS
    archive.mkdir()
    for name in (*CHANGED,'source-manifest.json','execution-source-manifest.json'):
        (archive/name).write_bytes((runtime/name).read_bytes())
    write(runtime/'tasks.json',tasks)
    write(runtime/'campaign-plan.json',plan_source(base,arm,digest(runtime/'tasks.json')))
    write(runtime/'runtime-config.json',config_source(config,arm))
    (runtime/'cached_model.py').write_bytes(model.read_bytes())
    (runtime/'worker.py').write_text(worker)
    build = read(runtime/'portable-build.json')
    build['cached_selection_protocol'] = SCHEMA
    build['tasks_sha256'] = digest(runtime/'tasks.json')
    build['adaptations'].append(dict(reason='Full native actor-context cached selection; separate selector-context protocol.',
                                    candidates=n,max_decisions=decisions,archive=ORIGINALS))
    write(runtime/'portable-build.json',build)
    receipt = dict(schema=SCHEMA,arm=arm,model='gpt-6-luna',proposal_count=n,max_decisions=decisions,
                   output_tokens=12288,selector_output_tokens=12288,reasoning_effort='high',
                   selector_reasoning_effort='high',benchmark_counts=helper().COUNTS,
                   judge_protocol=config['judge_protocol'],paid_authorization=False,
                   preparation_source_sha256=digest(__file__),cached_model_sha256=MODEL_SHA,
                   original_sha256={name:digest(archive/name) for name in (*CHANGED,'source-manifest.json','execution-source-manifest.json')},
                   generated_sha256={name:digest(runtime/name) for name in (*CHANGED,*ADDED)})
    write(runtime/RECEIPT,receipt)
    for path in [*[runtime/name for name in (*CHANGED,*ADDED,RECEIPT)],*archive.iterdir(),Path(__file__).resolve(),model]:
        manifest['source_hashes'][str(path)] = digest(path)
    manifest['plan_sha256'] = digest(runtime/'campaign-plan.json')
    write(runtime/'source-manifest.json',manifest)
    write(runtime/'execution-source-manifest.json',manifest)
    validate(runtime)
    return receipt


def validate(runtime):
    runtime = Path(runtime).resolve()
    r = read(runtime/RECEIPT)
    need(r['schema'] == SCHEMA and r['paid_authorization'] is False and
         r['preparation_source_sha256'] == digest(__file__), 'receipt/source differs')
    n, decisions = parameters(r['arm'])
    archive = runtime/ORIGINALS
    for name, expected in r['original_sha256'].items():
        need(digest(archive/name) == expected,'archived original differs')
    old = read(archive/'source-manifest.json')
    for name in CHANGED:
        need(old['source_hashes'][str(runtime/name)] == digest(archive/name),'original binding differs')
    for name, expected in old['source_hashes'].items():
        if name not in {str(runtime/name) for name in CHANGED}:
            need(digest(name) == expected,'unchanged original differs')
    need((runtime/'worker.py').read_text() == worker_source((archive/'worker.py').read_text(),decisions), 'worker differs')
    model = API/'browsercode/models/cached_luna_model.py'
    need(digest(model) == r['cached_model_sha256'] == MODEL_SHA and
         (runtime/'cached_model.py').read_bytes() == model.read_bytes(),'cached model differs')
    need(read(runtime/'tasks.json') == helper().task_roster(),'canonical task roster differs')
    need(read(runtime/'campaign-plan.json') == plan_source(read(archive/'campaign-plan.json'),r['arm'],digest(runtime/'tasks.json')),'plan differs')
    need(read(runtime/'runtime-config.json') == config_source(read(archive/'runtime-config.json'),r['arm']),'config differs')
    need(r['proposal_count'] == n and r['max_decisions'] == decisions and
         r['model'] == 'gpt-6-luna' and r['benchmark_counts'] == helper().COUNTS and
         r['output_tokens'] == r['selector_output_tokens'] == 12288 and
         r['reasoning_effort'] == r['selector_reasoning_effort'] == 'high' and
         r['judge_protocol'] == read(runtime/'runtime-config.json')['judge_protocol'], 'protocol declaration differs')
    for name, expected in r['generated_sha256'].items():
        need(digest(runtime/name) == expected,'generated source differs')
    manifest = read(runtime/'source-manifest.json')
    need(manifest == read(runtime/'execution-source-manifest.json'),'execution inventory differs')
    required = [*[runtime/name for name in (*CHANGED,*ADDED,RECEIPT)],*archive.iterdir(),Path(__file__).resolve(),model]
    need(all(manifest['source_hashes'].get(str(path)) == digest(path) for path in required),'complete source binding missing')
    inventory(runtime,manifest)
    return r
