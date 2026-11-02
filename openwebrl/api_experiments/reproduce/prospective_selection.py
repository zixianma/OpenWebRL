"""Prepare explicit Luna N2/30 or N5/60 local BrowserCode experiments.

New runtimes only; no provider/browser work or financial authorization. The
frozen GPT actor, parser, selected-only history and physical reservations are
retained. Terra uses the same pinned terminal-success source as other runs.
"""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
API=HERE.parent
BASE_CAMPAIGN='browsercode_gpt_high'
SCHEMA='browsercode-luna-prospective-selection-v1'
RECEIPT='selection-experiment.json'
ORIGINALS='selection-originals'
SUPPORTED={(2,30):'parallel2_select_high',(5,60):'parallel5_select_high'}
MODEL_SHA='6ea320ed903a6ad68ff1a8ac1d075ce141a2e2890dc2229e51bfa849bb7d2c2f'
CHANGED=('paired_model.py','worker.py','common_judge.py','episode_metrics.py','campaign-plan.json','runtime-config.json','tasks.json','portable-build.json')
ADDED=('claude_budget.py','request_budget.py')

def load(name):
 spec=importlib.util.spec_from_file_location('_selection_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def need(x,message):
 if not x:raise ValueError('Prospective selection: '+message)

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def digest(p):return load('prospective_browsercode').digest(p)
def roster():return load('prospective_browsercode').task_roster()
def parameters(candidates,decisions):
 need(type(candidates)is int and type(decisions)is int and (candidates,decisions)in SUPPORTED,'choose explicit N2/30 or N5/60')
 return SUPPORTED[candidates,decisions]

def replace(text,old,new):
 need(text.count(old)==1,'source anchor changed: '+old)
 return text.replace(old,new)

def portable_worker_source():
 text=(API/'browsercode/workers/gpt_high_worker.py').read_text()
 text=replace(text,'from preparation_gate import validate\n','# Private campaign registration is intentionally not imported by the portable episode runner.\n')
 text=replace(text,"PLAN['plan_sha256']","hashlib.sha256((ROOT/'campaign-plan.json').read_bytes()).hexdigest()")
 text=replace(text,"'WEBWRIGHT_BROWSER_BACKEND':'cdp'","'WEBWRIGHT_BROWSER_BACKEND':'cdp','PYTHONPATH':os.pathsep.join([str(ROOT),CONFIG['cuawright_source'],CONFIG['frozen_support_source']]),'PYTHONDONTWRITEBYTECODE':'1'")
 return text

def adapt_model(text,candidates,decisions):
 arm=parameters(candidates,decisions)
 import hashlib
 need(hashlib.sha256(text.encode()).hexdigest()==MODEL_SHA,'original GPT model differs')
 text=replace(text,"MODEL_BY_MODE={'serial_high'", "MODEL_BY_MODE={'parallel2_select_high':'gpt-6-luna','serial_high'")
 text=replace(text,"COUNTS={'serial_high'", "COUNTS={'parallel2_select_high':2,'serial_high'")
 text=replace(text,"if self.turn>=30: raise ValueError('30-decision cap')",f"if self.turn>={decisions}: raise ValueError('Prospective decision cap')")
 anchor="  self.expected_model=MODEL_BY_MODE[mode]\n"
 bound=("  declared=json.loads((Path(__file__).with_name('campaign-plan.json')).read_text())\n"
        f"  if declared.get('prospective_selection_protocol')!='{SCHEMA}' or mode!='{arm}' or declared['protocol']['max_decisions']!={decisions} or declared['arms'][0]['proposal_count']!={candidates}:raise ValueError('Prospective selection constructor differs')\n"
        "  if kwargs.get('max_output_tokens')!=12288 or kwargs.get('reasoning_effort')!='high':raise ValueError('Fixed high12288 constructor required')\n")
 text=replace(text,anchor,bound+anchor)
 old="if payload.get('model')!=self.expected_model or payload.get('reasoning')!={'effort':'high'} or payload.get('store') is not False:"
 text=replace(text,old,"if payload.get('max_output_tokens')!=12288 or "+old[3:])
 compile(text,'<prospective selection model>','exec');return text

def adapt_worker(text,decisions):
 need(text==portable_worker_source(),'original portable GPT worker differs')
 text=replace(text,'from common_judge import judge','from common_judge import judge\nfrom request_budget import ReservationBudget as JudgeBudget')
 text=replace(text,"'step_cap':30",f"'step_cap':{decisions}")
 text=replace(text,'step_limit=30',f'step_limit={decisions}')
 text=replace(text,"row={'task_id':item['task_id'],","row={'task_id':item['task_id'],'benchmark':item['benchmark'],")
 compile(text,'<prospective selection worker>','exec');return text

def metrics_source(text):
 return replace(text,"cost=api_cost(u,'judge' if role=='judge' else 'actor',pricing)['upper_usd']","cost=d['cost']['upper_usd'] if role=='judge' and 'cost'in d else api_cost(u,'judge' if role=='judge' else 'actor',pricing)['upper_usd']")

def protocol_sources():
 rejudge=load('rejudge');judge,budget,sources=rejudge.science()
 entries={x['file']:x for x in read(API/'source-provenance.json')['snapshots']['browsercode_luna_effort']['files']}
 files={name:API/entries[name]['published_path']for name in ('common_judge.py',*ADDED)}
 return deepcopy(judge.TERRA_PROTOCOL),files,sources

def derived_plan(base,candidates,decisions,tasks_sha,terra):
 arm=parameters(candidates,decisions);protocol=deepcopy(base['protocol'])
 protocol.update(max_steps=decisions,max_decisions=decisions,actor_max_output_tokens=12288,selector_max_output_tokens=12288,actor_effort='high',selector_effort='high',episode_actor_execution_deadline_seconds=900,judge_model=terra['model'],judge_protocol=terra)
 spec=dict(arm=arm,model='gpt-6-luna',proposal_count=candidates,reasoning_effort='high',output_tokens=12288,selector_reasoning_effort='high',selector_output_tokens=12288)
 return dict(schema='portable-scientific-campaign-v1',campaign='luna_prospective_selection',prospective_selection_protocol=SCHEMA,protocol=protocol,arms=[spec],arm_protocols={arm:dict(actor_model='gpt-6-luna',selector_model='gpt-6-luna',candidates=candidates)},cells=[dict(harness='browsercode',policy=arm)],task_manifest_sha256=tasks_sha,benchmark_counts={'online-mind2web':300,'webvoyager':595,'deepshop':150},task_order='same complete public task roster and order as Luna single output sweep')

def derived_config(base,decisions,terra):
 config=deepcopy(base);config['prospective_selection_protocol']=SCHEMA;config['judge_protocol']=terra
 old='Only30 model decisions are available.'
 need(config['system_templates']['browsercode'].count(old)==1,'actor prompt decision limit changed')
 config['system_templates']['browsercode']=config['system_templates']['browsercode'].replace(old,f'Only{decisions} model decisions are available.')
 for pricing in [config['pricing'],*config['model_pricing'].values()]:pricing.update(judge_input=terra['pricing']['input'],judge_cached=terra['pricing']['cached_input'],judge_output=terra['pricing']['output'])
 return config

def inventory(runtime,m):
 need(m['plan_sha256']==digest(runtime/'campaign-plan.json'),'plan inventory differs')
 for path,pin in m['source_hashes'].items():need(digest(path)==pin,'inventory source drift: '+path)

def configure(runtime,candidates,decisions):
 runtime=Path(runtime).resolve();arm=parameters(candidates,decisions)
 need(not runtime.is_relative_to(Path('/gpfs/home/zixianma')),'home is read-only')
 need(not any((runtime/n).exists()for n in(RECEIPT,ORIGINALS,'prospective-experiment.json','cloud-experiment.json','episodes','budget','orchestration')),'new unused unconfigured runtime required')
 base_manifest=read(runtime/'source-manifest.json');inventory(runtime,base_manifest)
 base=read(runtime/'campaign-plan.json');config=read(runtime/'runtime-config.json')
 need(config.get('portable_campaign')==BASE_CAMPAIGN,'requires original GPT-high base runtime')
 frozen=read(API/'portable_assets/campaign_specs'/BASE_CAMPAIGN/'runtime-config.json')
 need(all(config.get(k)==v for k,v in frozen.items()if k not in('cuawright_source','frozen_support_source','chromium','python','support_source_hashes')),'base scientific configuration differs')
 frozen_plan=read(API/'portable_assets/campaign_specs'/BASE_CAMPAIGN/'campaign-plan.json')
 need(base['protocol']==dict(frozen_plan['protocol'],max_decisions=30)and base['arm_protocols']==frozen_plan['arm_protocols'],'base plan differs')
 model=adapt_model((runtime/'paired_model.py').read_text(),candidates,decisions);worker=adapt_worker((runtime/'worker.py').read_text(),decisions)
 terra,replacements,source_pins=protocol_sources();metrics=metrics_source((runtime/'episode_metrics.py').read_text());tasks=roster()
 archive=runtime/ORIGINALS;archive.mkdir();original_names=[*CHANGED,*[n for n in ADDED if(runtime/n).exists()],'source-manifest.json','execution-source-manifest.json']
 for name in original_names:(archive/name).write_bytes((runtime/name).read_bytes())
 write(runtime/'tasks.json',tasks);write(runtime/'campaign-plan.json',derived_plan(base,candidates,decisions,digest(runtime/'tasks.json'),terra));write(runtime/'runtime-config.json',derived_config(config,decisions,terra))
 (runtime/'paired_model.py').write_text(model);(runtime/'worker.py').write_text(worker);(runtime/'episode_metrics.py').write_text(metrics)
 for name,path in replacements.items():(runtime/name).write_bytes(path.read_bytes())
 build=read(runtime/'portable-build.json');build['prospective_selection_protocol']=SCHEMA;build['tasks_sha256']=digest(runtime/'tasks.json');build['adaptations'].append(dict(reason='Explicit Luna parallel selection variant; original actor reservation/parser/history retained; common Terra first judge.',candidates=candidates,max_decisions=decisions,archive=ORIGINALS));write(runtime/'portable-build.json',build)
 names=(*CHANGED,*ADDED);receipt=dict(schema=SCHEMA,arm=arm,model='gpt-6-luna',reasoning_effort='high',proposal_count=candidates,max_decisions=decisions,output_tokens=12288,selector_output_tokens=12288,selector_reasoning_effort='high',output_multiplier=1,benchmark_counts={'online-mind2web':300,'webvoyager':595,'deepshop':150},judge_protocol=terra,preparation_source_sha256=digest(__file__),base_inventory_sha256=digest(archive/'source-manifest.json'),original_sha256={n:digest(archive/n)for n in original_names},generated_sha256={n:digest(runtime/n)for n in names},terra_source_sha256=source_pins,paid_authorization=False)
 write(runtime/RECEIPT,receipt)
 for path in [*[runtime/n for n in(*names,RECEIPT)],*archive.iterdir(),Path(__file__).resolve(),*replacements.values()]:base_manifest['source_hashes'][str(path)]=digest(path)
 base_manifest['plan_sha256']=digest(runtime/'campaign-plan.json');write(runtime/'source-manifest.json',base_manifest);write(runtime/'execution-source-manifest.json',base_manifest);validate(runtime);return receipt

def validate(runtime):
 try:return _validate(Path(runtime).resolve())
 except(KeyError,TypeError,OSError)as e:raise ValueError('Prospective selection: malformed or unpaired runtime: '+str(e))from e

def _validate(runtime):
 r=read(runtime/RECEIPT);need(r['schema']==SCHEMA and r['preparation_source_sha256']==digest(__file__)and r['paid_authorization']is False,'marker/source differs');n=r['proposal_count'];dec=r['max_decisions'];arm=parameters(n,dec);archive=runtime/ORIGINALS;base=read(archive/'source-manifest.json')
 need(r['base_inventory_sha256']==digest(archive/'source-manifest.json'),'base inventory changed')
 for name,pin in r['original_sha256'].items():need(digest(archive/name)==pin,'archived original drift')
 for name in CHANGED:need(base['source_hashes'].get(str(runtime/name))==digest(archive/name),'original inventory binding missing: '+name)
 for path,pin in base['source_hashes'].items():
  if path not in {str(runtime/n)for n in(*CHANGED,*ADDED)}:need(digest(path)==pin,'unchanged base source differs')
 terra,replacements,sources=protocol_sources();need(r['terra_source_sha256']==sources and r['judge_protocol']==terra,'Terra source/protocol differs')
 for name,pin in r['generated_sha256'].items():need(digest(runtime/name)==pin,'generated inventory differs')
 need((runtime/'paired_model.py').read_text()==adapt_model((archive/'paired_model.py').read_text(),n,dec),'model overlay differs')
 need((runtime/'worker.py').read_text()==adapt_worker((archive/'worker.py').read_text(),dec),'worker overlay differs')
 need((runtime/'episode_metrics.py').read_text()==metrics_source((archive/'episode_metrics.py').read_text()),'metric overlay differs')
 for name,path in replacements.items():need((runtime/name).read_bytes()==path.read_bytes(),'judge/budget bytes differ')
 need(read(runtime/'tasks.json')==roster(),'fixed1045 roster differs')
 need(read(runtime/'campaign-plan.json')==derived_plan(read(archive/'campaign-plan.json'),n,dec,digest(runtime/'tasks.json'),terra),'plan differs')
 need(read(runtime/'runtime-config.json')==derived_config(read(archive/'runtime-config.json'),dec,terra),'config differs')
 need({k:r[k]for k in('arm','model','reasoning_effort','output_tokens','selector_output_tokens','selector_reasoning_effort','output_multiplier','benchmark_counts')}==dict(arm=arm,model='gpt-6-luna',reasoning_effort='high',output_tokens=12288,selector_output_tokens=12288,selector_reasoning_effort='high',output_multiplier=1,benchmark_counts={'online-mind2web':300,'webvoyager':595,'deepshop':150}),'marker arm identity differs')
 m=read(runtime/'source-manifest.json');need(m==read(runtime/'execution-source-manifest.json'),'execution inventory differs')
 required=[runtime/n for n in(*CHANGED,*ADDED,RECEIPT)]+list(archive.iterdir())+[Path(__file__).resolve(),*replacements.values()]
 need(all(m['source_hashes'].get(str(path))==digest(path)for path in required),'complete inventory binding missing');inventory(runtime,m);return r
