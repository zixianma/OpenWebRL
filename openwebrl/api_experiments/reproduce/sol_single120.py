"""Prepare the prospective Sol high/120 BrowserCode protocol without API calls.

Use existing public canonical IDs with native_task_id/source_task_id metadata;
keep separate per-benchmark run outputs.
The generated model, worker decision/prompt limits and native Terra transport
extend the recorded Sol60 protocol only at the enforced and advertised decision
limit; budgets remain explicit at execution.
"""
from __future__ import annotations
import ast,hashlib,importlib,json
from contextlib import contextmanager
from collections import Counter
from copy import deepcopy
from pathlib import Path

HERE=Path(__file__).resolve().parent
API=HERE.parent
ARM='sol_single_high120'
MODEL='gpt-6-sol'
SCHEMA='browsercode-sol-high-single120-v1'
RECEIPT='sol-single120-experiment.json'
ORIGINALS='sol-single120-originals'
COUNTS={'online-mind2web':300,'webvoyager':595,'deepshop':150}
PRICING={'input':2.,'cached_input':.2,'cache_write':2.5,'output':10.}
CHANGED=('worker.py','common_judge.py','runtime-config.json','campaign-plan.json','tasks.json','portable-build.json')
MODEL_SHA='804ab88145912a5ebd4056cc7c17f67432f27225880de300fc0f23b51ffb5f54'
read=lambda path:json.loads(Path(path).read_text())
def need(condition,message):
 if not condition:raise ValueError('Sol high120: '+message)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def task_roster():
 from tasks import load_tasks,canonical_benchmark
 rows=load_tasks('browsercode_luna_effort')+load_tasks('browsercode_gpt_high_wv_ds')
 for row in rows:row['benchmark']=canonical_benchmark(row['benchmark'])
 need(dict(Counter(r['benchmark']for r in rows))==COUNTS,'canonical benchmark counts changed')
 need(len({(r['benchmark'],r['task_id'])for r in rows})==1045,'duplicate canonical benchmark/task identity')
 need(len({r['task_id']for r in rows})==1045,'duplicate globally canonical task ID')
 return rows

def derive_sol_model(original):
 tree=ast.parse(original)
 constants={'MODEL':MODEL,'MODES':{ARM:(1,'high',None)},'PRICING':{'input':2.,'cached_input':.2,'cache_write':2.5,'output':10.}}
 found=set()
 for node in tree.body:
  if isinstance(node,ast.Assign):
   for target in node.targets:
    if isinstance(target,ast.Name)and target.id in constants:
     node.value=ast.parse(repr(constants[target.id]),mode='eval').body;found.add(target.id)
  if isinstance(node,ast.ClassDef)and node.name=='LunaModel':
   node.name='SolModel'
   init=next(n for n in node.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
   init.body=ast.parse("need(http_gate is not None and http_gate is REQUIRED_HTTP_GATE, 'Root-owned single HTTP gate required')").body+init.body
   query=next(n for n in node.body if isinstance(n,ast.FunctionDef)and n.name=='query')
   guard=query.body[0]
   if ast.unparse(guard)!="need(self.turn < 30, 'Frozen30-decision cap')":raise ValueError('Exact original30 guard required')
   query.body[0]=ast.parse("need(self.turn < 120, 'Prespecified120-decision cap')").body[0]
   # Remove unreachable proposal selection code; a second model call is disallowed.
   selector=next(n for n in ast.walk(query)if isinstance(n,ast.If)and ast.unparse(n.test)=='self.count > 1')
   selector.body=ast.parse("raise ValueError('Sol single120 never makes a selector call')").body
  if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='PairedModel'for t in node.targets):
   node.value=ast.Name(id='SolModel',ctx=ast.Load())
 if found!=set(constants):raise ValueError('All exact transport constants required')
 tree.body.insert(1,ast.Assign(targets=[ast.Name(id='REQUIRED_HTTP_GATE',ctx=ast.Store())],value=ast.Constant(value=None)))
 return ast.unparse(ast.fix_missing_locations(tree))+'\n'

def gated_judge(source):
 tree=ast.parse(source);judge=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='judge');attempt=next(n for n in judge.body if isinstance(n,ast.Try))
 indices=[i for i,n in enumerate(attempt.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.slice,ast.Constant)and t.slice.value=='physical_http_dispatch_epoch'for t in n.targets)]
 if len(indices)!=1:raise ValueError('Exact Terra dispatch marker required')
 start=indices[0]-1
 if ast.unparse(attempt.body[start])!="need(not budget.halted(), 'Judge pool halted before HTTP')":raise ValueError('Native judge halt guard changed')
 end=next(i for i in range(start+1,len(attempt.body))if isinstance(attempt.body[i],ast.Try))+1
 wrapped=ast.With(items=[ast.withitem(context_expr=ast.parse('RUNTIME_HTTP_GATE.slot(ident)',mode='eval').body)],body=attempt.body[start:end],type_comment=None)
 attempt.body[start:end]=ast.parse("need(RUNTIME_HTTP_GATE is not None, 'Root Terra HTTP gate required')").body+[wrapped]
 tree.body.insert(1,ast.Assign(targets=[ast.Name(id='RUNTIME_HTTP_GATE',ctx=ast.Store())],value=ast.Constant(value=None)))
 return ast.unparse(ast.fix_missing_locations(tree))+'\n'

def adapt_worker(source):
 pairs=[('from luna_model import LunaModel as PairedModel,MODES','from sol_model import SolModel as PairedModel,MODES\nRUNTIME_HTTP_GATE=None'),("'step_cap':30","'step_cap':120"),('step_limit=30','step_limit=120'),("harness=kind,model_name=model_name","harness=kind,http_gate=RUNTIME_HTTP_GATE,model_name=model_name")]
 for old,new in pairs:
  need(source.count(old)==1,'native worker binding changed: '+old);source=source.replace(old,new)
 compile(source,'<Sol120 worker>','exec');return source

def derived_config(base):
 config=deepcopy(base);prompt=config['system_templates']['browsercode']
 need(prompt.count('Only30 model decisions are available.')==1,'original advertised decision limit changed')
 config['system_templates']['browsercode']=prompt.replace('Only30 model decisions are available.','Only120 model decisions are available.')
 config['model_pricing'][MODEL]=PRICING
 config.update(sol_single120_protocol=SCHEMA,max_decisions=120,portable_campaign='browsercode_sol_high120')
 return config

def derived_plan(base,tasks_sha,judge_protocol):
 return dict(schema=SCHEMA,campaign='browsercode_sol_high120',sol_single120_protocol=SCHEMA,source_campaign=base['campaign'],arms=[dict(arm=ARM,model=MODEL,proposal_count=1,reasoning_effort='high',output_tokens=12288,selector_reasoning_effort=None)],task_manifest_sha256=tasks_sha,task_count=1045,benchmark_counts=COUNTS,protocol=dict(harness='browsercode',max_decisions=120,prompt_max_decisions=120,actor_max_output_tokens=12288,actor_effort='high',proposal_count=1,selector_calls=0,selected_only_execution=True,actor_episode_seconds=900,http_timeout_seconds=180,browser_command_seconds=120,judge_model='gpt-5.6-terra',judge_protocol=judge_protocol,no_unchanged_episode_or_paid_judge_retry=True))

def inventory(manifest):
 for path,expected in manifest['source_hashes'].items():need(digest(path)==expected,'bound source changed: '+path)

def configure(runtime):
 runtime=Path(runtime).resolve()
 need(not(runtime/RECEIPT).exists()and not(runtime/ORIGINALS).exists(),'already configured')
 need(not any((runtime/name).exists()for name in('episodes','budget','orchestration')),'runtime has execution evidence')
 manifest=read(runtime/'source-manifest.json');inventory(manifest)
 config=read(runtime/'runtime-config.json');base=read(runtime/'campaign-plan.json')
 frozen=API/'portable_assets/campaign_specs/browsercode_luna_effort'
 frozen_config=read(frozen/'runtime-config.json');frozen_plan=read(frozen/'campaign-plan.json')
 need(config.get('portable_campaign')=='browsercode_luna_effort','requires original Luna transport base')
 for key,value in frozen_config.items():
  if key not in('cuawright_source','frozen_support_source','chromium','python','support_source_hashes'):need(config.get(key)==value,'base configuration changed: '+key)
 need(base['arms']==frozen_plan['arms']and base['protocol']==frozen_plan['protocol'],'base actor protocol changed')
 need(digest(runtime/'luna_model.py')==MODEL_SHA,'native Luna transport source changed')
 originals=runtime/ORIGINALS;originals.mkdir()
 for name in(*CHANGED,'source-manifest.json'):(originals/name).write_bytes((runtime/name).read_bytes())
 (runtime/'sol_model.py').write_text(derive_sol_model((runtime/'luna_model.py').read_text()))
 (runtime/'worker.py').write_text(adapt_worker((runtime/'worker.py').read_text()))
 (runtime/'common_judge.py').write_text(gated_judge((runtime/'common_judge.py').read_text()))
 write(runtime/'tasks.json',task_roster())
 config=derived_config(config);write(runtime/'runtime-config.json',config)
 write(runtime/'campaign-plan.json',derived_plan(base,digest(runtime/'tasks.json'),config['judge_protocol']))
 build=read(runtime/'portable-build.json');build['sol_single120_protocol']=SCHEMA;build['tasks_sha256']=digest(runtime/'tasks.json')
 build['adaptations'].append(dict(reason='Sol high single12288; exact enforced and advertised120 decisions; native900-second actor and Terra unchanged.',files=['sol_model.py','worker.py','common_judge.py']))
 write(runtime/'portable-build.json',build)
 receipt=dict(schema=SCHEMA,arm=ARM,model=MODEL,reasoning_effort='high',proposal_count=1,max_decisions=120,prompt_max_decisions=120,output_tokens=12288,actor_episode_seconds=900,benchmark_counts=COUNTS,benchmark_required=True,model_pricing=PRICING,judge_protocol=config['judge_protocol'],preparation_source_sha256=digest(__file__),base_inventory_sha256=digest(originals/'source-manifest.json'),original_sha256={n:digest(originals/n)for n in CHANGED},generated_sha256={n:digest(runtime/n)for n in(*CHANGED,'sol_model.py')})
 write(runtime/RECEIPT,receipt)
 for path in[*[runtime/n for n in(*CHANGED,'sol_model.py',RECEIPT)],*originals.iterdir(),Path(__file__).resolve()]:manifest['source_hashes'][str(path)]=digest(path)
 manifest['plan_sha256']=digest(runtime/'campaign-plan.json');write(runtime/'source-manifest.json',manifest);write(runtime/'execution-source-manifest.json',manifest)
 validate(runtime);return receipt

def validate(runtime):
 try:return _validate(Path(runtime).resolve())
 except(KeyError,TypeError,OSError)as error:raise ValueError('Sol high120: malformed or unpaired protocol: '+str(error))from error

def _validate(runtime):
 receipt=read(runtime/RECEIPT);need(receipt['schema']==SCHEMA and receipt['preparation_source_sha256']==digest(__file__),'preparation source differs')
 originals=runtime/ORIGINALS;base_inventory=read(originals/'source-manifest.json')
 need(receipt['base_inventory_sha256']==digest(originals/'source-manifest.json'),'base inventory changed')
 for name in CHANGED:
  need(receipt['original_sha256'][name]==digest(originals/name)==base_inventory['source_hashes'].get(str(runtime/name)),'original source differs: '+name)
 for name in(*CHANGED,'sol_model.py'):need(receipt['generated_sha256'][name]==digest(runtime/name),'generated source differs: '+name)
 for path,value in base_inventory['source_hashes'].items():
  if path not in{str(runtime/name)for name in CHANGED}:need(digest(path)==value,'unchanged base source differs: '+path)
 need(digest(runtime/'luna_model.py')==MODEL_SHA,'native transport source changed')
 need((runtime/'sol_model.py').read_text()==derive_sol_model((runtime/'luna_model.py').read_text()),'Sol transport overlay differs')
 need((runtime/'worker.py').read_text()==adapt_worker((originals/'worker.py').read_text()),'worker overlay differs')
 need((runtime/'common_judge.py').read_text()==gated_judge((originals/'common_judge.py').read_text()),'judge overlay differs')
 need(read(runtime/'tasks.json')==task_roster(),'canonical task roster differs')
 config=derived_config(read(originals/'runtime-config.json'));need(read(runtime/'runtime-config.json')==config,'configuration differs')
 need(read(runtime/'campaign-plan.json')==derived_plan(read(originals/'campaign-plan.json'),digest(runtime/'tasks.json'),config['judge_protocol']),'plan differs')
 expected=dict(arm=ARM,model=MODEL,reasoning_effort='high',proposal_count=1,max_decisions=120,prompt_max_decisions=120,output_tokens=12288,actor_episode_seconds=900,benchmark_counts=COUNTS,benchmark_required=True,model_pricing=PRICING,judge_protocol=config['judge_protocol'])
 need(all(receipt.get(k)==v for k,v in expected.items()),'declared protocol differs')
 manifest=read(runtime/'source-manifest.json');need(manifest==read(runtime/'execution-source-manifest.json'),'execution inventory differs')
 for path in[*[runtime/n for n in(*CHANGED,'sol_model.py',RECEIPT)],*originals.iterdir(),Path(__file__).resolve()]:need(manifest['source_hashes'].get(str(path))==digest(path),'complete overlay source pin missing')
 inventory(manifest);return receipt

def install_runtime_hooks(worker,root):
 runtime=Path(worker.__file__).resolve().parent
 model=importlib.import_module('sol_model');judge=importlib.import_module('common_judge');budgets=importlib.import_module('request_budget')
 need(all(Path(m.__file__).resolve().parent==runtime for m in(model,judge,budgets)),'different runtime modules loaded')
 class RuntimeHTTPGate(budgets.SharedHTTPGate):
  @contextmanager
  def slot(self,physical_id):
   # Keep the full Terra ID in request/budget receipts; normalize only the
   # shared gate's UUID key, as in the qualified native campaign adapter.
   gate_id=physical_id[6:]if type(physical_id)is str and physical_id.startswith('judge-')else physical_id
   with super().slot(gate_id):yield
 gate=RuntimeHTTPGate(Path(root)/'http-gate',1)
 need(model.REQUIRED_HTTP_GATE is None,'Sol gate already installed')
 model.REQUIRED_HTTP_GATE=gate;judge.RUNTIME_HTTP_GATE=gate;worker.RUNTIME_HTTP_GATE=gate
 worker.ReservationBudget=budgets.ReservationBudget;worker.JudgeBudget=budgets.ReservationBudget
 return gate


if __name__ == '__main__':
 import argparse,sys
 sys.dont_write_bytecode=True
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--runtime',required=True,type=Path)
 configure(parser.parse_args().runtime)
