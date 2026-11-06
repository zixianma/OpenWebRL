"""Keep a known max-output incomplete generation task-local, never retry it.

The frozen _request body is retained except its one generic halt statement.
All physical requests/responses, settled usage and invalid-task propagation stay
unchanged. Unknown/HTTP/counter/identity/accounting failures still halt normally.
"""
import ast,copy,hashlib,inspect,json
from decimal import Decimal
from pathlib import Path
SCHEMA='known-max-output-incomplete-task-invalid-no-pool-halt-v1'
HALT_REASON='Physical Responses/count failure; preserve all calls, no unchanged retry'
ERROR='Native incomplete/failed response preserved invalid'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def known_task_local(exc,native,row,known,admitted,count,budget,ident,payload,*,effort):
 """Fail closed: only already-settled, exactly typed native truncation qualifies."""
 try:
  if type(exc)is not ValueError or str(exc)!=ERROR or known is not True or admitted is not True or count is not False:return False
  if not isinstance(native,dict)or not isinstance(row,dict)or budget.halted():return False
  if native.get('status')!='incomplete'or native.get('incomplete_details')!={'reason':'max_output_tokens'}:return False
  if native.get('model')!='gpt-6-sol'or row.get('model')!='gpt-6-sol'or row.get('provider')!='gpt-6-sol':return False
  if (native.get('reasoning')or{}).get('effort')not in(None,effort):return False
  if type(ident)is not str or len(ident)!=32 or any(c not in'0123456789abcdef'for c in ident):return False
  if row.get('physical_id')!=ident or row.get('status')!='received'or row.get('role')!='candidate_0'or row.get('generation')is not True or row.get('endpoint')!='/v1/responses':return False
  if row.get('request')!=payload or row.get('response')!=native or row.get('usage')!=native.get('usage')or not row.get('physical_http_dispatch_epoch')or not row.get('response_received_epoch'):return False
  if payload.get('model')!='gpt-6-sol'or payload.get('max_output_tokens')!=12288 or payload.get('reasoning')!={'effort':effort}or payload.get('store')is not False or payload.get('service_tier')!='default':return False
  usage=native.get('usage');out=usage.get('output_tokens')if type(usage)is dict else None
  if type(out)is not int or out!=12288:return False
  state=budget.snapshot()['requests'].get(ident)
  if type(state)is not dict or state.get('physical_id')!=ident or state.get('status')!='known'or state.get('usage')!=usage:return False
  cost=Decimal(row['cost']['upper_usd_decimal'])
  if not cost.is_finite()or cost<0 or Decimal(state['charged_or_reserved_usd'])!=cost or Decimal(state['actual_usd'])!=cost:return False
  metadata=state.get('metadata')or{}
  if metadata.get('role')!='candidate_0'or metadata.get('request_sha256')!=row.get('request_sha256'):return False
  return True
 except Exception:return False

def derive(original,source_path,expected_sha256,effort):
 """Return the original function with exactly one halt decision guarded."""
 source_path=Path(source_path).resolve()
 if effort not in('high','xhigh')or Path(inspect.getsourcefile(original)).resolve()!=source_path:raise ValueError('Exact generated Sol source and effort required')
 source=source_path.read_text()
 if sha(source_path)!=expected_sha256:raise ValueError('Exact validated generated Sol source required')
 tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='SolModel');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_request');method=copy.deepcopy(method)
 replacement=ast.parse("if _known_task_local(exc,locals().get('native'),row,known,admitted,count,self.budget,ident,payload):\n row['task_local_known_incomplete_policy']={'schema':SCHEMA,'adapter_source_sha256':_adapter_sha(),'no_retry':True,'output_tokens_unchanged':12288}\nelse:\n self.budget.halt("+repr(HALT_REASON)+")").body[0]
 changed=0
 for n in ast.walk(method):
  if isinstance(n,ast.ExceptHandler):
   for i,stmt in enumerate(n.body):
    if isinstance(stmt,ast.Expr)and isinstance(stmt.value,ast.Call)and isinstance(stmt.value.func,ast.Attribute)and stmt.value.func.attr=='halt'and len(stmt.value.args)==1 and isinstance(stmt.value.args[0],ast.Constant)and stmt.value.args[0].value==HALT_REASON:
     n.body[i]=replacement;changed+=1
 if changed!=1:raise ValueError('Exactly one known generic physical halt decision required')
 module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]));scope=dict(original.__globals__);scope.update(_known_task_local=lambda *args:known_task_local(*args,effort=effort),_adapter_sha=lambda:sha(__file__),SCHEMA=SCHEMA)
 exec(compile(module,str(Path(__file__).resolve()),'exec'),scope)
 return scope['_request']

def install(worker,source_path,expected_sha256,effort):
 original=worker.PairedModel;method=original._request
 if original.__name__!='SolModel'or method.__name__!='_request':raise ValueError('Unmodified generated Sol model required')
 adapted=derive(method,source_path,expected_sha256,effort)
 worker.PairedModel=type('KnownIncompleteTaskLocalSolModel',(original,),{'_request':adapted,'__module__':__name__})
 return dict(schema=SCHEMA,source_path=str(Path(__file__).resolve()),source_sha256=sha(__file__),original_model_source_sha256=expected_sha256,scientific_output_cap=12288,task_invalid_still_raises=True,no_retry=True)
