from pathlib import Path
import base64,json,time
from openwebrl.luna_qwen_metrics import write_json,api_cost
campaign=json.loads(Path(__file__).with_name('runtime-config.json').read_text())

def judge(client,budget,task,actions,screenshot,directory):
 import ast
 tree=ast.parse((Path(campaign['frozen_support_source'])/'openwebrl/eval/reward_online_mind2web.py').read_text());prompts={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['SYSTEM_PROMPT','USER_PROMPT'] for t in n.targets):prompts[n.targets[0].id]=ast.literal_eval(n.value)
 trajectory='\n\n'.join(f'Thought {i+1}: \nAction {i+1}: {text}' for i,text in enumerate(actions))
 request=dict(model='o4-mini-2025-04-16',seed=42,max_completion_tokens=4096,store=False,messages=[{'role':'system','content':prompts['SYSTEM_PROMPT']},{'role':'user','content':[{'type':'text','text':prompts['USER_PROMPT'].format(task=task,thoughts_and_actions=trajectory)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(screenshot).read_bytes()).decode(),'detail':'high'}}]}])
 bound=sum(len(str(x).encode()) for x in actions)+len(task.encode())+sum(len(v.encode()) for v in prompts.values())+32768+4096
 reserve=(bound*1.1+4096*4.4)/1e6;ident=budget.update(reserve=reserve);path=directory/'requests'/f'judge-{ident:05d}.json';row=dict(request=request,status='reserved',reserved_usd=reserve,started_epoch=time.time());write_json(path,row)
 try:
  response=client.chat.completions.create(**request);native=response.model_dump();row.update(response=native,usage=native['usage'],status='received');write_json(path,row);budget.update(settle=(reserve,api_cost(native['usage'],'judge',campaign['pricing'])['upper_usd']))
  if native.get('model') not in ['o4-mini-2025-04-16','o4-mini']:raise ValueError('Wrong native judge model')
  text=native['choices'][0]['message']['content'] or '';row['verdict_text']=text
  if 'status:' not in text.lower():raise ValueError('Unparseable native judge verdict')
  return int('success' in text.lower().split('status:',1)[1])
 except BaseException as e:
  row['error_type']=type(e).__name__;write_json(budget.root/'halt.json',{'reason':'Judge failure; no retries'});raise
 finally:write_json(path,row)
