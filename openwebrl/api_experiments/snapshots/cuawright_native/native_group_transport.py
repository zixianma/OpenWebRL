"""Native-owner-only Unix relay. Provider secrets and private evidence stay host-side."""
import asyncio,json,socket,time
from pathlib import Path

# Existing gateway HTTP body cap plus bounded Unix-envelope overhead.
MAX_FRAME=150000000+65536

def rpc(socket_path,request,deadline):
 remaining=deadline-time.monotonic()
 if remaining<=0:raise TimeoutError('Group deadline elapsed')
 raw=json.dumps(request,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
 if len(raw)>MAX_FRAME:raise ValueError('Group frame exceeds bounded size')
 with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
  s.settimeout(remaining);s.connect(socket_path);s.sendall(raw)
  with s.makefile('rb') as stream:line=stream.readline(MAX_FRAME+1)
 if len(line)>MAX_FRAME or not line.endswith(b'\n'):raise RuntimeError('Invalid bounded group response')
 response=json.loads(line)
 if response.get('verified')is not True:raise RuntimeError('Group relay rejected: '+str(response.get('error')))
 return response['result']

class Group:
 def __init__(self,transport,opened,deadline,candidate_count=3):
  if type(candidate_count)is not int or candidate_count not in (3,5):raise ValueError('Unsupported group candidate count')
  generations=candidate_count+1
  self.transport=transport;self.id=opened['group_id'];self.deadline=deadline
  if opened.get('held_model_calls')!=2*generations or opened.get('candidate_count',3)!=candidate_count or opened.get('admitted')is not True or opened.get('reserved_generation_calls')!=generations or opened.get('shared_model_inflight_limit')!=8 or not isinstance(opened.get('held_usd'),(int,float)) or opened['held_usd']<=0:raise RuntimeError('Incomplete atomic generation/counter hold')
  self.permit={'admitted':True,'reserved_generation_calls':generations,'auxiliary_calls_share_all_attempt_caps':True,'shared_model_inflight_limit':8,'deadline_monotonic':deadline,'gateway_admission':opened}
 async def create(self,*,role,payload,deadline_monotonic):
  assert deadline_monotonic==self.deadline
  data={'group_id':self.id,'payload':payload}
  if role.startswith('candidate_'):data.update(role='candidate',candidate_index=int(role.rsplit('_',1)[1]))
  elif role=='selector':data['role']='selector'
  else:raise ValueError('Unknown group role')
  response=await self.transport.call('create',data,self.deadline)
  if response.get('status')!=200:raise RuntimeError('Physical group response failed: '+str(response.get('status')))
  return response['body']
 async def finish(self,*,selected_index,error_type):
  return await self.transport.call('finish',{'group_id':self.id,'selected_candidate_index':selected_index,'error_type':error_type},max(self.deadline,time.monotonic()+10))

class BrokerTransport:
 def __init__(self,socket_path,case):self.socket_path=str(socket_path);self.case=case;self.groups={};self.deadlines={}
 async def call(self,operation,data,deadline):
  return await asyncio.to_thread(rpc,self.socket_path,{'op':'group_'+operation,'data':data,'deadline_monotonic':deadline},deadline)
 async def begin_decision(self,*,decision_id,model,candidate_count,selector_count,max_output_tokens,common_payload,deadline_monotonic):
  assert type(candidate_count)is int and candidate_count in (3,5) and selector_count==1 and max_output_tokens==20000
  opened=await self.call('open',{'case':self.case,'decision':decision_id,'candidate_payload':common_payload,'candidate_count':candidate_count},deadline_monotonic)
  group=Group(self,opened,deadline_monotonic,candidate_count);self.groups[decision_id]=group;self.deadlines[decision_id]=deadline_monotonic;return group
 def persist(self,*,kind,identity,record):
  decision=identity.split('-',1)[0] if kind=='physical' else identity
  group=self.groups.get(decision)
  if group is None and (kind!='decisions' or record.get('committed')is not False):raise RuntimeError('No admitted group for physical evidence')
  deadline=max(self.deadlines.get(decision,time.monotonic()),time.monotonic()+10)
  return rpc(self.socket_path,{'op':'group_record','data':{'group_id':None if group is None else group.id,'kind':kind+'/'+identity,'record':record},'deadline_monotonic':deadline},deadline)
