"""Host-only authenticated relay from one native PID to private gateway control."""
import hashlib,json,re,time,urllib.request,urllib.error,uuid
from pathlib import Path

class GroupRelay:
 def __init__(self,config,owner):
  self.config=json.loads(Path(config).read_text());self.owner=owner
  from urllib.parse import urlparse
  parsed=urlparse(self.config['group_base_url'])
  if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1'):raise ValueError('Group gateway must be local')
  if not self.config.get('group_secret'):raise ValueError('Missing private group secret')
 def handle(self,peer_pid,peer_uid,request):
  operation=request['op'];assert operation in ('group_open','group_create','group_finish','group_record')
  self.owner.authorize(peer_pid,peer_uid,'group-control',operation)
  if operation=='group_record' and request['data'].get('group_id')is None:
   data=request['data'];record=data['record']
   if not re.fullmatch(r'decisions/[0-9]{3}',data['kind']) or record.get('committed')is not False:raise ValueError('Only failed pre-admission decisions may lack group ID')
   raw=json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
   directory=self.owner.evidence/'private-unadmitted-decisions';directory.mkdir(exist_ok=True)
   path=directory/(uuid.uuid4().hex+'.json')
   with path.open('xb')as f:f.write(raw)
   digest=hashlib.sha256(raw).hexdigest();self.owner.record({'event':'group_unadmitted_private_record','sha256':digest,'path':str(path)})
   return {'stored':True,'record_sha256':digest}
  remaining=float(request['deadline_monotonic'])-time.monotonic()
  if remaining<=0:raise TimeoutError('Group deadline elapsed')
  body=json.dumps(request['data'],separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
  req=urllib.request.Request(self.config['group_base_url'].rstrip('/')+'/group/'+operation.removeprefix('group_'),data=body,headers={'Content-Type':'application/json','X-Campaign-Group-Secret':self.config['group_secret']})
  with urllib.request.urlopen(req,timeout=remaining)as response:return json.loads(response.read())
