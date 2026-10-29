from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
REL='repairs/directory-http-gate-v1/fresh-canary-scope-v1.json'
PIN='d9ad45fd9367f94c2bc482f8b73549210351d3343786002a2aa22cebbcd4c61e'
def canary_ids(arm,config):
 p=ROOT/REL
 assert hashlib.sha256(p.read_bytes()).hexdigest()==PIN,'Fresh canary scope changed'
 d=json.loads(p.read_text());assert d['root_approved']is True and d['actor_rerun']is False and d['budget_reset']is False
 assert hashlib.sha256((ROOT/'campaign-plan.json').read_bytes()).hexdigest()==d['base_plan_sha256']
 plan=json.loads((ROOT/'campaign-plan.json').read_text());matches=[x for x in plan['shards']if x['arm']==arm and x['shard']==0]
 assert len(matches)==1 and config==matches[0],'Exact frozen shard0 config required'
 if arm not in d['changes']:return config['canary_task_ids']
 s=d['changes'][arm];assert config['shard']==0 and s['preserved_original_task_ids']==config['task_ids'][:2] and s['fresh_canary_task_ids']==config['task_ids'][2:4]
 ref=d['terminal_audit'];assert hashlib.sha256((ROOT/ref['path']).read_bytes()).hexdigest()==ref['sha256']
 root=ROOT/'arms'/arm/'shard-00/attempts/attempt-000/episodes'
 for tid in s['preserved_original_task_ids']:
  row=json.loads((root/(tid+'-browsercode-'+arm)/'result.json').read_text())
  assert row['task_id']==tid and row['job_id']==s['original_job_id'] and row['valid']is False and row['reward']is None and row['mode']==arm and row['harness']=='browsercode'
 return s['fresh_canary_task_ids']
