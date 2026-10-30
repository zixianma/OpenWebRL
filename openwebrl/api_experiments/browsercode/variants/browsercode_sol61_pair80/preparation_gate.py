"""Fail-closed plan, task-grid, vendor and approval validation."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def validate(require_approval=False):
 p=json.loads((ROOT/"campaign-plan.json").read_text()); tasks=json.loads((ROOT/"tasks.json").read_text()); study=json.loads((ROOT/"study-tasks.json").read_text())
 assert digest({k:v for k,v in p.items() if k!="plan_sha256"})==p["plan_sha256"],"plan hash"
 assert len(tasks)==80 and len({x["task_id"] for x in tasks})==80,"extension task set"
 assert len(study)==100 and len({x["task_id"] for x in study})==100,"100-task study set"
 assert hashlib.sha256((ROOT/"tasks.json").read_bytes()).hexdigest()==p["extension_task_manifest_sha256"],"extension manifest hash"
 assert hashlib.sha256((ROOT/"study-tasks.json").read_bytes()).hexdigest()==p["study_task_manifest_sha256"],"study manifest hash"
 expected={(t,h,m) for s in p["shards"] for t in s["task_ids"] for h,m in [(c["harness"],c["policy"]) for c in p["cells"]]}
 assert len(expected)==320 and len({x[0] for x in expected})==80,"320 fixed extension grid"
 assert all(len(s["task_ids"])==20 and s["episode_count"]==80 for s in p["shards"]),"shard grid"
 assert sum(sum(1 for t in s["task_ids"] if s["task_orders"][t][0]==c) for s in p["shards"] for c in p["cells"])==80,"task orders"
 for n,w in p["preparation_source_hashes"].items(): assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==w,n
 v=json.loads((ROOT/"browsercode/source-manifest.json").read_text()); assert v["revision"]==p["browsercode_revision"]
 for n,w in v["files"].items(): assert hashlib.sha256((ROOT/"browsercode"/n).read_bytes()).hexdigest()==w,n
 off=json.loads((ROOT/"offline-native-preflight.json").read_text()); assert off["results"]["execution"] and off["results"]["model_requests"]==0 and off["results"]["browser_sessions"]==0
 a=json.loads((ROOT/"approval.json").read_text())
 if require_approval and (not a.get("approved") or a.get("plan_sha256")!=p["plan_sha256"]): raise PermissionError("Exact approved resource/API caps required")
 return {"valid":True,"approved":a.get("approved"),"study_tasks":100,"new_tasks":80,"new_episodes":320,"shards":4}
if __name__=="__main__":
 import argparse
 q=argparse.ArgumentParser();q.add_argument("--require-approval",action="store_true");args=q.parse_args();print(json.dumps(validate(args.require_approval)))
