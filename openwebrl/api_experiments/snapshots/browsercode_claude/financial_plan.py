"""Strict financial-only view. The scientific campaign-plan file/hash never changes.

No overlay without a separately sealed explicit user approval. This module reads
only; the operator owns activation at a source-qualified safe handoff.
"""
from copy import deepcopy
from pathlib import Path
from decimal import Decimal
import hashlib,json
ROOT=Path(__file__).resolve().parent
BASE_SHA='b10be00d54d2f5fb86cef9768f8286be97be56f5da306db6cff98298e6fcd949'
PROPOSED_SHA='ad419f3b1988078f659b28e291fdbebfed2f432edf1af5e19c467958eea943fd'
PROPOSED_PATH='analysis/bulk-api-budget-amendment-proposed-plan-v1.json'
APPROVAL_PATH='orchestration/financial-approval.json'
NEW_CAPS={'opus_single':21,'sonnet_n3':19,'sonnet_n5':22}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _unique(pairs):
 d={}
 for key,value in pairs:
  if key in d:raise ValueError('Duplicate JSON key')
  d[key]=value
 return d

def read(path):return json.loads(Path(path).read_text(),object_pairs_hook=_unique)
def _require(value,message):
 if not value:raise ValueError(message)
def _ref(root,ref):
 _require(type(ref)is dict and set(ref)=={'path','sha256'},'Typed exact evidence reference required')
 p=Path(ref['path']);p=p if p.is_absolute()else root/p;p=p.resolve()
 _require(p.is_relative_to(root.resolve())and p.is_file()and sha(p)==ref['sha256'],'Financial evidence missing or changed')
 return p

def expected_changes(base):
 return [{'arm':s['arm'],'shard':s['shard'],'old_model_usd':s['model_usd_all_attempts'],'new_model_usd':NEW_CAPS[s['arm']]}for s in base['shards']if s['arm']in NEW_CAPS and type(s['shard'])is int and 2<=s['shard']<30]

def _same(a,b):return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)

def qualify(base,proposed,approval,user):
 """Pure complete structural comparison: no tasks/protocol/resources/calls change."""
 changes=expected_changes(base);_require(len(changes)==84,'Exact84 bulk cap changes required')
 _require(type(approval)is dict and approval.get('schema')=='claude-bulk-financial-amendment-v1'and approval.get('approved')is True and approval.get('explicit_user_approval')is True,'Explicit financial approval required')
 _require(approval.get('base_plan_sha256')==BASE_SHA and approval.get('proposed_plan_sha256')==PROPOSED_SHA,'Pinned financial plan identities required')
 _require(_same(approval.get('cap_changes'),changes),'Exact84 approval cap changes required')
 totals={'model_usd':1996,'judge_usd':300,'combined_usd':2296,'additional_model_usd':196}
 for name,value in totals.items():
  _require(type(approval.get(name))is int and approval[name]==value,'Exact typed financial total '+name)
 _require(approval.get('transfers_allowed')is False,'No financial transfers')
 _require(user.get('schema')=='explicit-user-financial-approval-v1'and user.get('approved')is True and user.get('source_role')=='user'and isinstance(user.get('approval_text'),str)and bool(user['approval_text'].strip()),'User-authored explicit approval evidence required')
 _require(user.get('base_plan_sha256')==BASE_SHA and user.get('proposed_plan_sha256')==PROPOSED_SHA and _same(user.get('cap_changes'),changes),'User approval exact financial scope required')
 for name,value in totals.items():_require(type(user.get(name))is int and user[name]==value,'User approval typed financial total '+name)
 _require(user.get('transfers_allowed')is False,'User approval no transfers required')
 expected=deepcopy(base)
 for s in expected['shards']:
  if s['arm']in NEW_CAPS and 2<=s['shard']<30:s['model_usd_all_attempts']=NEW_CAPS[s['arm']]
 expected['api_caps'].update(model_usd=1996,judge_usd=300,combined_usd=2296)
 _require(_same(proposed,expected),'Financial amendment altered frozen science/resources/tasks/calls/pilots or unexpected fields')
 _require(sum(Decimal(str(s['model_usd_all_attempts']))for s in proposed['shards'])==1996 and sum(Decimal(str(s['judge_usd_all_attempts']))for s in proposed['shards'])==300,'Per-shard cap sum mismatch')
 return deepcopy(proposed)

def effective_plan(root=None):
 root=Path(root)if root is not None else ROOT;basepath=root/'campaign-plan.json';_require(sha(basepath)==BASE_SHA,'Frozen scientific base plan changed');base=read(basepath);approvalpath=root/APPROVAL_PATH
 if not approvalpath.exists():return base
 approval=read(approvalpath);_require(approval.get('approved')is True,'Present financial overlay is unapproved')
 proposedpath=root/PROPOSED_PATH;_require(sha(proposedpath)==PROPOSED_SHA,'Pinned proposed financial plan changed')
 userpath=_ref(root,approval.get('user_approval'));effective=qualify(base,read(proposedpath),approval,read(userpath))
 # Enrolled predecessors of these84 previously unused logical shards cannot be
 # silently upgraded. Only explicit enrollment under this exact approval enters.
 registry=root/'orchestration/registry.json'
 if registry.exists():
  for entry in read(registry).get('attempts',[]):
   if entry['arm']in NEW_CAPS and 2<=entry['shard']<30:
    _require(entry.get('financial_approval_sha256')==sha(approvalpath),'Affected bulk owner predates exact financial approval')
 for s in effective['shards']:
  if s['arm']not in NEW_CAPS or s['shard']<2:continue
  ledger=root/'arms'/s['arm']/f'shard-{s["shard"]:02d}'/'budget/model/ledger.json'
  if ledger.exists():
   state=read(ledger);_require(Decimal(str(state['max_usd']))==Decimal(str(s['model_usd_all_attempts'])),'Existing affected ledger cannot be silently upgraded')
 return effective

def enrollment_binding(root=None):
 root=Path(root)if root is not None else ROOT;effective_plan(root);p=root/APPROVAL_PATH
 return {'financial_approval_sha256':sha(p)if p.exists()else None}

def verify_enrollment(entry,root=None):
 root=Path(root)if root is not None else ROOT;effective=effective_plan(root);base=read(root/'campaign-plan.json');p=root/APPROVAL_PATH;binding=entry.get('financial_approval_sha256');cfg=lambda plan:next(s for s in plan['shards']if s['arm']==entry['arm']and s['shard']==entry['shard'])
 if binding is None:
  _require(cfg(base)==cfg(effective),'Legacy enrollment cannot inherit changed financial cap');expected=cfg(base)
 else:
  _require(p.exists()and binding==sha(p),'Enrollment financial approval changed');expected=cfg(effective)
 _require(_same(entry.get('shard_caps'),expected),'Enrolled shard caps differ from qualified financial scope')
 return deepcopy(expected)

def shard_config(arm,shard,root=None):
 matches=[s for s in effective_plan(root)['shards']if s['arm']==arm and s['shard']==shard]
 _require(len(matches)==1,'Unknown or duplicate arm/shard');return matches[0]
