"""Create native-audit-v5 from the frozen native-audit-v4 (v4 stays untouched; it is hash-bound by v8 packages).

v5 independently restates the v9 science (user 2026-10-26 "follow that config ... read the papers again to confirm.
also let's do B"): one Table 5 site per cohort (shopping / shopping_admin / reddit / gitlab), the WebArbiter /
WebShepherd Best-of-20 top-5-frequency candidate protocol (re-aggregated from the saved 20-sample replies during the
whole-coordinator replay), site-dispatched reset audits (Magento: the reviewed audit_cms_reset chain of the running
package), and state disposal for cells invalid only because the retired fuzzy judge could not be called, after their
closure/reset audit and with their evaluator transcript retained (the outcome stays invalid).
"""
import shutil
from pathlib import Path

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
V4, V5 = D/'native-audit-v4', D/'native-audit-v5'
V5.mkdir(exist_ok=False)
for n in ('native_cell_audit.py', 'native_lowlevel_replay.py'):
    shutil.copy2(V4/n, V5/n)


def patch(path, pairs):
    t = path.read_text()
    for a, b in pairs:
        assert t.count(a) == 1, (path.name, a[:100])
        t = t.replace(a, b)
    path.write_text(t)


patch(V5/'native_cell_audit.py', [
    ('"""Independent offline audit of a prospective original-coordinator Reddit cell.',
     '"""Independent offline audit of a prospective original-coordinator v9 Table 5 cell (Shopping/CMS/Reddit/GitLab).'),
    ("SCHEMA='native_reddit54_v8_independent_cell_audit_v4'",
     "SCHEMA='native_table5_v9_independent_cell_audit_v5'\n"
     "# v5 (2026-10-26): v9 cohorts. Independently restated: one released single-site Table 5 site per cohort, Map excluded,\n"
     "# Reddit task 5 dropped; arbiter7b/shepherd8b candidates are the 5 most frequent valid actions of 20 samples\n"
     "# (n=20, temperature 1, top_p 1, single-action envelope); no_search unchanged.\n"
     "V9_PROTOCOL='released_coordinator_original_browser_table5_v9'\n"
     "V9_SITE_TASKS={'shopping':45,'shopping_admin':35,'reddit':18,'gitlab':30}\n"
     "V9_RECEIPT=dict(samples=20,top_n=5,sampling=dict(n=20,temperature=1.0,top_p=1.0),prompt='disclosed_single_action_envelope_v2',\n"
     " ranking='frequency_desc_then_first_sampled',validity='released_filter_candidates',confidence='sample_share_of_valid',\n"
     " methods=['arbiter7b','shepherd8b'],no_search='unchanged_v8')\n"
     "V9_FUZZY_ERROR='Native fuzzy transport/accounting failed; no retry'\n"
     "MAGENTO={'shopping','shopping_admin'}"),
    (" require(len(cells)==len(indexed)==54 and indexed.get(cell.get('cell_id'))==cell,'Fixed54 cell membership differs')",
     " domains={c['domain'] for c in cells};require(len(domains)==1 and next(iter(domains)) in V9_SITE_TASKS,'One v9 Table 5 site per cohort');site=next(iter(domains))\n"
     " require(len(cells)==len(indexed)==3*V9_SITE_TASKS[site] and indexed.get(cell.get('cell_id'))==cell,'Fixed v9 cell membership differs')"),
    (" released=read(reference(taskref));reddit=[t for t in released if t.get('sites')==['reddit']]\n"
     " require(len(released)==165 and len(reddit)==19,'Original released165/singleReddit19 membership differs')\n"
     " rawmap={t['task_id']:t for t in reddit if t['task_id'] not in DROPPED_TASKS};require(len(rawmap)==18,'Reddit18 membership differs');selectors={'no_search','arbiter7b','shepherd8b'}",
     " released=read(reference(taskref));single=[t for t in released if t.get('sites')==[site]]\n"
     " require(len(released)==165,'Original released165 membership differs')\n"
     " rawmap={t['task_id']:t for t in single if t['task_id'] not in DROPPED_TASKS};require(len(rawmap)==V9_SITE_TASKS[site],'v9 site task membership differs');selectors={'no_search','arbiter7b','shepherd8b'}"),
    ("  require(c['domain']=='reddit' and c['task_config_sha256']==expected,'Released task hash/domain differs')",
     "  require(c['domain']==site and c['task_config_sha256']==expected,'Released task hash/domain differs')"),
    (" require(plan.get('protocol')==cohort.get('protocol')=='released_coordinator_original_browser_reddit19_v1','Prospective native protocol differs')",
     " require(plan.get('protocol')==cohort.get('protocol')==V9_PROTOCOL,'Prospective native protocol differs')"),
    (" require(context['run_id']==expected_run and context['domain']=='reddit' and context['domain_knowledge']==(native.history.load_domain_knowledge('reddit') or '')",
     " site=next(iter(plan['endpoints']))\n"
     " require(context['run_id']==expected_run and context['domain']==site and context['domain_knowledge']==(native.history.load_domain_knowledge(site) or '')"),
    ("max_attempts=512,selector=method,no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT if method=='no_search' else None,v8=V8_RECEIPT)\n require(cfg==expected,",
     "max_attempts=512,selector=method,no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT if method=='no_search' else None,v8=V8_RECEIPT,v9=V9_RECEIPT if method!='no_search' else None)\n require(cfg==expected,"),
    (" replay=load('_native54_v8_independent_lowlevel_replay_v4',helper)",
     " replay=load('_native_table5_v9_independent_lowlevel_replay_v5',helper)"),
    ("  instance=legacy.audit_reddit_reset(dict(inputs,cell=dict(cell,selector=method)),paths,plan,result)\n  bundle=",
     "  instance=v9_site_reset(inputs,dict(inputs,cell=dict(cell,selector=method)),paths,plan,result,legacy)\n  bundle="),
    (" else:\n  require(result.get('error_type') or result.get('error') or result.get('watchdog',{}).get('reason'),'Invalid outcome lacks explicit saved error')",
     " else:\n  require(result.get('error_type') or result.get('error') or result.get('watchdog',{}).get('reason'),'Invalid outcome lacks explicit saved error')\n"
     "  if result.get('attempted') and result.get('error_type')=='EvaluationNotReady' and result.get('error')==V9_FUZZY_ERROR:\n"
     "   # v5: the retired gpt-4-1106-preview judge alone made this cell invalid. Its site state may be disposed once its\n"
     "   # reset/instance chain verifies and the evaluator transcript (offline re-score input) is retained; still invalid.\n"
     "   require({'evaluator_inputs','instance','lease','copy_receipt','state_claim','native_result'}<=paths.keys(),'Fuzzy-unavailable cell lacks its evaluator transcript or reset chain')\n"
     "   v9_site_reset(inputs,dict(inputs,cell=dict(cell,selector=method)),paths,plan,result,legacy)\n"
     "   from openwebrl.webprm_webarena import evaluation_requirements\n"
     "   require(evaluation_requirements(task.native_config())['paid_fuzzy_judge_may_be_required'],'Fuzzy-unavailable error on a task without a fuzzy judge')\n"
     "   fuzzy_disposable=True"),
    (" native_replay=None;bundle=None;outcome=",
     " native_replay=None;bundle=None;fuzzy_disposable=False;outcome="),
    ("selector=method,domain='reddit',task_config_sha256=", "selector=method,domain=cell['domain'],task_config_sha256="),
    ("complete_for_disposal=bool(result['valid'] and native_replay and native_replay['verified'] and bundle and bundle['verified']))",
     "complete_for_disposal=bool((result['valid'] and native_replay and native_replay['verified'] and bundle and bundle['verified']) or fuzzy_disposable),\n"
     "  fuzzy_judge_unavailable_state_disposable=fuzzy_disposable)"),
    ("def audit_identity(inputs,plan,cohort):",
     "def v9_site_reset(inputs,inner,paths,plan,result,legacy):\n"
     " site=inner['cell']['domain']\n"
     " if site not in MAGENTO:return legacy.audit_reddit_reset(inner,paths,plan,result)\n"
     " matches=[Path(p) for p in inputs['source_sha256'] if p.endswith('/scripts/audit_webprm_table5_cms_cell.py') and Path(p).resolve().is_relative_to(ROOT.resolve())]\n"
     " require(len(matches)==1 and sha(matches[0])==inputs['source_sha256'][str(matches[0])],'One source-bound Magento reset auditor required')\n"
     " magento=load('native_v9_frozen_magento_reset_helpers',matches[0])\n"
     " require(magento.site_module(site) is not None,'Magento reset auditor lacks this site')\n"
     " return magento.audit_cms_reset(inner,paths,plan,result)\n\n"
     "def audit_identity(inputs,plan,cohort):"),
])

patch(V5/'native_lowlevel_replay.py', [
    ("def require(v,m):",
     "# v5 (2026-10-26): independent re-implementation of the v9 Best-of-20 top-5-frequency candidate protocol\n"
     "# (WebShepherd App. F.2 / WebArbiter Sec. 5.2.1) for arbiter7b/shepherd8b policy replies.\n"
     "V9_N,V9_TOP=20,5\n"
     "def v9_candidates(response,observation,sampling,one):\n"
     " from collections import Counter\n"
     " choices=response.get('choices');require(isinstance(choices,list) and len(choices)==V9_N,'v9 reply must carry 20 samples')\n"
     " valid=[]\n"
     " for i,c in enumerate(choices):\n"
     "  try:\n"
     "   p=json.loads(v7_strip(one(dict(choices=[c],usage=response.get('usage')))));b=p.get('BLOCKS') if isinstance(p,dict) else None\n"
     "   if not isinstance(b,list) or len(b)!=1 or not isinstance(b[0],dict):continue\n"
     "   th=b[0].get('thought').strip() if isinstance(b[0].get('thought'),str) else '';ac=b[0].get('action').strip() if isinstance(b[0].get('action'),str) else ''\n"
     "   if th and ac and sampling.filter_candidates([sampling.BlockInfo(thought=th,action=ac,confidence=None)],observation):valid.append((i,th,ac))\n"
     "  except Exception:continue\n"
     " n=Counter(a for _,_,a in valid);first={}\n"
     " for i,th,a in valid:first.setdefault(a,(i,th))\n"
     " top=sorted(n,key=lambda a:(-n[a],first[a][0]))[:V9_TOP]\n"
     " return json.dumps(dict(BLOCKS=[dict(thought=first[a][1],action=a,confidence=n[a]/len(valid),frequency=n[a]) for a in top]))\n\n"
     "def require(v,m):"),
    ("   if method=='no_search' and value['purpose']=='native_policy':\n"
     "    value['upstream_unadapted_payload']=deepcopy(value['payload']);value['payload']['messages'][0]['content']=single_action_system_prompt(value['payload']['messages'][0]['content'],version=ENVELOPE_NO_SEARCH_PROMPT);value['policy_adaptation']='disclosed_single_action_envelope_v2'\n",
     "   if value['purpose']=='native_policy':\n"
     "    value['upstream_unadapted_payload']=deepcopy(value['payload']);value['payload']['messages'][0]['content']=single_action_system_prompt(value['payload']['messages'][0]['content'],version=ENVELOPE_NO_SEARCH_PROMPT);value['policy_adaptation']='disclosed_single_action_envelope_v2'\n"
     "    if method!='no_search':value['payload'].update(n=20,temperature=1.0,top_p=1.0);value['policy_adaptation']='v9_best_of_20_top5_frequency'\n"),
    ("  def response_text(self,response):\n   text,usage=super().response_text(response)\n",
     "  def response_text(self,response):\n"
     "   if response.get('_webprm_broker',{}).get('purpose')=='native_policy' and method!='no_search':\n"
     "    require('text' in observed,'v9 policy reply replayed without its observation')\n"
     "    one=lambda value:super(ReplayBridge,self).response_text(value)[0]\n"
     "    super().response_text(dict(choices=[dict(finish_reason='stop',message=dict(role='assistant',content='{}'))],usage=response.get('usage')))\n"
     "    return v9_candidates(response,observed['text'],native.tools['sampling'],one),deepcopy(response.get('usage'))\n"
     "   text,usage=super().response_text(response)\n"),
    (" async def adapter(*,native,agent,context,inputs):\n  stash={};",
     " observed={}\n async def adapter(*,native,agent,context,inputs):\n  observed['text']=inputs['observation_text'];stash={};"),
    ("geometry=o['geometry'];require(geometry==dict(screenWidth=V8_W,screenHeight=V8_H,innerWidth=V8_W,innerHeight=V8_H,devicePixelRatio=1),'Native screen/viewport/DPR differs from v8 1024x2490')",
     "geometry=o['geometry'];require(geometry==dict(screenWidth=V8_W,screenHeight=V8_H,innerWidth=V8_W,innerHeight=V8_H,devicePixelRatio=1),'Native screen/viewport/DPR differs from v8 1024x2490')"),
])
print('native-audit-v5 written')
