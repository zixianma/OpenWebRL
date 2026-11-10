#!/usr/bin/env python3
"""Build a private, inspectable critic source catalog and aggregate public table.

This inventories source families and derived views; counts must not be summed
as independent data. Large torch archives are not deserialized.
"""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path

R=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
REPO=Path(__file__).resolve().parents[1]
OUT=R/'arm-turn-bonus-preparation/outcome-reward-20260927/critic-comparison-20261004/source-review'
HF='https://huggingface.co/datasets/PTeterwak/action-reward-models-data/tree/main/'


def read(p):return json.loads(Path(p).read_text())
def lines(p):
    with Path(p).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    piotr=R/'arm-reproduction/joint-data-inputs-0d83b48/openwebrl_actor'
    joint=R/'arm-reproduction/joint-data-v2-20260910'
    c2=R/'arm-reproduction/runs/c2-full-282782-20260908T075414Z'
    outcome=R/'arm-turn-bonus-preparation/outcome-reward-20260927'
    refresh=R/'arm-turn-bonus-preparation/arm-refresh-20260924/baseline40'
    pool=R/'arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929'
    bench={str(r['metadata']['task_id']) for r in lines(REPO/'openwebrl/data/eval/online-mind2web.jsonl')}
    future=list(lines(refresh/'data/future.jsonl'))
    future_ids={str(r['task_id']) for r in future}
    assert len(future)==371 and len(future_ids)==82 and future_ids<=bench
    raw_baseline=defaultdict(int);expanded=defaultdict(int);provenance=[]
    for f in sorted(R.glob('runs/*/completed_rollout_archive/iteration_*/manifest.json')):
        a=read(f)
        if not a['complete']:continue
        dest=expanded if f.parents[2].name.startswith('outcome-only-expanded') else raw_baseline
        for key in ('groups','trajectories'):dest[key]+=a[key]
        dest['manifests']+=1
        provenance.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                               iteration=a['reward_iteration'],trajectories=a['trajectories']))
    arm_archives=defaultdict(lambda:dict(collection_markers=0,label_files=0,group_files=0,group_bytes=0))
    examples={}
    for f in sorted(R.glob('evaluations/*/iterations/*/collection_complete.json')):
        name=f.parents[2].name
        if 'diagnostic' in name or 'calibration' in name: family='diagnostics_exclude'
        elif name.startswith('arm-mixed-bonus'):family='mixed_bonus'
        elif name.startswith('arm-mixed-reweight'):family='mixed_reweight'
        elif name.startswith(('arm-failure','arm-gate','arm-ablation','arm-additive')):family='additive_gate_ablation'
        elif name.startswith('arm-turn-bonus-fresh-allfailure'):family='allfailure'
        elif name.startswith('arm-turn-bonus-fresh'):family='original_bonus'
        else:family='other_unclassified'
        a=arm_archives[family];a['collection_markers']+=1
        a['label_files']+=sum(1 for _ in (f.parent/'labels').glob('*.json'))
        for p in (f.parent/'groups').glob('*/*.pt'):
            a['group_files']+=1;a['group_bytes']+=p.stat().st_size
        examples.setdefault(family,str(f.parent))
    catalog=[]
    def add(ident,name,count,label,pref,value,post,adv,readiness,caveat,refs,fields,parent=None):
        catalog.append(dict(id=ident,name=name,count=count,label=label,preference=pref,value_q=value,
            post_action=post,advantage=adv,readiness=readiness,caveat=caveat,
            references=[str(p) for p in refs],fields=fields,derived_from=parent))
    add('P1','Piotr OpenWebRL raw states/candidates/labels','3,085 states;412 episodes;49,536 candidate draws /49,360 IDs',
        'Teacher winner among five unexecuted proposals','Yes','No measured returns','No for sampled alternatives','No',
        'Local; identity and schema audited','176 conflicting duplicate draw records; repeated draws are not new states; split by task/state, not draw.',
        [piotr/'states_full.jsonl',piotr/'candidates_merged.jsonl',piotr/'labels_drawlevel.jsonl'],
        'state_id,episode_id,turn,prompt_text,images,demo_action; candidates; selection,reasoning')
    add('P2','Piotr source demonstration adjacency','2,673 adjacent pairs across412 episodes',
        'Executed source demonstration, without saved binary terminal verdict','Requires relabeling','No current return labels','Potential after demo lineage audit','No',
        'Local states; not converted to critic transitions','After-image belongs to demo_action, not the later preferred proposal; demonstration quality is not a progress label.',
        [piotr/'states_full.jsonl'],'episode_id,turn,prompt_text,images,demo_action',parent='P1')
    add('P3','Piotr packaged selection SFT / scalar BT','Remote prepared train/validation files; row counts not re-audited',
        'Same teacher choices as P1, reformatted','Yes','No calibrated success values','No','No',
        'Remote derivative; not new evidence','Original draw-level validation can share states across splits; rebuild task-disjoint splits for new claims.',
        [HF+'openwebrl_actor/selection_sft',HF+'openwebrl_actor/scalar_rm'],'ShareGPT/image selection targets or chosen/rejected preference branches',parent='P1')
    add('M1','MolmoWeb PRM / preference data','reward_pairs_15k.jsonl,101.6MB; filename count not independently verified',
        'GPT-5.5 per-candidate PRM scores','Yes','Teacher score only, not return','No observed branches established','No measured advantage',
        'Remote schema sampled','Different actor/action syntax; screenshot restoration is a prerequisite; teacher counterfactual scoring does not prove executed counterfactual returns.',
        [HF+'molmoweb_actor/reward_pairs_15k.jsonl'],'state_id,screenshot,cands[{system,user_pre,user_post,score,action_str}]')
    add('M2','MolmoWeb actor-distillation rounds','Four published state/candidate/selection/SFT files',
        'PRM-selected candidate; original demonstration action','Yes','No binary outcomes verified','No observed candidate branches verified','No',
        'Remote schemas sampled; screenshots referenced in external archive','Actor-distribution shift and missing archived screenshots; selection/SFT views are derivatives of the same rounds.',
        [HF+'molmoweb_actor/actor_distillation'],'state_id,sample_id,bucket,task,step_idx,url,screenshot,history,gt_action_str; candidates; selections')
    c=read(c2/'collection-summary.json')
    add('C1','C2 complete guided collection',f"{c['attempted']:,} episodes;{c['valid']:,} valid;{c['successes']:,} successes /805 valid failures",
        'Frozen SelectionARM choices plus terminal o4-mini/AgentTrek verdict','Yes, model preferences','Yes, after conversion','Executed path only; adjacency audit needed','Derived, not directly observed',
        'Complete collection and outcome audit already available','Continuation policy includes ARM at every step; success prediction is for that composite policy, not ordinary SFT. Keep failures.',
        [c2/'collection-summary.json',c2/'results',c2/'selections',c2/'samples'],'result.reward,valid; candidate thoughts/actions,selected_index; saved causal contexts/images')
    add('C2','C2 successful-turn SFT view','8,394 usable turns from1,151 successes',
        'Executed ARM winner from successful episode','Can rejoin candidate traces','Positive-only: inadequate alone','Can rejoin underlying episode','No direct labels',
        'Local audited derivative','Success filtering removes negative outcomes; chosen/rejected candidate is a preference contrast, not successful/failed counterfactual.',
        [c2/'training.jsonl',c2/'dataset-audit.json'],'task_id,turn,source,source_sha256,prompt_token_ids,image_grid_thw,selected_index',parent='C1')
    add('J1','Cleaned joint C2 + Piotr preference pairs','5,540 train /658 validation: C2 3,464/442; Piotr2,076/216',
        'Frozen ARM winner for C2; teacher winner for Piotr','Yes','C2 positive-only; Piotr unobserved','Underlying source joins required','No direct labels',
        'Local, task-group split and ambiguity/dedup audited','Not an additional raw collection. 120 normalized task texts shared across sources; preserve the common split.',
        [joint/'joint_pairs.train.jsonl',joint/'joint_pairs.validation.jsonl',joint/'audit.json',joint/'split-manifest.jsonl'],
        'task,task_group,split,prompt,images,chosen,rejected,label_source,terminal_success,source_path',parent='P1+C2')
    add('R0','Outcome-only baseline raw completed archives',f"{raw_baseline['trajectories']:,} archived trajectories in{raw_baseline['manifests']} collection manifests (iterations24–100)",
        'Native GPT-4.1/action_history terminal outcome','No alternative preference unless joined','Yes, after validity/split audit','Yes where successor observation retained','Cross-fitted MC/TD targets',
        'Manifest inventory; full eligible count beyond R1 pending','Includes invalid attempts and all outcome-group types; behavior changes with checkpoint. Manifest count is not usable-example count.',
        [R/'runs',outcome/'manifest.json'],'trajectories[valid,reward,success,turns[prompt,response,metadata,multimodal_inputs]]')
    add('R1','Prepared outcome-critic corpus','2,000 train /250 dev /500 later test;17,584 prefixes;14,825 matched transitions',
        'Valid binary terminal return; no local correctness labels','Not direct preference','Yes, ready indices','Yes:11,023/1,239/2,563 matched transitions','Derived; cross-fit V first',
        'Source and screenshot hashes verified; causal resolver tested','Derived from R0. Preserve task splits. Final-turn after-images absent; future changes both tasks and actor. Never expose broadcast terminal metadata.',
        [outcome/'critic-comparison-20261004/audit.json',outcome/'critic-comparison-20261004/train.jsonl',outcome/'train.jsonl',outcome/'dev.jsonl',outcome/'future.jsonl'],
        'source_group,trajectory_offset,turn_index,terminal_return,before_images,after_images,matched_pre_post_eligible; advantage_label=null',parent='R0')
    add('R2','Expanded4,102-task outcome-only archives',f"{expanded['trajectories']:,} archived trajectories in{expanded['manifests']} manifests (collection iterations3–21)",
        'Native terminal outcomes','No direct preference','Potential after audit','Potential after conversion','Derived only',
        'Manifest snapshot; training still incomplete','These are collection indices, not verified optimizer progress. Early browser-failure cohort must stay distinguishable; filter invalids and freeze splits before fitting.',
        [R/'runs/outcome-only-expanded-4102-20261003/completed_rollout_archive'],'Same complete-group archive schema as R0')
    for family,title in [('original_bonus','Original mixed-outcome ARM bonus'),('allfailure','Historical failure-only ARM'),
                         ('additive_gate_ablation','Additive / Gate B/C / beta / sampling ablations'),
                         ('mixed_bonus','Fresh mixed-only bonus'),('mixed_reweight','Fresh mixed-only outcome reweighting')]:
        a=arm_archives[family]
        add('A-'+family,title,f"{a['collection_markers']} collection markers;{a['label_files']:,} label files;{a['group_files']:,} saved group payload files",
            'Frozen ARM shadow preference + original native outcome in joined payload','Yes where label admitted','After extracting native outcome','After recovering retained tensors/images/successor links','Derived only; current bonus is not advantage',
            'Filesystem/schema inventory only; usable deduplicated counts pending','Retries/resume snapshots may overlap. Files are not trajectories. Exclude diagnostics, identify actor checkpoint/gate and separate native return from shaped training reward.',
            [examples[family]],'labels:prompt,candidates,permutation,selected_index,eligible,policy_id,context[group_index,parent_sample_index]; groups/*.pt: joined rollout payload')
    screen=pool/'actor-screen-min5-2000/sharded-20261002';guided=pool/'actor-arm-allfailure-20261002'
    add('S1','New2,000-task ordinary actor screen','10,000 episodes;9,936 valid;3,597 successes;five attempts/task',
        'Original SFT + GPT-4.1/action_history outcomes','No candidate preference','Strong clean behavior-policy source','Requires lightweight archive conversion','MC/TD derivation; only root task repeats, not same intermediate state',
        'Completed archive/verdict audit; transition conversion pending','Five attempts share task identity: keep them in the same split. Attempts with native reward−1 are actor-format failures, not successes.',
        [screen/'agent-completion-review.json',screen/'shard-0/records',screen/'shard-0/attempts'],
        'record:task_id,attempt,seed,policy_id,judge_id,valid,reward,artifact,steps,tokens; trajectory .pt + verdict .json')
    add('S2','Guided rescue after five valid failures','682 episodes;679 valid;78 successes',
        'SelectionARM choices and native outcome','Yes, model preferences','Yes for guided failure-selected behavior','Executed path only, after conversion','Derived only',
        'Completed rollout/verdict/candidate audit; conversion pending','Subset of S1 tasks selected by five failures; do not compare its yield to those conditioning failures as a fresh control.',
        [guided/'agent-completion-review.json',guided/'shard-0/records',guided/'shard-0/attempts'],
        'S1 record fields plus selector_revision,candidate_count,candidate_representation; candidate traces',parent='S1 task subset; new episodes')
    add('S3','Small late-actor rescue pilot','320 screen +48 fresh retry episodes;6 separate smoke episodes',
        'Iteration90 ordinary/guided actor + native outcome','Guided traces only','Yes, small conditional panel','After conversion','Derived; no same-state branches',
        'Completed audit;64 training tasks, eight retry tasks','Exclude smoke from scientific fit/evaluation; actor differs from SFT, and retry tasks were failure-selected.',
        [R/'evaluations/arm-rescue-yield-313264/yield-summary.json',R/'evaluations/arm-rescue-yield-313264/trajectories'],
        'phase,mode,attempt,valid,reward,seed,artifact; saved trajectories/candidate traces')
    add('S4','SFT actor /2-turn /4-turn ARM-prefix pilot','288 episodes;48 tasks ×3 conditions ×2 repeats;283 valid',
        'Native outcomes and intervention-prefix labels','Only during guided prefix','Yes with behavior condition retained','After conversion','Derived; continuation changes at prefix boundary',
        'Complete; training tasks exclude OM2W','Useful controlled policy-mixture data, not same-state branching. Keep all conditions for a task in one split.',
        [R/'evaluations/arm-prefix-pilot-331932/completion-audit.json',R/'evaluations/arm-prefix-pilot-331932/sft'],
        'prefix condition,task,attempt,terminal return,actor_suffix_turns,trace/trajectory artifacts')
    add('F1','Baseline40 ARM refresh: new preference states','2,000 train /250 dev states;plus858 Piotr replay;2,858 formatted train rows',
        'GPT-5.5 teacher selections on current actor candidates','Yes','Can join R0 outcomes for executed action only','Can join R0 demo transition; sampled alternatives unexecuted','No direct labels',
        'Local trained selection corpus; teacher labels complete','New sampled candidates are not the archived executed action. Piotr858 is reused P1 data, not new outcome supervision;200 Piotr retention rows are held out.',
        [refresh/'data/train.jsonl',refresh/'data/dev.jsonl',refresh/'data/replay.jsonl',refresh/'teacher/labels.jsonl',refresh/'llamafactory/train.jsonl'],
        'task,history,screenshot,candidates,responses,permutation,classes,source_group; teacher selected_slot/selected_original',parent='R0 states + P1 replay')
    add('E1','ARM refresh future preference panel','371 states on82 OM2W benchmark tasks;all82 IDs match benchmark',
        'Teacher choices; frozen-ARM choice; benchmark rollout outcome','Evaluation only','Evaluation only','Evaluation only if recoverable','No training',
        'Benchmark contamination check verified this audit','This future.jsonl is NOT R1 future.jsonl. It must not be folded into critic training.',
        [refresh/'data/future.jsonl',R/'evaluations/arm-task-success-314664/task-success-summary.json'],
        'record,archive,task_id,frozen_selected_index,candidates,rollout_valid,rollout_reward,split=future')
    add('E2','Held-out inference / checkpoint evaluations','Original900 OM2W episodes;many additional full300, stealth, WebVoyager595 cohorts;controlled1,800 planned',
        'Benchmark success verdicts; candidates in guided arms','Evaluation only','Evaluation only','Evaluation only','No training',
        'Per-cohort audits; do not pool different judges/protocols','Keep benchmark tasks and trajectories out of all critic training; repeated evaluations are not independent training examples.',
        [R/'arm-reproduction/runs/dedicated-282209-20260908T005547Z/full',R/'arm-turn-bonus-preparation/controlled-inference-20261004'],
        'task outcomes + rollout/screenshot artifacts; model,judge,sampling,attempt provenance')
    add('T0','Task pool / URL availability / instruction screening','Task instructions and screening metadata, not executed transitions',
        'Availability, quality, task-level difficulty','No action label','No trajectory return without S1/S2 joins','No','No',
        'Useful for task sampling and joins','Do not treat site availability or instruction-quality scores as action rewards.',
        [pool],'task_id,intent,start_url,exclusion/dedup flags,availability,quality scores')
    report=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),sources=catalog,
        raw_baseline=dict(raw_baseline),expanded=dict(expanded),arm_filesystem_inventory=dict(arm_archives),
        refresh_future_benchmark_overlap=dict(rows=len(future),tasks=len(future_ids),benchmark_tasks=len(future_ids&bench)),
        inventory_scope='Known ARM source families in the HF release and project artifacts; derived views and overlapping collections are explicitly identified. Counts have the verification levels shown.',
        default_split_rule='Group task aliases/normalized goals across ALL sources before splitting; preserve existing dev/test/retention sets and exclude reported benchmark tasks.',
        no_direct_advantage_labels=True)
    (OUT/'catalog.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'raw-archive-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    columns=['id','name','count','label','preference','value_q','post_action','advantage','readiness','caveat','derived_from','references','fields']
    with (OUT/'catalog.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in catalog:writer.writerow({**row,'references':' | '.join(row['references'])})
    # Private interactive inspection table. It contains source paths, not copied
    # raw tasks/trajectories; never include it in automatic public publication.
    cells=[]
    for row in catalog:
        links=''.join(f'<li><a href="{html.escape(ref,quote=True)}">{html.escape(ref)}</a></li>' for ref in row['references'])
        detail=f'<details><summary>Inspect fields and source artifacts</summary><p><b>Fields:</b> {html.escape(row["fields"])}</p><p><b>Derived from:</b> {html.escape(row["derived_from"] or "Raw/source family")}</p><ul>{links}</ul></details>'
        cells.append('<tr>'+''.join(f'<td>{html.escape(str(row[k]))}</td>' for k in columns[:10])+'<td>'+detail+'</td></tr>')
    page='''<!doctype html><meta charset="utf-8"><title>ARM critic source review</title>
<style>body{font:14px system-ui;margin:28px;color:#182339}h1{font-size:25px}input{padding:10px;width:650px}table{border-collapse:collapse;margin-top:20px}th{position:sticky;top:0;background:#e9eff8}td,th{text-align:left;vertical-align:top;padding:10px;border:1px solid #dce1e8}td{min-width:120px;max-width:340px}li{overflow-wrap:anywhere}details{min-width:220px}small{color:#586477}</style>
<h1>ARM critic data sources</h1><p>Counts overlap: raw collections and derived views are not additive. No source has measured counterfactual advantages for every candidate.</p>
<p><b>R1</b> is the prepared transition corpus. <b>E1/E2</b> are benchmark evaluation data. Expand each row to inspect fields and files.</p>
<input id="search" placeholder="Filter sources, e.g. Piotr, post, benchmark, mixed, ready"><small> Local artifact paths may be easier to open in your editor.</small>
<table><thead><tr>'''+''.join('<th>'+html.escape(k.replace('_',' '))+'</th>' for k in columns[:10])+'''<th>Inspect</th></tr></thead><tbody>'''+''.join(cells)+'''</tbody></table><script>document.getElementById('search').oninput=e=>{let q=e.target.value.toLowerCase();for(let r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(q)}</script>'''
    (OUT/'critic-source-review.html').write_text(page)
    # Public aggregate receipt contains no task content, local source paths, or
    # per-task IDs. Human-oriented Markdown remains in the existing topic doc.
    public={k:v for k,v in report.items() if k!='sources'}
    public['sources']=[{k:v for k,v in row.items() if k not in ('references','fields')} for row in catalog]
    (REPO/'openwebrl/docs/arm_results/rl_integration/critic-source-inventory-20261004.json').write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps(dict(sources=len(catalog),private_output=str(OUT),raw_baseline=dict(raw_baseline),expanded=dict(expanded),refresh_future_benchmark_overlap=report['refresh_future_benchmark_overlap']),indent=2))


if __name__=='__main__':main()
