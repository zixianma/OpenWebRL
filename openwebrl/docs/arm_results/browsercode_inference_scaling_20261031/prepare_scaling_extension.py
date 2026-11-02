#!/usr/bin/env python3
"""Prepare a local-cohort plot extension from sanitized qualified inputs.

No task IDs, runtime paths, trajectories or billing projections are exported.
Stealth results are a separate population; this script never supplies placeholder
stealth coordinates.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_scaling_extension_renderer', HERE/'plot_browsercode_scaling.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
BMAP = {'online-mind2web':'online_mind2web', 'webvoyager':'webvoyager', 'deepshop':'deepshop'}
CAVEAT = 'Own valid-task means; frozen native usage pricing upper bound excludes Terra and is not an invoice. Actor latency excludes Terra. Missing task outcomes are never imputed.'
PROMPT_CAVEAT = 'Historical 60-decision single enforced 60 but advertised 30; new single120 enforces and advertises 120. Live windows and validity subsets differ.'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metric(mean, valid, observed=None):
    observed = valid if observed is None else observed
    return {'mean':mean, 'observed_tasks':observed, 'missing_valid_tasks':valid-observed}

def common(benchmark, method, row, source_sha, *, single=False):
    candidates, decisions = {'parallel2_select_high':(2,30), 'parallel5_select_high_60steps':(5,60), 'luna_single_high_120steps':(1,120)}[method]
    valid, requested = row['valid'], row['planned']
    native = row['native_saved']
    claimed = 0 if single else row['claimed_missing_native']
    admin = 0 if single else row.get('administrative_missing_native',0)
    protocol = {'model':'gpt-6-luna', 'reasoning_effort':'high', 'max_output_tokens':12288, 'max_decisions':decisions, 'actor_wall_alarm_seconds':900, 'selection_candidates':candidates}
    if single: protocol['prompt_max_decisions'] = 120
    return {'benchmark':BMAP[benchmark], 'method':method, 'harness':'browsercode', 'completion':'verified_disposition_endpoint',
            'requested':requested, 'recorded':native, 'native_saved':native, 'valid':valid,
            'invalid':row['invalid'] if single else row['native_invalid'], 'claimed_missing_native':claimed,
            'administrative_missing_native':admin, 'unclaimed':0 if single else row['unclaimed'],
            'successes':row['successes'], 'success_rate':row['successes']/valid,
            'overall_success_rate':row['successes']/requested, 'overall_is_confirmed_success_lower_bound':claimed>0,
            'collection_window':{'status':'not_exported'}, 'evidence_sha256':source_sha,
            'browser_commands_evidence_sha256':source_sha, 'actor_protocol':protocol, 'metric_caveat':CAVEAT}

def selection_points(data, source_sha, *, candidates):
    plot.need(data['schema'] == f'sanitized-browsercode-luna-high-N{candidates}-max{60 if candidates==5 else 30}-terra-results-v1', 'Unexpected selection consumer schema')
    plot.need((data['actor'],data['reasoning_effort'],data['harness'],data['selection_candidates'],data['max_decisions'],data['max_output_tokens'],data['judge']) == ('luna','high','browsercode',candidates,60 if candidates==5 else 30,12288,'terra'), 'Selection protocol differs')
    plot.need(data['planned']==1045 and data['unclaimed']==0 and data['native_saved']==1044, 'Qualified selection endpoint coverage differs')
    rows=data['rows']
    plot.need(len(rows)==3 and {r['benchmark'] for r in rows}==set(BMAP), 'Three unique benchmark consumers required')
    method='parallel5_select_high_60steps' if candidates==5 else 'parallel2_select_high'
    output=[]
    for row in rows:
        p=common(row['benchmark'],method,row,source_sha)
        v=p['valid']
        plot.need(abs(row['valid_only_success_percent']/100-p['success_rate'])<1e-12 and abs(row['overall_success_percent']/100-p['overall_success_rate'])<1e-12, 'Consumer rate denominators differ')
        p['metrics']={
            'actor_usd':metric(row['actor_cost_upper_usd_per_valid_task'],v),
            'browser_commands':metric(row['selected_code_batches_per_valid_task'],v),
            'actor_seconds':metric(row['actor_latency_seconds_per_valid_task'],v),
            'output_tokens':metric(None,v,0), 'reasoning_tokens':metric(None,v,0)}
        output.append(p)
    for key in ('native_saved','valid','native_invalid','successes','claimed_missing_native','unclaimed'):
        plot.need(sum(r[key] for r in rows)==data[key], 'Selection consumer total mismatch: '+key)
    return output

def single_points(data, source_sha):
    plot.need(data['schema']=='private-single120-sanitized-qualified-consumer-v1' and data['method']=='luna_single_high_120steps', 'Qualified Single120 consumer required')
    plot.need(data['publication_ready'] is True and data['independent_final_qualification_pending'] is False and data['qualified_for_aggregate_results_integration'] is True and plot.digest(data['independent_final_qualification_sha256']), 'Single120 independent qualification required')
    plot.need(data['protocol']=={'max_decisions':120,'prompt_max_decisions':120,'output_tokens':12288,'reasoning_effort':'high','actor_wall_seconds':900,'terra_effort':'medium','terra_output_tokens':4096,'terra_seed':42}, 'Single120 protocol differs')
    plot.need(set(data['benchmarks'])==set(BMAP), 'Three Single120 benchmarks required')
    output=[]
    names={'actor_usd':'actor_known_priced_usd','browser_commands':'selected_code_batches','actor_seconds':'task_latency_seconds','output_tokens':'actor_output_tokens','reasoning_tokens':'actor_reasoning_tokens'}
    for b,row in data['benchmarks'].items():
        p=common(b,'luna_single_high_120steps',row,source_sha,single=True)
        plot.need(abs(row['valid_only_success_rate']-p['success_rate'])<1e-12 and abs(row['overall_success_rate']-p['overall_success_rate'])<1e-12, 'Single120 denominator differs')
        p['metrics']={name:metric(row['valid_task_means'][source]['mean'],p['valid'],row['valid_task_means'][source]['observations']) for name,source in names.items()}
        output.append(p)
    return output

def prepare(base, single, n5, n2, hashes, n5_qualification):
    plot.validate(base)
    plot.need(base['profile']=='completed-step-sweep' and len(base['points'])==24, 'Exact completed 24-point base required')
    plot.need(n5_qualification['status']=='VERIFIED_COMPLETE_AUTHORIZED107_UNCLAIMED_RECOVERY' and n5_qualification['all_paid_attempts_and_owned_children_verified_terminal'] is True and n5_qualification['independently_qualified_finite_aggregation_and_caps'] is True, 'N5 endpoint qualification required')
    plot.need(n5_qualification['sanitized_paper_consumer_sha256']==hashes['parallel5_select_high_60steps'], 'N5 consumer is not bound to final endpoint')
    plot.need(plot.digest(n2.get('independent_aggregation_review_sha256')) and plot.digest(n2.get('independent_native_terra_payload_review_sha256')), 'N2 independent endpoint/payload qualification required')
    old=copy.deepcopy(base['points']);data=copy.deepcopy(base)
    data.update(schema='browsercode-scaling-paper-v4',profile='qualified-decision-sweep')
    data['points']+=selection_points(n2,hashes['parallel2_select_high'],candidates=2)
    data['points']+=selection_points(n5,hashes['parallel5_select_high_60steps'],candidates=5)
    data['points']+=single_points(single,hashes['luna_single_high_120steps'])
    data['extension_metadata']={'publication_state':'qualified_local_endpoints_released',
        'base_aggregate_sha256':hashes['base'],
        'base_points_sha256':hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'source_consumer_sha256':{k:hashes[k] for k in plot.EXTENSIONS},'added_methods':list(plot.EXTENSIONS),
        'paired_comparisons_for_extensions':False,'prompt_limit_caveat':PROMPT_CAVEAT,'stealth_results_included':False}
    plot.validate(data)
    plot.need(data['points'][:24]==old,'Published 24-point dictionaries changed')
    return data

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('base','single120','n5','n2','n5-qualification','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    paths={'base':a.base,'luna_single_high_120steps':a.single120,'parallel5_select_high_60steps':a.n5,'parallel2_select_high':a.n2}
    d={k:json.loads(p.read_text()) for k,p in paths.items()}
    out=prepare(d['base'],d['luna_single_high_120steps'],d['parallel5_select_high_60steps'],d['parallel2_select_high'],{k:sha(p) for k,p in paths.items()},json.loads(a.n5_qualification.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'preview_only':False,'points':len(out['points']),'output_sha256':sha(a.output)}))

if __name__=='__main__':main()
