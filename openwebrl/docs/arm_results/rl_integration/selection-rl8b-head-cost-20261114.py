"""Audit run-1 cost for RL 8B + default 22.6M SFT-feature head.
Actor FLOPs = 2 * RL8B params * (uncached prompt + generated tokens).
Selector FLOPs are the logged frozen-SFT4B re-encoding + head FLOPs.
Model latency sums decision_seconds, excluding browser time. Scheduled-task
means include page-load failures with no decisions. Success is the published
two-run overall result; cost is run 1, matching existing cost-plot accounting.
Only aggregates are exported; task payloads and model responses remain local.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-rl8b-heads')
SLUG = 'head5x-91e7a5a67417'
ACTOR, ENCODER, HEAD = 8767123696, 4437815808, 22562818
BENCH = {'online_mind2web':300, 'webvoyager':595, 'deepshop':150}

def episode(path):
    raw=path.read_bytes(); r=json.loads(raw)
    out=dict(actor_flops=0, selector_flops=0, seconds=0., steps=0, requests=0,
             trace_count=0, unmetered_failed_requests=0, fingerprint=hashlib.sha256(raw).hexdigest())
    request_files=set()
    for trace in {t['path'] if isinstance(t,dict) else t for t in r.get('trace_files') or []}:
        q=Path(trace)
        assert q.is_file(), q
        out['trace_count']+=1
        for line in q.read_text().splitlines():
            x=json.loads(line); t=x['timing']; s=x['selector_telemetry']
            assert s['head_sha256'].startswith('91e7a5a67417')
            expected=2*ENCODER*(s['input_tokens']+s['candidate_tokens'])+2*HEAD*5
            assert s['flops']==expected, (s['flops'],expected)
            out['selector_flops']+=s['flops'];out['seconds']+=t['decision_seconds'];out['steps']+=1
        request_files.update((q.parent.parent/'actor-requests').glob('*.json'))
    for q in sorted(request_files):
        request=json.loads(q.read_text()); meta=request.get('meta') or {}
        if not meta:
            assert request.get('error_type'), q
            out['unmetered_failed_requests']+=1
            continue
        assert all(k in meta for k in ['prompt_tokens','completion_tokens','cached_tokens']), q
        tokens=meta['prompt_tokens']-(meta['cached_tokens'] or 0)+meta['completion_tokens']
        assert tokens>=0
        out['actor_flops']+=2*ACTOR*tokens;out['requests']+=1
    assert not out['steps'] or out['requests']+out['unmetered_failed_requests']>=out['steps']*5
    return out

if __name__=='__main__':
    success=json.loads((HERE/'selection-rl8b-heads-20261114.json').read_text())['conditions']['head-22.6M']
    output=dict(label='RL 8B + ADM-23M', plot_key='rl8b-head-gpt55-k3',
                system_parameters=ACTOR+ENCODER+HEAD,
                parameters=dict(actor=ACTOR,frozen_sft_encoder=ENCODER,head=HEAD),
                source_success='selection-rl8b-heads-20261114.json',cost_run=1,success_runs=2,
                accounting=__doc__,reported_mean_overall=100*success['mean_overall'],benchmarks={})
    with ThreadPoolExecutor(max_workers=8) as pool:
        for bench,expected in BENCH.items():
            paths=sorted((ROOT/'run1'/bench/SLUG/bench/'collection'/'records').glob('*.json'))
            assert len(paths)==expected,(bench,len(paths))
            records=list(pool.map(episode,paths))
            sums={key:sum(r[key] for r in records) for key in ['actor_flops','selector_flops','seconds','steps','requests','trace_count','unmetered_failed_requests']}
            output['benchmarks'][bench]=dict(cost_tasks=expected,
                overall=100*success[bench]['overall'],
                pflop_per_episode=(sums['actor_flops']+sums['selector_flops'])/expected/1e15,
                model_seconds_per_episode=sums['seconds']/expected,
                steps_per_episode=sums['steps']/expected,
                cost_totals=sums, records_sha256=hashlib.sha256(''.join(r['fingerprint'] for r in records).encode()).hexdigest())
            print(bench,output['benchmarks'][bench],flush=True)
    output['mean_pflop_per_episode']=sum(r['pflop_per_episode'] for r in output['benchmarks'].values())/3
    output['mean_model_seconds_per_episode']=sum(r['model_seconds_per_episode'] for r in output['benchmarks'].values())/3
    output['caveat']='Different evaluation window from existing RL references; 73 tasks failed page load in every head run. Overall success is not a matched estimate of adding the head. Frozen SFT-4B re-encoding is included in system parameters, FLOPs and latency. Failed actor requests without token telemetry are counted separately and excluded from FLOPs, making FLOPs a lower bound if those requests performed model work.'
    Path(__file__).with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Mean',output['reported_mean_overall'],output['mean_pflop_per_episode'],output['mean_model_seconds_per_episode'],flush=True)
