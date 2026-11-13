"""Recover released Kev-4B system costs from the runs linked by ARM_JEV_KEV_CROSSBENCH.md.
No model or API execution. Only aggregate counts are exported; trajectories remain private.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/jev-kev-crossbench-eval-md-20261111/run1')
ACTOR_PARAMS = 4022468096 + 415347712
# Qwen3.5-4B-Base model.language_model tensor elements, excluding vision and MTP.
# Kev's AutoModelForCausalLM(...).model uses this text backbone; LoRA is merged at serving.
KEV_PARAMS = 4205751296
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')


def episode(path):
    record = json.loads(path.read_text())
    out = dict(actor_prompt=0, actor_cached=0, actor_generated=0, selector_input=0,
               selector_calls=0, decision_seconds=0., steps=record['browser_steps'],
               state_hashes=[], missing_selector_usage=0, traces=0)
    actor_dirs = set()
    for entry in record.get('trace_files', []):
        trace = Path(entry['path'])
        assert trace.is_file(), trace
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == entry['sha256'], trace
        actor_dirs.add(trace.parent.parent / 'actor-requests')
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            out['traces'] += 1
            out['decision_seconds'] += row['timing']['decision_seconds']
            raw = row.get('selector_raw') or {}
            usage = raw.get('usage') or {}
            if 'input_tokens' in usage:
                out['selector_calls'] += 1
                out['selector_input'] += usage['input_tokens']
            else:
                out['missing_selector_usage'] += 1
            h = (row.get('selector_telemetry') or {}).get('state_sha256')
            if h:
                out['state_hashes'].append(h)
    for directory in actor_dirs:
        for request in directory.glob('*.json'):
            m = json.loads(request.read_text()).get('meta') or {}
            out['actor_prompt'] += m.get('prompt_tokens') or 0
            out['actor_cached'] += m.get('cached_tokens') or 0
            out['actor_generated'] += m.get('completion_tokens') or 0
    return out


def main():
    result = dict(label='Kev-4B', source_doc='ARM_JEV_KEV_CROSSBENCH.md#jev-kev-selectors-20261111',
                  run_root=str(RUNTIME), actor_parameters=ACTOR_PARAMS, selector_parameters=KEV_PARAMS,
                  system_parameters=ACTOR_PARAMS + KEV_PARAMS,
                  accounting='Actor PFLOP = 2 × actor params × (prompt − cached + generated tokens) / 1e15. Selector PFLOP proxy = 2 × text-backbone params × reported input tokens / 1e15. Serialized response tokens are excluded: Kev scores choices without generating text. Model seconds = summed generation + selection decision seconds / recorded episodes; browser time excluded.',
                  caveats=['Selector cache savings are not exposed in responses; all selector input tokens are charged. Selector FLOPs are a full-input proxy, not an exact hardware count.',
                           'System parameters exclude the small pointer head and merged LoRA overhead, matching the backbone-only parameter convention of the other selectors.',
                           'Run 1 costs from November 11; reported success rates supplied by user. Recorded-episode cost denominators are reported separately from scheduled-task success denominators.'],
                  reported_mean_overall=39.1, benchmarks={})
    for b, success in zip(BENCH, (31.3, 42.5, 43.3)):
        paths = sorted((RUNTIME / b / 'kev5-4b' / b / 'collection/records').glob('*.json'))
        with ThreadPoolExecutor(4) as pool:
            eps = list(pool.map(episode, paths))
        assert eps
        sums = {k: sum(e[k] for e in eps) for k in eps[0] if k != 'state_hashes'}
        assert sums['missing_selector_usage'] == 0, sums
        hashes = [h for e in eps for h in e['state_hashes']]
        actor_pf = 2 * ACTOR_PARAMS * (sums['actor_prompt'] - sums['actor_cached'] + sums['actor_generated']) / 1e15
        selector_pf = 2 * KEV_PARAMS * sums['selector_input'] / 1e15
        row = dict(overall=success, recorded_episodes=len(eps), actor_pflop_per_episode=actor_pf/len(eps),
                   selector_pflop_per_episode=selector_pf/len(eps), pflop_per_episode=(actor_pf+selector_pf)/len(eps),
                   model_seconds_per_episode=sums['decision_seconds']/len(eps), totals=sums,
                   selector_state_hashes=len(hashes), unique_selector_state_hashes=len(set(hashes)))
        result['benchmarks'][b] = row
        print(b, {k:v for k,v in row.items() if k != 'totals'}, flush=True)
    (HERE / 'selection-kev-costs-20261113.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
