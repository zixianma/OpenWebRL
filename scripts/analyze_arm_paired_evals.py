#!/usr/bin/env python3
"""Paired fixed-checkpoint inference from saved stealth outcomes; no model calls.

OM2W resamples tasks, retaining all three repeats and methods for each task.
The null swaps the complete response vectors of two methods within a task.
Integer success-count differences permit an exact sign-flip distribution.
WebVoyager has one repeat, so the same exact test reduces to exact McNemar.
Only aggregate statistics and plots are public; per-task records stay in runtime.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from scipy import stats

from report_arm_om2w_difficulty import load_records, read, sha

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
RESULTS = REPO / 'openwebrl/docs/arm_results/rl_integration'
METHODS = ['baseline', 'additive', 'gate-b']
LABELS = ['Outcome-only', 'Additive', 'Gate B']
PAIRS = [(1, 0), (2, 0), (2, 1)]
SEED = 20260930


def exact_signflip(difference):
    """Exact two-sided paired-vector label-swap p; integer arithmetic for tails."""
    d = np.asarray(difference)
    assert np.all(d == np.rint(d)) and d.ndim == 1
    weights = np.abs(d.astype(int))
    weights = weights[weights > 0]
    distribution = np.ones(1)
    for w in weights:
        following = np.zeros(len(distribution) + 2 * w)
        following[:len(distribution)] += .5 * distribution
        following[2*w:] += .5 * distribution
        distribution = following
    support = np.arange(len(distribution)) - weights.sum()
    return float(min(1., distribution[np.abs(support) >= abs(int(d.sum()))].sum()))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order]))
    return adjusted.tolist()


def verified_record(record, path, judge, prompt):
    m = record['metrics']
    assert record['judge_model'] == judge and record['judge_prompt_variant'] == prompt
    assert m['trajectories'] == 1 and m['valid_trajectories'] in (0, 1)
    assert m['successes'] in (0, 1) and m['successes'] <= m['valid_trajectories']
    assert m['invalid_trajectories'] == 1-m['valid_trajectories']
    archive = (path.parent / record['rollout_file']).resolve()
    assert archive.is_relative_to(path.parent.resolve()) and archive.stat().st_size > 0
    if m['valid_trajectories']:
        assert record['reward_metadata']['judge_text'].strip()
    return int(m['successes']), int(m['valid_trajectories'])


def load_om2w():
    source = RUNTIME / 'reference-arm-stealth90-o4-t06-20260929/online_mind2web_monitor.jsonl'
    metadata = {str(t['metadata']['task_id']): t['metadata']
                for t in map(json.loads, source.read_text().splitlines())}
    ids = sorted(metadata)
    assert len(ids) == 300
    official = read(RESULTS / 'stealth-o4-t06-iteration90-three-repeat-summary.json')
    success = np.zeros((300, 3, 3), dtype=int)  # task, method, repeat
    valid = np.zeros_like(success)
    sources, private = [], []
    for mi, method in enumerate(METHODS):
        for cohort in official['methods'][method]['per_repeat']:
            rep = cohort['repeat']
            audit_path = RESULTS / cohort['source']
            assert sha(audit_path) == cohort['sha256']
            audit = read(audit_path)
            if rep == 1:
                original = load_records(Path(audit['original_root']))
                root = Path(audit['recovery_root'])
                recovery = load_records(root)
                expected_retry = set(read(root / 'evaluation_manifest.json')['provider_recovery']['expected_task_ids'])
                assert set(original) == set(ids) and set(recovery) == expected_retry
                assert all(original[k][0]['metrics']['valid_trajectories'] == 0 for k in expected_retry)
                records = original | recovery
            else:
                root = RUNTIME / audit['root']
                records = load_records(root)
            assert set(records) == set(ids)
            manifest = read(root / 'evaluation_manifest.json')
            assert manifest['completed_iteration'] == 90 and manifest['actor_temperature'] == .6
            for ti, ident in enumerate(ids):
                record, path = records[ident]
                success[ti, mi, rep-1], valid[ti, mi, rep-1] = verified_record(record, path, 'o4-mini', 'agenttrek')
                private.append(dict(task_id=ident, method=method, repeat=rep,
                                    record=str(path), sha256=sha(path)))
            assert success[:, mi, rep-1].sum() == cohort['successes']
            assert valid[:, mi, rep-1].sum() == cohort['valid']
            sources.append(dict(file=audit_path.name, sha256=sha(audit_path)))
    sites = [urlsplit(metadata[i]['start_url']).hostname.lower().removeprefix('www.') for i in ids]
    return success, valid, sites, sources, private


def load_webvoyager():
    success, valid, sources, private = [], [], [], []
    ids = None
    for method in METHODS:
        path = RESULTS / f'webvoyager-gpt4o-t06-{method}-iteration90-audit.json'
        audit = read(path)
        root = RUNTIME / audit['root']
        records = load_records(root)
        if ids is None:
            ids = sorted(records)
        assert set(ids) == set(records) and len(ids) == 595
        pairs = []
        for ident in ids:
            record, record_path = records[ident]
            pairs.append(verified_record(record, record_path, 'gpt-4o', 'webvoyager'))
            private.append(dict(task_id=ident, method=method, repeat=1,
                                record=str(record_path), sha256=sha(record_path)))
        s, v = np.array(pairs).T
        assert s.sum() == audit['successes'] and v.sum() == audit['valid']
        success.append(s); valid.append(v)
        sources.append(dict(file=path.name, sha256=sha(path)))
    return np.stack(success, axis=1)[:, :, None], np.stack(valid, axis=1)[:, :, None], sources, private


def summarize(success, valid, *, resamples, sites=None):
    n, _, repeats = success.shape
    means = success.mean(axis=2)
    quantities = np.column_stack([means] + [means[:, a]-means[:, b] for a, b in PAIRS])
    boot = stats.bootstrap((quantities.T,), np.mean, axis=-1, vectorized=True,
                           method='BCa', confidence_level=.95, n_resamples=resamples,
                           batch=256, rng=np.random.default_rng(SEED))
    low, high = boot.confidence_interval
    results = dict(tasks=n, repeats=repeats, methods={}, comparisons=[])
    for i, name in enumerate(METHODS):
        rates = success[:, i].mean(axis=0)
        results['methods'][name] = dict(overall_mean=float(means[:, i].mean()),
            overall_ci95=[float(low[i]), float(high[i])], per_repeat_overall=rates.tolist(),
            per_repeat_successes=success[:, i].sum(axis=0).tolist(),
            per_repeat_valid=valid[:, i].sum(axis=0).tolist(),
            valid_only_mean=float((success[:, i].sum(axis=0)/valid[:, i].sum(axis=0)).mean()))
    for index, (a, b) in enumerate(PAIRS):
        d = success[:, a].sum(axis=1)-success[:, b].sum(axis=1)
        common = np.all(valid[:, [a, b], :] == 1, axis=(1, 2))
        row = dict(comparison=f'{METHODS[a]} minus {METHODS[b]}',
            delta_pp=float(100*d.mean()/repeats),
            delta_ci95_pp=[float(100*low[3+index]), float(100*high[3+index])],
            p_exact=exact_signflip(d),
            task_count_difference_histogram={str(k):int(v) for k, v in zip(*np.unique(d, return_counts=True))},
            common_valid_sensitivity=dict(tasks=int(common.sum()),
                delta_pp=float(100*d[common].mean()/repeats), p_exact=exact_signflip(d[common])),
            per_repeat_delta_pp=(100*(success[:, a]-success[:, b]).mean(axis=0)).tolist())
        if repeats > 1:
            delta = np.array(row['per_repeat_delta_pp'])
            half = stats.t.ppf(.975, repeats-1)*stats.sem(delta)
            row['repeat_level_t_sensitivity'] = dict(n=repeats,df=repeats-1,
                ci95_pp=[float(delta.mean()-half),float(delta.mean()+half)],
                limitation='Only three evaluation windows; normal/independent repeat-difference assumptions are poorly checkable. Not training seeds.')
        results['comparisons'].append(row)
    for row, p in zip(results['comparisons'], holm([r['p_exact'] for r in results['comparisons']])):
        row['p_holm_three_pairs'] = p
    if sites is not None:
        unique = sorted(set(sites))
        sums = np.stack([quantities[np.array(sites) == site].sum(axis=0) for site in unique])
        counts = np.array([sites.count(site) for site in unique])
        rng = np.random.default_rng(SEED+1)
        draws = []
        for start in range(0, resamples, 256):
            pick = rng.integers(len(unique), size=(min(256,resamples-start), len(unique)))
            draws.append(sums[pick].sum(axis=1)/counts[pick].sum(axis=1)[:,None])
        limits = np.quantile(np.concatenate(draws), [.025,.975], axis=0)
        results['website_cluster_sensitivity'] = dict(websites=len(unique),
            method='Percentile bootstrap of start-URL hostname clusters, retaining all tasks/repeats/methods; task-weighted ratio of cluster totals.',
            comparison_ci95_pp=(100*limits[:,3:].T).tolist(),
            limitation='Website-level sensitivity; different task mixtures across draws. Hostnames are not necessarily independent organizations.')
    return results


def plot(result, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False})
    colors = ['#2563eb','#16a34a','#7e22ce']
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), gridspec_kw={'width_ratios':[1,1.25]})
    for i, method in enumerate(METHODS):
        row = result['methods'][method]
        mean = 100*row['overall_mean'];lo,hi = 100*np.array(row['overall_ci95'])
        axes[0].errorbar(mean,2-i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[i],capsize=5,ms=8,lw=2)
        axes[0].text(73,2-i+.19,f'{mean:.2f}% [{lo:.2f}, {hi:.2f}]',va='bottom',ha='right',fontsize=10)
    axes[0].set(yticks=[2,1,0],yticklabels=LABELS,xlim=(45,74),ylim=(-.55,2.55),xlabel='Overall task success (%)',title='Three-repeat means · 95% CIs')
    for i,row in enumerate(result['comparisons']):
        mean=row['delta_pp'];lo,hi=row['delta_ci95_pp'];color=colors[PAIRS[i][0]]
        axes[1].errorbar(mean,2-i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=5,ms=8,lw=2)
        axes[1].text(11,2-i+.19,f'{mean:+.2f} pp [{lo:+.2f}, {hi:+.2f}] · Holm p={row["p_holm_three_pairs"]:.3f}',ha='right',va='bottom',fontsize=9)
    axes[1].axvline(0,color='#64748b',ls='--',lw=1)
    axes[1].set(yticks=[2,1,0],yticklabels=['Additive − outcome-only','Gate B − outcome-only','Gate B − additive'],xlim=(-6,11.5),ylim=(-.55,2.55),xlabel='Paired success-rate difference (percentage points)',title='Paired differences · 95% CIs')
    for ax in axes:
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Online-Mind2Web · iteration 90 · stealth / o4-mini / T0.6',fontsize=14,fontweight='bold')
    fig.text(.02,.04,'300 task clusters; three repeats kept together. BCa 95% CIs are pointwise; p-values use Holm correction across three pairs.',fontsize=9,color='#475569')
    fig.text(.02,.005,'Uncertainty across tasks for fixed checkpoints and observed evaluation windows; not training-seed uncertainty. Invalid attempts count as failures.',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.10,1,.94),w_pad=2)
    fig.savefig(stem.with_suffix('.png'),dpi=200)
    fig.savefig(stem.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resamples',type=int,default=50000)
    parser.add_argument('--private-output',type=Path,required=True)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    args.private_output.mkdir(parents=True,exist_ok=True)
    s,v,sites,sources,private=load_om2w()
    om=summarize(s,v,resamples=args.resamples,sites=sites)
    ws,wv,wsources,wprivate=load_webvoyager()
    web=summarize(ws,wv,resamples=args.resamples)
    report=dict(iteration=90,primary='OM2W full300 overall mean over three repeats; invalid=0',
        inference_unit='Task; repeats are never treated as900 independent tasks.',
        test='Exact two-sided task-wise method-vector swap test; sign-flip distribution by convolution of integer differences. With one repeat this equals exact McNemar.',
        null_assumption='Method response vectors exchangeable within each independent task under the null; conditional on the observed evaluation windows.',
        confidence_interval=f'{args.resamples:,} paired task bootstrap draws, BCa pointwise95%; fixed checkpoints/windows.',
        multiplicity='Holm correction across the three method pairs separately within each benchmark; WebVoyager secondary.',
        resamples=args.resamples,seed=SEED,om2w=om,webvoyager=web,
        caveats=['No independent training seeds; not a causal significance claim about training recipes.',
                 'Checkpoint90 was considered after inspecting local learning curves; analysis is exploratory, not preregistered.',
                 'Task bootstrap does not capture arbitrary shared website/time shocks; hostname-cluster and three-window t intervals are sensitivity checks.',
                 'Repeat1 includes approved credit-blocked recovery; no valid original outcome replaced.',
                 'Common-valid sensitivity restricts to tasks valid for both methods in every repeat; differs from marginal valid-only rates.',
                 'Non-significance is not proof of equivalence; pointwise95% CIs do not encode Holm familywise decisions.'],
        sources=sources+wsources,implementation_sha256=sha(Path(__file__)),
        references=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html',
                    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html',
                    'https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html',
                    'https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/mcnemar.htm'],
        new_model_calls=0,new_gpu_hours=0)
    (args.private_output/'task-provenance.json').write_text(json.dumps(private+wprivate,indent=2)+'\n')
    target=RESULTS/'stealth-iteration90-paired-inference.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    if args.plot:plot(om,RESULTS/'stealth-iteration90-confidence')
    for label,result in [('OM2W',om),('WebVoyager',web)]:
        print(label)
        for row in result['comparisons']:print(json.dumps(row))
    print(target)


if __name__=='__main__':main()
