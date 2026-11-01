#!/usr/bin/env python3
"""Standalone renderer for completed, public, aggregate-only BrowserCode scaling data.

Python + matplotlib only. Main figure: three benchmarks by actor cost/latency.
Optional token figure: actual output and reasoning consumption, never token caps.
No runtime/source imports, credentials, task identifiers, API or browser calls.
The historical-baseline profile is explicitly a real-data layout preview; the
completed-sweep profile requires exactly seven methods on every benchmark.
"""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/openwebrl-scaling-matplotlib')
BENCHMARKS = {'online_mind2web': ('OM2W', 300), 'webvoyager': ('WebVoyager', 595), 'deepshop': ('DeepShop', 150)}
BASELINE = ('serial_high', 'luna_single_high', 'parallel3_select_high', 'parallel5_select_high')
NEW = ('luna_single_xhigh_12288', 'luna_single_high_24576', 'luna_single_high_6144')
METHODS = BASELINE + NEW
PAIRED_STATISTICS = {'population': 'common_valid_content_identical_tasks', 'interval': 'paired_task_bootstrap_percentile_95', 'multiple_testing_correction': 'none', 'coverage': 'available_eligible_pairs_only', 'collection_window_caveat': 'Pairing controls task identity, not collection time.'}
PROTOCOL = {'model': 'gpt-5.6-terra', 'reasoning_effort': 'medium', 'max_completion_tokens': 4096,
            'seed': 42, 'version': 'gpt56-terra-terminal-success-v1'}
# Model identity fixes shape: all Luna methods circles, Sol square. Color denotes
# single/selection; blue shade/fill distinguishes effort/cap within Luna single.
STYLES = {'serial_high': ('Sol high single', 's', '#08519c', True),
          'luna_single_high': ('Luna high (1× cap)', 'o', '#4292c6', True),
          'parallel3_select_high': ('Luna high N3', 'o', '#a50f15', False),
          'parallel5_select_high': ('Luna high N5', 'o', '#a50f15', True),
          'luna_single_xhigh_12288': ('Luna xhigh (1× cap)', 'o', '#08306b', True),
          'luna_single_high_24576': ('Luna high (2× cap)', 'o', '#2171b5', True),
          'luna_single_high_6144': ('Luna high (0.5× cap)', 'o', '#9ecae1', True)}
METRICS = ('actor_usd', 'actor_seconds', 'output_tokens', 'reasoning_tokens')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_accounting(d, accounting):
    roles = ('actor_generation', 'input_token_count', 'judge')
    need(type(accounting) is dict and set(accounting) == set(roles), 'Accounting must cover generation, token-counter and judge roles separately')
    for role in roles:
        row = accounting[role]
        need(type(row) is dict and set(row) == {'unknown_usage_calls', 'authorized_quarantined_unknown_usage_calls', 'retained_unknown_usd', 'active_reserved_calls'}, 'Exact role-specific accounting fields required')
        for key in ('unknown_usage_calls', 'authorized_quarantined_unknown_usage_calls', 'active_reserved_calls'):
            need(integer(row[key]), 'Nonnegative accounting count required')
        need(row['active_reserved_calls'] == 0, 'Active reservations cannot be marked complete')
        need(row['unknown_usage_calls'] == row['authorized_quarantined_unknown_usage_calls'], 'Every terminal unknown must be explicitly quarantined')
        need(type(row['retained_unknown_usd']) is str, 'Retained unknown reserve needs exact decimal text')
        try:
            retained = Decimal(row['retained_unknown_usd'])
        except InvalidOperation:
            raise ValueError('Invalid unknown reserve decimal') from None
        need(retained.is_finite() and retained >= 0, 'Invalid retained unknown reserve')
        need(row['unknown_usage_calls'] > 0 or retained == 0, 'Unknown reserve without unknown calls')
        if role != 'input_token_count':
            need((retained > 0) == (row['unknown_usage_calls'] > 0), 'Paid unknown attempts retain nonzero reserve')
    need(accounting['actor_generation']['unknown_usage_calls'] == d['missing_output_usage_calls'], 'Unknown generations must remain missing output usage')


def validate(data):
    need(type(data) is dict and set(data) == {'schema', 'profile', 'judge_protocol', 'metric_population', 'comparison_caveat', 'points', 'paired_comparisons', 'paired_statistics', 'unavailable_prespecified_comparisons'}, 'Only documented public aggregate fields are allowed')
    need(data.get('schema') == 'browsercode-scaling-paper-v1', 'Unknown aggregate schema')
    need(data.get('profile') in ('historical-baseline', 'completed-sweep'), 'Unknown completeness profile')
    need(data.get('judge_protocol') == PROTOCOL and all(type(data['judge_protocol'][k]) is type(v) for k, v in PROTOCOL.items()), 'Shared Terra protocol required')
    need(data.get('metric_population') == 'resolved_valid_tasks', 'Valid-task metric population required')
    need(data.get('comparison_caveat') == 'Task validity subsets and live collection windows differ; historical-versus-fresh comparisons are descriptive.', 'Comparability caveat required')
    methods = METHODS if data['profile'] == 'completed-sweep' else BASELINE
    expected = {(b, m) for b in BENCHMARKS for m in methods}
    points = data.get('points')
    need(type(points) is list and len(points) == len(expected), 'Exact completed profile point count required')
    actual = [(p.get('benchmark'), p.get('method')) for p in points]
    need(len(set(actual)) == len(actual) and set(actual) == expected, 'Exact unique benchmark/method identities required')
    for p in points:
        allowed = {'benchmark', 'method', 'harness', 'completion', 'requested', 'recorded', 'valid', 'invalid', 'successes', 'success_rate', 'metrics', 'collection_window', 'evidence_sha256', 'actor_protocol', 'independent_completion_audit_sha256', 'all_attempt_generations', 'accounting_disposition'}
        need(set(p) <= allowed, 'Point contains undocumented/private fields')
        need(p.get('harness') == 'browsercode' and p.get('completion') == 'verified', 'Only verified BrowserCode cohorts may be rendered')
        for key in ('requested', 'recorded', 'valid', 'invalid', 'successes'):
            need(integer(p.get(key)), 'Nonnegative exact count required: ' + key)
        need(p['requested'] == BENCHMARKS[p['benchmark']][1] and p['recorded'] == p['requested'], 'Every planned cell needs a final disposition')
        need(0 < p['valid'] <= p['recorded'] and p['valid'] + p['invalid'] == p['recorded'] and p['successes'] <= p['valid'], 'Disposition counts disagree')
        need(digest(p.get('evidence_sha256')), 'Aggregate source/audit hash required')
        need(number(p.get('success_rate')) and abs(p['success_rate'] - p['successes']/p['valid']) <= 1e-12, 'Wrong valid-only success denominator')
        window = p.get('collection_window')
        need(type(window) is dict and set(window) <= {'status', 'start_epoch', 'end_epoch'}, 'Only aggregate collection window fields are allowed')
        need(window.get('status') in ('observed', 'not_exported'), 'Collection window status required')
        if window['status'] == 'observed':
            need(number(window.get('start_epoch')) and number(window.get('end_epoch')) and 0 < window['start_epoch'] <= window['end_epoch'], 'Invalid observed collection window')
        metrics = p.get('metrics')
        need(type(metrics) is dict and set(metrics) == set(METRICS), 'Exact actor metric set required')
        for name in METRICS:
            metric = metrics[name]
            need(type(metric) is dict and set(metric) == {'mean', 'observed_tasks', 'missing_valid_tasks'}, 'Metric mean/coverage schema differs')
            observed, missing = metric['observed_tasks'], metric['missing_valid_tasks']
            need(integer(observed) and integer(missing) and observed + missing == p['valid'], 'Metric coverage differs from valid count')
            need((observed == 0 and metric['mean'] is None) or (observed > 0 and number(metric['mean']) and metric['mean'] >= 0), 'Missing usage must remain missing')
            if name in ('actor_usd', 'actor_seconds'):
                need(observed == p['valid'] and metric['mean'] > 0, 'Main plotted actor metric needs full valid-task coverage')
        if p['method'] in NEW:
            cap = 12288 if p['method'].endswith('12288') else 24576 if p['method'].endswith('24576') else 6144
            effort = 'xhigh' if 'xhigh' in p['method'] else 'high'
            need(p.get('actor_protocol') == {'model': 'gpt-6-luna', 'reasoning_effort': effort, 'max_output_tokens': cap, 'max_decisions': 30}, 'Approved new actor protocol differs')
            need(window['status'] == 'observed' and digest(p.get('independent_completion_audit_sha256')), 'New cohorts need observed windows and independent audit')
            d = p.get('all_attempt_generations')
            need(type(d) is dict and set(d) == {'population', 'physical_generation_calls', 'known_output_usage_calls', 'missing_output_usage_calls', 'known_reasoning_usage_calls', 'output_at_cap_calls', 'known_stop_reason_calls', 'output_limit_stop_calls'}, 'Exact all-attempt generation diagnostics required for fresh cohorts')
            for key in ('physical_generation_calls', 'known_output_usage_calls', 'missing_output_usage_calls', 'known_reasoning_usage_calls', 'output_at_cap_calls', 'known_stop_reason_calls', 'output_limit_stop_calls'):
                need(integer(d.get(key)), 'Nonnegative all-attempt generation count required: ' + key)
            need(d['known_output_usage_calls'] + d['missing_output_usage_calls'] == d['physical_generation_calls'], 'All-attempt usage coverage mismatch')
            need(d['known_reasoning_usage_calls'] <= d['known_output_usage_calls'] and d['output_at_cap_calls'] <= d['known_output_usage_calls'], 'Cap/reasoning count exceeds usage observations')
            need(d['output_limit_stop_calls'] <= d['known_stop_reason_calls'] <= d['physical_generation_calls'], 'Stop-reason count exceeds observations')
            need(d.get('population') == 'all_actor_generation_attempts_including_invalid_episodes', 'Truncation diagnostics must include invalid attempts')
            validate_accounting(d, p.get('accounting_disposition'))
    need(data.get('paired_statistics') == PAIRED_STATISTICS, 'Explicit paired population/statistics/time caveat required')
    # Pairwise common-valid counts stay aggregate-only: no source identifiers.
    pairs = data.get('paired_comparisons', [])
    need(type(pairs) is list, 'Paired comparisons must be a list')
    identities = set()
    for p in pairs:
        need(set(p) == {'benchmark', 'left_method', 'right_method', 'common_valid', 'both_success', 'left_only_success', 'right_only_success', 'both_failure', 'excluded_invalid_or_missing', 'success_rate_difference', 'paired_bootstrap_95_percentile_interval'}, 'Only aggregate paired counts/intervals allowed')
        b, left, right = p.get('benchmark'), p.get('left_method'), p.get('right_method')
        need(b in BENCHMARKS and left in methods and right in methods and left != right, 'Unknown paired identity')
        identity = (b, *sorted((left, right)))
        need(identity not in identities, 'Duplicate paired comparison')
        identities.add(identity)
        n = p.get('common_valid')
        need(integer(n) and n > 0 and n <= BENCHMARKS[b][1], 'Invalid paired denominator')
        cells = [p.get(k) for k in ('both_success', 'left_only_success', 'right_only_success', 'both_failure')]
        need(all(integer(x) for x in cells) and sum(cells) == n, 'Paired outcome counts disagree')
        need(integer(p.get('excluded_invalid_or_missing')) and p['excluded_invalid_or_missing'] + n == BENCHMARKS[b][1], 'Paired exclusions must remain explicit')
        need(number(p.get('success_rate_difference')) and abs(p['success_rate_difference'] - (cells[1]-cells[2])/n) <= 1e-12, 'Paired effect disagrees with counts')
        interval = p.get('paired_bootstrap_95_percentile_interval')
        need(type(interval) is list and len(interval) == 2 and all(number(v) for v in interval) and -1 <= interval[0] <= interval[1] <= 1, 'Paired bootstrap interval required')
    unavailable = data.get('unavailable_prespecified_comparisons')
    need(type(unavailable) is list, 'Unavailable prespecified pair reasons required')
    absent = set()
    for row in unavailable:
        need(type(row) is dict and set(row) == {'benchmark', 'left_method', 'right_method', 'reason'}, 'Only unavailable-pair identity/reason fields allowed')
        b, left, right = row['benchmark'], row['left_method'], row['right_method']
        need(b in BENCHMARKS and left in methods and right in methods and left != right, 'Unknown unavailable pair identity')
        identity = (b, *sorted((left, right)))
        need(identity not in absent and identity not in identities, 'Contradictory or duplicate pair coverage')
        need(type(row['reason']) is str and row['reason'].strip(), 'Unavailable pair needs a concrete reason')
        absent.add(identity)
    if data['profile'] == 'completed-sweep':
        prespecified = {(b, *sorted((a, 'luna_single_high'))) for b in BENCHMARKS for a in NEW}
        prespecified |= {(b, *sorted((a, c))) for b in BENCHMARKS for i, a in enumerate(NEW) for c in NEW[i+1:]}
        need(absent == prespecified - identities, 'Every unavailable prespecified contrast needs an explicit reason')
    else:
        need(not absent, 'Baseline layout has no fresh-sweep prespecified comparisons')
    return data


def render(data, output_dir, stem='browsercode_scaling', *, tokens=False, token_methods=None):
    validate(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, NullFormatter, StrMethodFormatter
    columns = ('output_tokens', 'reasoning_tokens') if tokens else ('actor_usd', 'actor_seconds')
    labels = ('Output tokens / valid task', 'Reasoning tokens / valid task') if tokens else ('Actor cost ($, log)', 'Actor latency (s)')
    available = METHODS if data['profile'] == 'completed-sweep' else BASELINE
    need(tokens or token_methods is None, 'Token method subset requires token figure')
    if token_methods is not None:
        need(type(token_methods) in (list, tuple) and token_methods and len(set(token_methods)) == len(token_methods) and set(token_methods) <= set(available), 'Token subset must explicitly name unique available methods')
    selected = tuple(m for m in available if token_methods is None or m in token_methods)
    plotted = [p for p in data['points'] if p['method'] in selected]
    # An explicit token-only subset is allowed; no silent method omission.
    if tokens:
        for p in plotted:
            for name in columns:
                need(p['metrics'][name]['observed_tasks'] == p['valid'], 'Token figure requires full valid-task token coverage for every method')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.labelsize': 7.5,
                         'axes.titlesize': 8, 'axes.linewidth': .6, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'svg.hashsalt': 'openwebrl-browsercode-scaling-v1'})
    fig, axes = plt.subplots(3, 2, figsize=(7, 4.35), sharey='row', squeeze=False)
    fig.subplots_adjust(left=.077, right=.765, top=.90, bottom=.105, hspace=.53, wspace=.25)
    for i, (benchmark, (title, _)) in enumerate(BENCHMARKS.items()):
        rows = sorted((p for p in plotted if p['benchmark'] == benchmark), key=lambda p: METHODS.index(p['method']))
        rates = [100*p['success_rate'] for p in rows]
        low = max(0, 5*math.floor((min(rates)-1)/5))
        high = min(100, 5*math.ceil((max(rates)+1)/5))
        axes[i, 0].set_title(f'({"abc"[i]}) {title}', loc='left', weight='bold', pad=4)
        for j, name in enumerate(columns):
            ax = axes[i, j]
            ax.set_ylim(low, high)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(length=2.5, width=.6, pad=2)
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis='y', linewidth=.35, color='#dddddd', zorder=0)
            if name == 'actor_usd':
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if j == 0:
                ax.set_ylabel('Success (%)', labelpad=3)
            if i == 2:
                ax.set_xlabel(labels[j], labelpad=3)
            if i == 0:
                ax.text(.5, 1.20, labels[j], ha='center', transform=ax.transAxes, fontsize=7.5)
            for p in rows:
                _, marker, color, filled = STYLES[p['method']]
                ax.scatter(p['metrics'][name]['mean'], 100*p['success_rate'], s=27, marker=marker,
                           edgecolors=color, facecolors=color if filled else 'white', linewidths=.9, zorder=3)
    methods = selected
    handles = [Line2D([], [], linestyle='none', marker=STYLES[m][1], color=STYLES[m][2],
                      markerfacecolor=STYLES[m][2] if STYLES[m][3] else 'white', markeredgewidth=.9,
                      markersize=4.8, label=STYLES[m][0]) for m in methods]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.785, .55), frameon=False,
               fontsize=7.2, handletextpad=.3, labelspacing=.75, borderaxespad=0)
    note = 'Terra judged\nValid-task means\nSingle-call token caps\n1× = 12,288 tokens' + ('\nOutput includes reasoning' if tokens else '\nCaps ≠ actual usage')
    fig.text(.79, .20, note, fontsize=6.8, color='#555555', va='top')
    for extension in ('png', 'svg', 'pdf'):
        metadata = {'Date': None} if extension == 'svg' else {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {}
        fig.savefig(output_dir / f'{stem}.{extension}', dpi=300, metadata=metadata)
    plt.close(fig)
    fields = ['benchmark', 'method', 'requested', 'recorded', 'valid', 'invalid', 'successes', 'success_rate']
    fields += [f'{name}_{key}' for name in METRICS for key in ('mean', 'observed_tasks', 'missing_valid_tasks')]
    with (output_dir / f'{stem}_coordinates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for p in plotted:
            row = {k: p[k] for k in fields[:8]}
            row.update({f'{name}_{key}': value for name, metric in p['metrics'].items() for key, value in metric.items()})
            writer.writerow(row)
    return {'schema': 'browsercode-scaling-render-v1', 'profile': data['profile'], 'points': len(plotted), 'selected_methods': list(selected),
            'dimensions_inches': [7, 4.35], 'token_figure': tokens,
            'omitted_token_method_coverage': [
                {'benchmark': p['benchmark'], 'method': p['method'], 'valid': p['valid'],
                 'output_observed_tasks': p['metrics']['output_tokens']['observed_tasks'],
                 'reasoning_observed_tasks': p['metrics']['reasoning_tokens']['observed_tasks']}
                for p in data['points'] if tokens and p['method'] not in selected]}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stem', default='browsercode_scaling')
    parser.add_argument('--tokens', action='store_true')
    parser.add_argument('--token-method', action='append', choices=METHODS, help='Explicit token-only subset; repeat per method. Omitted-method coverage is exported in provenance.')
    args = parser.parse_args()
    need(args.stem and Path(args.stem).name == args.stem and args.stem not in ('.', '..'), 'Output stem must be a filename')
    raw = args.aggregate.read_bytes()
    data = validate(json.loads(raw))
    result = render(data, args.output_dir, args.stem, tokens=args.tokens, token_methods=args.token_method)
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    result['renderer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['matplotlib_version'] = __import__('matplotlib').__version__
    result['output_sha256'] = {extension: hashlib.sha256((args.output_dir / (args.stem+extension)).read_bytes()).hexdigest() for extension in ('.png', '.svg', '.pdf', '_coordinates.csv')}
    (args.output_dir / (args.stem+'_render_provenance.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
