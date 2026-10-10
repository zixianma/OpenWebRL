#!/usr/bin/env python3
"""Build a standalone, offline ARM RL chart from published aggregate results.

Run after adding an audit. No raw trajectories or private runtime paths are
embedded. Historical counts come from the published comparison data;
newer audited counts override them. Local and stealth protocols stay separate.
"""
import json
import html as html_module
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / 'openwebrl/docs'
AUDITS = DOCS / 'arm_results/rl_integration'
REMOTE = 'https://github.com/zixianma/OpenWebRL/blob/arm/openwebrl/docs/'
SPECS = [
    ('outcome_only', 'Outcome-only baseline', '#2563eb', True, 'baseline'),
    ('outcome56', 'Outcome-only · 56 groups (2,102 tasks)', '#111827', True, 'outcome56'),
    ('uniform8', 'Outcome-only · uniform G8, 48 groups (2,102 tasks)', '#0f766e', True, 'uniform8'),
    ('expanded4102', 'Outcome-only · expanded 4,102 tasks', '#ea580c', True, 'expanded4102'),
    ('all_failure', 'All-failure ARM', '#dc2626', True, 'allfailure'),
    ('additive', 'Additive ARM', '#16803c', True, 'additive'),
    ('gate_b', 'Gate B · relaxed gate', '#7e22ce', True, 'gate-b'),
    ('gate_c', 'Gate C · duplicate-aware credit', '#a16207', False, 'gate-c'),
    ('mixed_bonus', 'Mixed-only bonus · relaxed B', '#0891b2', True, 'mixed-bonus'),
    ('mixed_reweight', 'Mixed-only reweight · relaxed B', '#db2777', True, 'mixed-reweight'),
    ('original', 'Original bonus', '#64748b', False, None),
    ('failure_beta1', 'Failure bonus β = 1', '#be123c', False, 'failure-beta1'),
    ('failure_coverage', 'Failure sampling 40%', '#57534e', False, 'failure-coverage'),
]


def point(iteration, counts, source):
    tasks = counts['tasks']
    valid = counts.get('valid', counts.get('valid_tasks'))
    successes = counts['successes']
    assert tasks == 300 and 0 <= successes <= valid <= tasks, (source, counts)
    assert all(isinstance(x, int) for x in (iteration, tasks, valid, successes))
    for name, denominator in [('overall', tasks), ('valid_only', valid)]:
        if name in counts:
            assert abs(counts[name] - successes / denominator) < 1e-9, (source, name)
    return dict(iteration=iteration, tasks=tasks, successes=successes, valid=valid,
                source=REMOTE + source)


def build_data():
    historical = json.loads((DOCS / 'rl_results/baseline_vs_arm_full300.json').read_text())
    series = []
    for key, label, color, visible, prefix in SPECS:
        points = {
            row['iteration']: point(row['iteration'], dict(row, tasks=300),
                                    'rl_results/baseline_vs_arm_full300.json')
            for row in historical['series'].get(key, [])
        }
        if key == 'outcome_only':
            points[10] = point(10, dict(tasks=300, successes=70, valid=234), 'RL_RESULTS.md')
        if key == 'original':
            path = AUDITS / 'original-backfill-20260927.json'
            for row in json.loads(path.read_text())['rows']:
                points[row['iteration']] = point(row['iteration'], row['full300'],
                                                'arm_results/rl_integration/' + path.name)
        if prefix:
            for path in sorted(AUDITS.glob(f'{prefix}-iteration*-audit.json')):
                iteration = int(path.name.split('-iteration')[1].split('-')[0])
                data = json.loads(path.read_text())
                assert data.get('iteration', data.get('completed_iterations', iteration)) == iteration
                if key in {'expanded4102', 'outcome56'}:
                    assert data.get('evaluation_verified_complete') is True, path
                if key == 'uniform8':
                    assert (data.get('evaluation_verified_complete') is True or
                            (iteration == 10 and data.get('status') == 'verified_complete')), path
                counts = data.get('full300')
                if counts is None and key == 'uniform8':
                    m = data['evaluation']
                    counts = dict(tasks=m['tasks'], successes=m['successes'],
                                  valid=m['valid'], overall=m['overall_success_rate'],
                                  valid_only=m['valid_only_success_rate'])
                if counts is None:
                    m = data['metrics']
                    counts = dict(tasks=m['trajectories'], successes=m['successes'],
                                  valid=m['valid_trajectories'],
                                  overall=m['success_rate_all_completed'],
                                  valid_only=m['success_rate_valid'])
                points[iteration] = point(iteration, counts,
                                          'arm_results/rl_integration/' + path.name)
                if key in {'outcome56', 'uniform8'}:
                    updates = data.get('checkpoint_adam_updates',
                                       data.get('checkpoint', {}).get('completed_adam_updates'))
                    assert isinstance(updates, int) and updates >= 0, path
                    points[iteration]['adam_updates'] = updates
        assert points, key
        series.append(dict(id=key, label=label, color=color, visible=visible,
                           points=[points[i] for i in sorted(points)]))
    return dict(series=series, benchmark='Online-Mind2Web · full 300',
                protocol='Local browser · GPT-4.1 / action_history · actor temperature 0',
                limitation='Historical evaluations use different collection dates and valid-task sets. '
                           'Lines are visual guides, not additional evaluations or evidence of significance. '
                           'Training iterations are not matched optimizer-update counts. '
                           'The expanded outcome-only run uses 4,102 training tasks; other runs use the original pool. '
                           'Stealth and WebVoyager results are separate in the summary.')


def main():
    data = build_data()
    template = (REPO / 'scripts/templates/arm_rl_comparison.html').read_text()
    assert template.count('__ARM_DATA__') == 1
    html = template.replace('__ARM_DATA__', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c'))
    reward_path = DOCS / 'rl_results/arm_reward_hacking.html'
    reward = ''
    if reward_path.exists():
        reward = ('<details class="card table-card" id="reward-hacking">'
                  '<summary>Reward-hacking diagnostic: ARM reward versus task success</summary>'
                  '<iframe title="ARM reward and task success throughout training" '
                  'style="width:100%;height:1450px;border:0;margin-top:15px" loading="lazy" '
                  'srcdoc="' + html_module.escape(reward_path.read_text(), quote=True) + '"></iframe></details>')
    assert template.count('__ARM_REWARD_CHART__') == 1
    html = html.replace('__ARM_REWARD_CHART__', reward)
    target = DOCS / 'rl_results/arm_rl_interactive.html'
    target.write_text(html)
    print(f'{target}: {len(data["series"])} runs, '
          f'{sum(len(s["points"]) for s in data["series"])} evaluated checkpoints')


if __name__ == '__main__':
    main()
