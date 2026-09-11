#!/usr/bin/env python3
"""Compare completed joint-data endpoints, including the historical base control."""
import json
from datetime import datetime, timezone
from pathlib import Path

from summarize_arm_c2_full300 import load_records, mcnemar_exact, paired_report, rate_report
try:
    from project_docs import write_document_section
except ModuleNotFoundError:  # Imported as scripts.summarize_arm_joint_comparison.
    from scripts.project_docs import write_document_section

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / 'openwebrl/docs'
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction')


def main():
    ids = [json.loads(x)['task_id'] for x in
           (REPO / 'openwebrl/data/eval/online-mind2web.jsonl').read_text().splitlines()]
    roots = {name: RUNTIME / f'runs/joint-v2-{name}-2gpu-r2' for name in ['sft', 'dpo']}
    records = {}
    for name, root in roots.items():
        complete = json.loads((root / 'complete.json').read_text())
        if complete['updates'] != 174 or complete['evaluated_tasks'] != 300:
            raise ValueError(f'Incomplete endpoint: {name}')
        records[name] = json.loads((root / 'evaluation/all300-records.json').read_text())
    baseline = Path(json.loads((DOCS / 'arm_results/base_vs_1a_comparison.json').read_text())['base_run'])
    records['historical-base'] = load_records(baseline, ids)
    if len(ids) != 300 or any(set(r) != set(ids) for r in records.values()):
        raise ValueError('All runs must cover the same 300 unique task IDs')
    rates = {name: rate_report(rows, ids) for name, rows in records.items()}
    pairs = {}
    for left, right in [('sft', 'dpo'), ('historical-base', 'sft'), ('historical-base', 'dpo')]:
        a, b = records[left], records[right]
        success = lambda row: bool(row.get('valid')) and row.get('reward') == 1
        wins = sum(success(b[t]) and not success(a[t]) for t in ids)
        losses = sum(success(a[t]) and not success(b[t]) for t in ids)
        common = paired_report(a, b, ids)
        pairs[f'{right}_vs_{left}'] = dict(left=left, right=right,
            all_scheduled=dict(tasks=300, right_wins=wins, right_losses=losses,
                net_wins=wins-losses, difference_pp=(wins-losses)/3,
                mcnemar_exact_two_sided_p=mcnemar_exact(wins, losses)),
            common_valid=common)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), rates=rates,
                  paired_comparisons=pairs, roots={k: str(v) for k, v in roots.items()},
                  historical_base=str(baseline),
                  notes=['One training seed per objective; same matched training data and original actor.',
                         'Overall includes unavailable outcomes as failures; common-valid conditions on availability.',
                         'Base evaluation is historical; live-site drift limits causal attribution.',
                         'Paired p-values are exploratory, without multiple-comparison adjustment.'])
    output = DOCS / 'arm_results/joint_data_v2/joint-sft-vs-dpo-om2w.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Joint C2 + Piotr SFT versus DPO', '',
             'Both endpoints: update 174, 5,540 matched training states, independent',
             'initialization from the original SFT actor. Fresh full-300 OM2W evaluation',
             'uses one candidate, no inference ARM, and the o4-mini/AgentTrek judge.', '',
             '| Policy | Overall | Valid-only | Unavailable |',
             '| --- | ---: | ---: | ---: |']
    for name in ['historical-base', 'sft', 'dpo']:
        r = rates[name]
        lines.append(f"| {name} | {r['successes']}/300 = {r['overall']:.1%} | "
                     f"{r['successes']}/{r['valid']} = {r['valid_only']:.1%} | {r['unavailable']} |")
    lines += ['', 'All tasks have result files; unavailable outcomes have not been replaced.', '',
              '## Paired evidence', '',
              'Wins/losses below favor the first named policy. The all-scheduled analysis',
              'treats unavailable outcomes as failures; common-valid is supporting evidence.', '',
              '| Comparison | All-300 wins / losses | Exact p | Common-valid tasks | Wins / losses | Exact p |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, value in pairs.items():
        a, c = value['all_scheduled'], value['common_valid']
        lines.append(f"| {name} | {a['right_wins']} / {a['right_losses']} | "
                     f"{a['mcnemar_exact_two_sided_p']:.4f} | {c['common_valid']} | "
                     f"{c['right_wins']} / {c['right_losses']} | {c['mcnemar_exact_two_sided_p']:.4f} |")
    lines += ['', 'One training seed per objective. Historical-base comparisons are subject',
              'to live-site drift. P-values are exploratory and unadjusted for multiple',
              'comparisons; valid-only marginal rates use different task populations.', '',
              '- [SFT results and rollouts](ARM_JOINT_SFT_RESULTS.md)',
              '- [DPO results and rollouts](ARM_JOINT_DPO_RESULTS.md)',
              '- [Training diagnostics](ARM_JOINT_TRAINING_MONITOR.md)',
              '- [Machine-readable report](arm_results/joint_data_v2/joint-sft-vs-dpo-om2w.json)', '']
    write_document_section(DOCS / 'ARM_JOINT_SFT_VS_DPO_RESULTS.md', '\n'.join(lines))
    print(json.dumps(dict(rates=rates, pairs=pairs), indent=2))


if __name__ == '__main__':
    main()
