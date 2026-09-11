#!/usr/bin/env python3
"""Refresh the joint experiment's local monitoring document from durable logs."""
import json
from datetime import datetime, timezone
from pathlib import Path
try:
    from project_docs import write_document_section
except ModuleNotFoundError:  # Imported as scripts.report_arm_joint_training.
    from scripts.project_docs import write_document_section

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction')


def main():
    pointer = RUNTIME / 'current_joint_runs.json'
    data = json.loads(pointer.read_text())
    data['checked_utc'] = datetime.now(timezone.utc).isoformat()
    lines = ['# Joint SFT and DPO training monitoring', '',
             f"Last checked: {data['checked_utc']}.", '',
             'Both runs independently start from the original SFT actor; 174 updates',
             'on 5,540 states. Automatic fresh full-300 OM2W evaluation follows each',
             'endpoint. Training metrics below are not browser task success rates.', '',
             '## Current status', '']
    validations = []
    for run in data['runs']:
        root = Path(run['run_root'])
        rows = []
        for line in (root / 'student/metrics.jsonl').read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A concurrent final JSONL write can be incomplete.
        run['training_status'] = json.loads((root / 'training-status.json').read_text())
        pipeline = json.loads((root / 'status.json').read_text())
        run['status'] = (run['training_status']['phase']
                         if pipeline['phase'] == 'training_startup' else pipeline['phase'])
        run['latest_metrics'] = [r for r in rows if 'train/loss' in r][-1]
        run['latest_checkpoint'] = json.loads((root / 'student/latest-checkpoint.json').read_text())
        count = sum(len(list((root / f'evaluation/shard-{i}/online-mind2web/results').glob('*.json')))
                    for i in range(2))
        run['evaluation_results_written'] = count
        name = run['objective'].upper()
        lines.append(f"- **{name}**: job {run['job_id']} on {run['node']}; "
                     f"{run['status']}; {run['latest_metrics']['updates']}/174 updates; "
                     f"{count}/300 evaluation result files. "
                     f"[W&B]({run['wandb_url']}).")
        for row in rows:
            if 'val/C2/winner_ce' not in row:
                continue
            for source in ['C2', 'Piotr']:
                validations.append(f"| {name} | {row['updates']} | {source} | "
                    f"{row[f'val/{source}/winner_ce']:.5f} | "
                    f"{row[f'val/{source}/dpo_loss']:.5f} | "
                    f"{row[f'val/{source}/full_ranking']:.2%} | "
                    f"{row[f'val/{source}/action_ranking']:.2%} |")
    lines += ['', '## Fixed held-out diagnostics', '',
              'Each panel contains 256 C2 and 216 Piotr pairs. Ranking compares',
              'sequence-summed chosen/rejected log probabilities; the action column',
              'restricts scored tokens to the action. Neither is task success.', '',
              '| Run | Update | Source | Winner CE | Preference loss | Full ranking | Action ranking |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: |'] + validations
    lines += ['', 'SFT reduces held-out winner CE, with most improvement by update 87,',
              'but does not improve preference loss. DPO improves preference loss',
              'while increasing winner CE. Assess policy quality with the predeclared',
              'endpoint browser evaluations, not either offline loss alone.', '',
              '[Run plan and recovery details](ARM_JOINT_DATA_TRAINING_PLAN.md).', '',
              'Refresh this report with `python3 scripts/report_arm_joint_training.py`.', '']
    write_document_section(REPO / 'openwebrl/docs/ARM_JOINT_TRAINING_MONITOR.md', '\n'.join(lines))
    temporary = pointer.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(pointer)
    print('\n'.join(lines[10:12]))


if __name__ == '__main__':
    main()
