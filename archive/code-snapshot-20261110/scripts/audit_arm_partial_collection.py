#!/usr/bin/env python3
"""Read-only calibration estimates from completed native group journals.

Uses the preserved native normalization and ARM panel. Does not load images,
change rewards/configuration, admit a partial batch, or start workers.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace


def audit(root):
    manifest=json.loads((root/'launch_manifest.json').read_text())
    source=Path(manifest['source'])
    sys.path.insert(0,str(source))
    import torch
    from slime.utils.types import Sample
    from openwebrl.arm_turn_bonus import panel
    from openwebrl.arm_turn_bonus_runtime import calibration_decision
    torch.set_num_threads(1)
    config=json.loads((root/'arm-config.json').read_text())
    iteration=Path(config['output'])
    live=json.loads((iteration/'live_metrics.json').read_text())
    if live.get('arm_collection/excluded_trajectories',0):
        raise ValueError('Compact journals omit exclusion metadata; use the full native calibration instead')
    groups=[json.loads(p.read_text()) for p in sorted((iteration/'groups'/str(config['rollout_id'])).glob('*.json'))]
    accepted=[g for g in groups if g['accepted']]
    samples=[]
    for group in accepted:
        if group['policy_id']!=config['policy_id']:
            raise ValueError('Mixed policy versions in group journals')
        kept=set(group['accepted_sample_indices'])
        for row in group['rows']:
            if row['index'] not in kept: continue
            samples.append(Sample(index=row['index'],group_index=group['group_id'],
                response_length=row['response_length'],reward=row['reward'],
                remove_sample=row['remove_sample'],loss_mask=[int(row['trainable'])],
                metadata=dict(task_id=row['task_id'],trajectory_id=row['trajectory_id'],
                    turn_index=0,arm_turn_bonus=row['arm_turn_bonus'])))
    if not samples: raise ValueError('No accepted groups yet')
    def group_by(rows,key):
        result=defaultdict(list)
        for row in rows: result[key(row)].append(row)
        return result
    tree=ast.parse((source/'slime/ray/rollout.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_post_process_rewards')
    ns=dict(torch=torch,group_by=group_by,logger=logging.getLogger(__name__),
        logging_utils=SimpleNamespace(append_progress_log=lambda *args:None),
        os=SimpleNamespace(environ={}))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source/'slime/ray/rollout.py'),'exec'),ns)
    args=SimpleNamespace(advantage_estimator='grpo',rewards_normalization=True,
        grpo_std_normalization=True,reward_key=None)
    _,normalized=ns['_post_process_rewards'](SimpleNamespace(args=args,custom_reward_post_process_func=None),samples)
    result=panel(samples,normalized,config['policy_id'])
    result.update(journal_groups=len(groups),accepted_groups=len(accepted),
        checked_utc=datetime.now(timezone.utc).isoformat(),
        policy_id=config['policy_id'],source=str(source),root=str(root),
        decision=calibration_decision(result),partial_only=True,
        note='Incomplete collection estimate only; the native full-batch gate still controls training. No configuration or training artifacts changed.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); result=audit(args.run_root)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
