#!/usr/bin/env python3
"""CPU-only audit of archived zero-reward groups through the new admission rule."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def audit(root):
    import torch
    torch.set_num_threads(1)
    from slime.utils.types import Sample
    from openwebrl import arm_turn_bonus as bonus
    from openwebrl.arm_failure_bonus import filter_groups
    root=Path(root).resolve()
    config=json.loads((root/'arm-config.json').read_text())
    current=dict(config=dict(config,admit_all_failure_groups=True),excluded_ids=set())
    args=SimpleNamespace(reward_key=None)
    counts=Counter(); retained=[]; scanned=0
    for journal in sorted((root/'iterations').glob('*/groups/*/*.json')):
        header=json.loads(journal.read_text())
        if header.get('filter_reason')!='zero_std_0.0':continue
        scanned+=1
        archive=torch.load(journal.with_suffix('.pt'),map_location='cpu',weights_only=False,mmap=True)
        current['config']['policy_id']=archive['policy_id']
        grouped=defaultdict(list)
        for row in archive['samples']:
            sample=Sample.from_dict(dict(row))
            grouped[(sample.metadata or {}).get('trajectory_id',sample.index)].append(sample)
        with patch.object(bonus,'_STATE',current):
            result=filter_groups(args,list(grouped.values()))
        counts[result.reason]+=1
        if result.keep:
            rows=[s for t in result.samples for s in t]
            units=[bonus.unit_bonus(s,archive['policy_id']) for s in rows]
            retained.append(dict(group_id=header['group_id'],task_id=rows[0].metadata['task_id'],
                policy_id=archive['policy_id'],rows=len(rows),labels=sum(u!=0 for u in units),
                positive_labels=sum(u>0 for u in units),negative_labels=sum(u<0 for u in units)))
    return dict(root=str(root),scanned_zero_groups=scanned,decisions=dict(counts),admitted_groups=retained,
        eligible_groups=len(retained),eligible_turns=sum(r['rows'] for r in retained),
        eligible_labels=sum(r['labels'] for r in retained),gpu_used=False,optimizer_updates=0,
        limitation='Read-only prior-policy audit. New training must collect fresh groups; this is not a training replay or held-out evaluation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=audit(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='admitted_groups'}))
