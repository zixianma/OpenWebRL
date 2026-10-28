"""Print V9_BLOCKS (selector:ids;...) for cells of a crashed package that have no valid recorded outcome in any of
the given attempt dirs, plus the USD the attempts spent. Usage: replacement_blocks.py PACKAGE [PACKAGE...]"""
import json, sys, glob
from pathlib import Path
done, spent, planned = set(), 0.0, []
for pkg in map(Path, sys.argv[1:]):
    prop = json.loads((pkg/'proposal.json').read_text())
    if not planned:
        planned = [(b['selector'], t) for b in prop['v8_cell_blocks'] for t in b['task_ids']]
    for res in glob.glob(str(pkg/'campaign/attempt-*/cells/*/episodes/*/*/result.json')):
        d = json.load(open(res))
        if d.get('valid') or 'fuzzy' in str(d.get('error', '')).lower() or str(d.get('error', '')).startswith(('Invalid action', 'List requirements must contain')):  # agent-chosen unparseable action: final (counted as failure in sensitivity), never resampled
            done.add((d['selector'], int(d['task_id'].rsplit('_', 1)[1])))
        if False:  # fuzzy-judge invalids: trajectory kept for gpt-4.1 rescore
            done.add((d['selector'], int(d['task_id'].rsplit('_', 1)[1])))
    for b in glob.glob(str(pkg/'campaign/attempt-*/broker-closure.json')) + glob.glob(str(pkg/'campaign/attempt-*/financial-contract.json')):
        pass
    for bud in glob.glob(str(pkg/'campaign/shared-budget.json')):
        try: spent += float(json.load(open(bud)).get('charged', 0))
        except Exception: pass
todo = [(s, t) for s, t in planned if (s, t) not in done]
blocks = ';'.join(f"{s}:{','.join(str(t) for ss, t in todo if ss == s)}" for s in ('no_search', 'arbiter7b', 'shepherd8b') if any(ss == s for ss, _ in todo))
print(json.dumps(dict(valid_done=len(done), remaining=len(todo), blocks=blocks, spent_usd=round(spent, 4))))
