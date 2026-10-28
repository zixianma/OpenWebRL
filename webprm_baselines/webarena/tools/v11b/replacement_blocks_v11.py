"""Cells of a v11b package (and its earlier replacements) that still need a run, as interleaved replacement chunks.
A planned (selector, task) cell is final when any attempt recorded: a valid outcome; a fuzzy-judge invalid (retired
gpt-4-1106-preview); an agent-chosen 'Invalid action ...'; or 'List requirements must contain nonempty strings'
(evaluator config). Those are never rerun. Everything else (no result yet, or any other invalid: infrastructure) is pending.
Chunks keep the cohort's task-major order (no_search then piotr_selectionarm per task), break only between tasks, and
hold at most MAX cells (default 20: under the 1 TiB scratch cap even if no state is disposed).
Usage: replacement_blocks_v11.py [--max N] PACKAGE [PACKAGE...]   (first package defines the planned cells)
Prints JSON: final, pending, infra_invalid, chunks=[{cells, tasks, blocks}], spent_usd (charged, all given packages)."""
import glob, json, sys
from pathlib import Path
SELECTORS = ('no_search', 'piotr_selectionarm')
FINAL_PREFIXES = ('Invalid action', 'List requirements must contain')
args = sys.argv[1:]; cap = 20
if args and args[0] == '--max':
    cap = int(args[1]); args = args[2:]
final, invalid, spent, planned = {}, {}, 0.0, []
for pkg in map(Path, args):
    prop = json.loads((pkg/'proposal.json').read_text())
    if not planned:
        cohort = json.loads((pkg/'cohort.json').read_text())
        wanted = {(b['selector'], t) for b in prop['v8_cell_blocks'] for t in b['task_ids']}
        planned = [(c['selector'], c['task_id']) for c in cohort['cells'] if (c['selector'], c['task_id']) in wanted]  # cohort (task-major) order
    for res in glob.glob(str(pkg/'campaign/attempt-*/cells/*/episodes/*/*/result.json')):
        d = json.load(open(res)); key = (d['selector'], int(d['task_id'].rsplit('_', 1)[1])); err = str(d.get('error', ''))
        if d.get('valid'):
            final[key] = 'valid'
        elif 'fuzzy' in err.lower():
            final.setdefault(key, 'fuzzy_judge_invalid')
        elif err.startswith(FINAL_PREFIXES):
            final.setdefault(key, 'agent_or_config_invalid')
        else:
            invalid.setdefault(key, (d.get('error_type'), err[:120]))
    for bud in glob.glob(str(pkg/'campaign/shared-budget.json')):
        spent += float(json.load(open(bud)).get('charged', 0))
todo = [c for c in planned if c not in final]
chunks, cur = [], []
for task in dict.fromkeys(t for _, t in todo):
    group = [c for c in todo if c[1] == task]
    if cur and len(cur) + len(group) > cap:
        chunks.append(cur); cur = []
    cur += group
if cur:
    chunks.append(cur)
def blocks(cells):
    return ';'.join(f"{s}:{','.join(str(t) for ss, t in cells if ss == s)}" for s in SELECTORS if any(ss == s for ss, _ in cells))
print(json.dumps(dict(planned=len(planned), final=len(final), final_kinds={k: sum(v == k for v in final.values()) for k in set(final.values())},
                      pending=len(todo), infra_invalid=[dict(selector=s, task=t, error=invalid[(s, t)]) for s, t in todo if (s, t) in invalid],
                      chunks=[dict(cells=len(c), tasks=sorted({t for _, t in c}), blocks=blocks(c)) for c in chunks],
                      spent_usd=round(spent, 4)), indent=1))
