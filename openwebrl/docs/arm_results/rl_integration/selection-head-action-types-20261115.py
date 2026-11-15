"""Action types picked by lookahead- vs pre-execution-label heads (November 14-15 window, eval-pre): for every decision step of every
episode (3 runs x 1,045 tasks), the tool-call name of the selected candidate, per head (4 sizes x 2 label types), plus SFT 4B alone
(its single sample) and all five candidates offered to the heads (the actor's proposal mix). Unparseable actions count as "other".
Writes the adjacent JSON."""
import collections, glob, json, os, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/eval-pre"
SLUG = {"sft": "sft1", "probe-lookahead": "head5-ef597d45f3b1", "probe-pre": "head5-c8dc992eb5ad", "d128-lookahead": "head5-5fbddd8b2031",
        "d128-pre": "head5-cf9c67a0d0f2", "d512-lookahead": "head5-91e7a5a67417", "d512-pre": "head5-cecccef33bba",
        "d4096-lookahead": "head5-7bbad4567d44", "d4096-pre": "head5-cd581ffe7dd0"}
NAME = re.compile(r'"name"\s*:\s*"([A-Za-z_]+)"')


def kind(c):
    m = NAME.search(str(c.get("action") if isinstance(c, dict) else c))
    return m.group(1) if m else "other"


def episode(rec):
    r = json.load(open(rec)); picked, offered = collections.Counter(), collections.Counter()
    for tf in r.get("trace_files") or []:
        q = Path(tf["path"] if isinstance(tf, dict) else tf)
        if not q.exists(): continue
        for line in open(q):
            x = json.loads(line); cands = x.get("candidates") or []
            i = x.get("selected_index") or 0
            if cands: picked[kind(cands[i])] += 1; offered.update(kind(c) for c in cands)
    return picked, offered


res = {}
with ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
    for k, slug in SLUG.items():
        recs = glob.glob(f"{ROOT}/run*/*/{slug}/*/collection/records/*.json")
        p, o = collections.Counter(), collections.Counter()
        for a, b in pool.map(episode, recs):
            p.update(a); o.update(b)
        res[k] = dict(episodes=len(recs), picked=dict(p), offered=dict(o))
        print(k, len(recs), sum(p.values()), p.most_common(8), flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
