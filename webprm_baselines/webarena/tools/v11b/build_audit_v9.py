"""Build native-audit-v9 = native-audit-v8 with the cohort-membership size taken from the cohort's own method list.

v11b job 392816 (reddit-a) stopped after 21/36 cells: every independent audit rejected its cell with 'Fixed v9 cell
membership differs' because audit_identity still required len(cohort cells) == 3 x site tasks (v9: three methods);
v11 cohorts have two (no_search, piotr_selectionarm). A rejected audit keeps the cell's ~50 GiB site state on node
scratch (no disposal), so cumulative scratch reached the 1 TiB cap (admission reason cumulative_scratch_capacity).
native-audit-v8 stays untouched (v11/v11b packages freeze it). v9 change (one anchored replacement): the membership
size is len(cohort['selectors']) x site tasks; the exact method set {no_search, piotr_selectionarm} and the
one-cell-per-(task, method) grid checks that follow are unchanged.
Usage: build_audit_v9.py   (writes D/native-audit-v9; refuses to overwrite)
"""
import shutil
from pathlib import Path

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
SRC, DST = D/'native-audit-v8', D/'native-audit-v9'
assert not DST.exists(), DST
DST.mkdir()
for name in ('native_cell_audit.py', 'native_lowlevel_replay.py'):
    shutil.copy2(SRC/name, DST/name)
p = DST/'native_cell_audit.py'
t = p.read_text()
for a, b in [
    ("SCHEMA='native_table5_v11_independent_cell_audit_v8'\n",
     "SCHEMA='native_table5_v11_independent_cell_audit_v9'\n"
     "# v9 (2026-10-27): cohort membership size = len(cohort selectors) x site tasks (v8 required 3x; job 392816 rejected all 21 audits).\n"),
    ("require(len(cells)==len(indexed)==3*V9_SITE_TASKS[site] and indexed.get(cell.get('cell_id'))==cell,'Fixed v9 cell membership differs')",
     "require(len(cells)==len(indexed)==len(cohort['selectors'])*V9_SITE_TASKS[site] and indexed.get(cell.get('cell_id'))==cell,'Fixed v9 cell membership differs')"),
    ("replay=load('_native_table5_v11_independent_lowlevel_replay_v8',helper)", "replay=load('_native_table5_v11_independent_lowlevel_replay_v9',helper)"),
]:
    assert t.count(a) == 1, a[:80]
    t = t.replace(a, b)
p.write_text(t)
print('wrote', DST)
