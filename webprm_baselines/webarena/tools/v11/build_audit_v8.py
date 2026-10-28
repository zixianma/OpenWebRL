"""Build native-audit-v8 = native-audit-v7 (four sites + Piotr replay) for v11 cohorts.

native-audit-v7 stays untouched (v10 packages freeze it). v8 changes, each an exact anchored replacement:
  both files   the native screen/viewport is 1280x720 (user-chosen v11 setup; v8-v10: 1024x2490)
  native_cell_audit.py   v11 cohorts hold (no_search, piotr_selectionarm); the expected bridge config carries
                         v11=<webprm_native_piotr_v11.RECEIPT> for piotr_selectionarm (key v10 in v7)
  native_lowlevel_replay.py  the Piotr selector is webprm_native_piotr_v11 (the actor's own 1280x720 screenshot, unscaled)
Usage: build_audit_v8.py   (writes D/native-audit-v8; refuses to overwrite)
"""
import shutil
from pathlib import Path

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
SRC, DST = D/'native-audit-v7', D/'native-audit-v8'
assert not DST.exists(), DST
DST.mkdir()
for name in ('native_cell_audit.py', 'native_lowlevel_replay.py'):
    shutil.copy2(SRC/name, DST/name)


def patch(path, pairs):
    text = path.read_text()
    for a, b in pairs:
        assert text.count(a) == 1, (path.name, a[:100])
        text = text.replace(a, b)
    path.write_text(text)


patch(DST/'native_cell_audit.py', [
    ("SCHEMA='native_table5_v10_independent_cell_audit_v7'\n",
     "SCHEMA='native_table5_v11_independent_cell_audit_v8'\n"
     "# v8 (2026-10-27): v11 cohorts (user chose the 1280x720 setup): no_search (rerun) + piotr_selectionarm per task,\n"
     "# 1280x720 actor viewport; the ARM sees the actor's own screenshot.\n"),
    ("selectors={'piotr_selectionarm'}", "selectors={'no_search','piotr_selectionarm'}"),
    (" from openwebrl.webprm_native_piotr_v10 import RECEIPT as V10_RECEIPT\n", " from openwebrl.webprm_native_piotr_v11 import RECEIPT as V11_RECEIPT\n"),
    ("v10=json.loads(json.dumps(V10_RECEIPT)) if method=='piotr_selectionarm' else None)",
     "v11=json.loads(json.dumps(V11_RECEIPT)) if method=='piotr_selectionarm' else None)"),
    ("V8_RECEIPT=dict(viewport={'width':1024,'height':2490},", "V8_RECEIPT=dict(viewport={'width':1280,'height':720},  # v8 audit: v11 1280x720\n "),
    ("replay=load('_native_table5_v10_independent_lowlevel_replay_v7',helper)", "replay=load('_native_table5_v11_independent_lowlevel_replay_v8',helper)"),
])

patch(DST/'native_lowlevel_replay.py', [
    ("V8_W,V8_H=1024,2490\n", "V8_W,V8_H=1280,720  # v8 audit (2026-10-27): v11 cohorts use the user-chosen 1280x720 viewport\n"),
    ("'Native screen/viewport/DPR differs from v8 1024x2490')", "'Native screen/viewport/DPR differs from v11 1280x720')"),
    (" from openwebrl.webprm_native_piotr_v10 import NativePiotrSelector\n", " from openwebrl.webprm_native_piotr_v11 import NativePiotrSelector\n"),
])
print('wrote', DST)
