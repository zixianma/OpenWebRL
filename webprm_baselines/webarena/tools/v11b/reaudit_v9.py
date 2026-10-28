"""Offline re-audit of finished v11b cells with native-audit-v9 (the v8 auditor rejected every v11 cell on the 3-method
cohort-size check). For each cell of a package attempt: copy its frozen audit-input.json with the two native-audit-v9
files added to source_sha256 (the only change), run the v9 auditor, and write campaign/reaudit-v9/<cell-key>/
{audit-input.json, audit.json, audit.record.json}. Never touches the original cell artifacts, never disposes state, no
network. Usage: reaudit_v9.py PACKAGE_DIR [ATTEMPT_ID]   (run with the browser-env python in a CPU-only job)"""
import hashlib, importlib.util, json, os, sys
from collections import Counter
from pathlib import Path
D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
AUD = D/'native-audit-v9'
pkg = Path(sys.argv[1]).resolve()
os.environ['WEBPRM_NATIVE_SOURCE_ROOT'] = str(pkg/'source'); sys.path.insert(0, str(pkg/'source'))
spec = importlib.util.spec_from_file_location('native_audit_v9', AUD/'native_cell_audit.py')
audit = importlib.util.module_from_spec(spec); sys.modules[spec.name] = audit; spec.loader.exec_module(audit)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
attempts = sorted((pkg/'campaign').glob('attempt-*')) if len(sys.argv) < 3 else [pkg/'campaign'/('attempt-'+sys.argv[2])]
summary, rows = Counter(), []
for att in attempts:
    for inp in sorted(att.glob('cells/*/audit-input.json')):
        key = inp.parent.name; out = pkg/'campaign/reaudit-v9'/att.name/key; out.mkdir(parents=True, exist_ok=True)
        if (out/'audit.json').exists():
            report = json.loads((out/'audit.json').read_text())
        else:
            value = json.loads(inp.read_text())
            value['source_sha256'] = dict(value['source_sha256'], **{str(AUD/n): sha(AUD/n) for n in ('native_cell_audit.py', 'native_lowlevel_replay.py')})
            (out/'audit-input.json').write_text(json.dumps(value, sort_keys=True, indent=1)+'\n')
            report = audit.audit_cell(out/'audit-input.json', out/'audit.json')
        ok = report.get('audit_verified') is True
        err = '' if ok else (report.get('errors') or [{}])[0].get('message', '')[:160]
        summary['verified' if ok else 'rejected: '+err] += 1
        rows.append(dict(attempt=att.name, cell=json.loads(inp.read_text())['cell']['cell_id'], verified=ok, error=err or None))
(pkg/'campaign/reaudit-v9/summary.json').write_text(json.dumps(dict(auditor=str(AUD), counts=dict(summary), cells=rows), indent=1)+'\n')
print(json.dumps(dict(summary), indent=1))
