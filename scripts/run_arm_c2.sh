#!/usr/bin/env bash
# Resume the prepared C2 run inside an already assigned single-GPU allocation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/h200_env.sh
C2_QUEUE="${1:?Pass the prepared C2 queue JSON}"
C2_PYTHON="$OPENWEBRL_RUNTIME_ROOT/arm-reproduction/venv/bin/python"
C2_ACTOR_PID=""
cleanup() {
    if [[ -n "$C2_ACTOR_PID" ]]; then
        kill "$C2_ACTOR_PID" 2>/dev/null || true
        wait "$C2_ACTOR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
C2_OUTPUT="$($C2_PYTHON - "$C2_QUEUE" <<'PY'
import json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime
q=json.loads(Path(sys.argv[1]).read_text())
assert q['authorized'] is True and q['ready'] is True
assert os.getenv('SLURM_JOB_ID') == q['allocation']
assert '/job_'+q['allocation']+'/' in Path('/proc/self/cgroup').read_text()
assert time.time() < datetime.fromisoformat(q['stop_utc']).timestamp()-300
info=subprocess.check_output(['scontrol','show','job',q['allocation'],'-o'],text=True)
assert 'JobState=RUNNING' in info
ids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
assert ids == [q['gpu_uuid']]
pids=subprocess.check_output(['nvidia-smi','--id='+q['gpu_uuid'],'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not pids, 'Assigned GPU is occupied; leave existing workloads untouched'
print(q['output'])
PY
)"
CUDA_VISIBLE_DEVICES=0 "$OPENWEBRL_RUNTIME_ROOT/venv/bin/python" -m sglang.launch_server \
    --model-path /gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT \
    --host 127.0.0.1 --port 19100 --dtype bfloat16 --tp 1 \
    --mem-fraction-static 0.4 --context-length 32768 --max-running-requests 48 \
    --chunked-prefill-size 4096 --cuda-graph-max-bs 48 \
    >>"$C2_OUTPUT/actor-resume.log" 2>&1 &
C2_ACTOR_PID=$!
C2_DEADLINE=$((SECONDS+600))
until curl --noproxy '*' -fs http://127.0.0.1:19100/health_generate >/dev/null; do
    kill -0 "$C2_ACTOR_PID"
    if (( SECONDS > C2_DEADLINE )); then exit 1; fi
    sleep 2
done
"$C2_PYTHON" - "$C2_QUEUE" "$C2_ACTOR_PID" <<'PY'
import json,os,subprocess,sys
from pathlib import Path
from scripts.run_arm_retry import atomic_json,process_identity
path=Path(sys.argv[1]);q=json.loads(path.read_text())
q['slurm_step']=os.environ['SLURM_STEP_ID']
pids=subprocess.check_output(['nvidia-smi','--id='+q['gpu_uuid'],'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
assert sys.argv[2] in pids and len(pids)==2, 'Unexpected actor GPU process set'
q['resident_actor_processes']=[process_identity(int(pid)) for pid in pids]
for identity in q['resident_actor_processes']:
    assert '/job_'+q['allocation']+'/' in identity['cgroup']
atomic_json(path,q)
atomic_json(Path(q['output'])/'queue-manifest.json',q)
PY
"$C2_PYTHON" scripts/run_arm_c2.py --queue "$C2_QUEUE" >>"$C2_OUTPUT/controller.log" 2>&1
