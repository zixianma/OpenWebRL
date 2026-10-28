#!/usr/bin/env python3
"""Build one frozen OM2W Table 5-style stage package; never submits or calls a provider.

The package copies the frozen Table 5 CMS source snapshot (identical worker, broker,
critic and lifecycle code) and overlays the OM2W harness, its task file, strata and
judge prompt source. Critic identities and actor routes come from the frozen Table 5
proposal. The API cap is the user-approved value; the approval names it again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl import webprm_om2w_table5 as om
from openwebrl.webprm_webarena import digest

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012')
REFERENCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/cms-table5-350-v1')
OVERLAY = ('openwebrl/webprm_om2w_table5.py', 'scripts/run_webprm_om2w_table5_cell.py', 'scripts/launch_webprm_om2w_table5.py',
    'openwebrl/webprm_worker.py', 'openwebrl/webprm_runtime.py', 'openwebrl/webprm_concurrent_broker.py',
    'scripts/collect_webprm_om2w_table5.py', 'scripts/estimate_webprm_om2w_table5_cost.py',
    'scripts/prepare_webprm_om2w_table5.py', 'scripts/validate_webprm_om2w_table5_fixture.py', 'tests/test_webprm_om2w_table5.py',
    'scripts/audit_webprm_om2w_saved_calls.py', 'scripts/audit_webprm_om2w_native_evidence.py',
    'tests/test_collect_webprm_om2w_table5.py', 'tests/test_audit_webprm_om2w_native_evidence.py',
    'tests/test_audit_webprm_om2w_saved_calls.py',
    'tests/test_webprm_concurrent_backlog.py',
    'openwebrl/eval/reward_online_mind2web.py', om.TASK_FILE, om.COHORTS_FILE)
BROWSER = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python'
# Stage envelopes. Requests: per task (10 cells) at the 30-turn cap, 8 cells x (3 x 30 + judge) plus two
# GPT-4o-mini selector cells x (7 x 30 + judge) paid; Shepherd 6, Arbiter 4, Piotr 1 local calls per turn.
# pilot10 matches the user's receipt exactly: 1 H200, 8 CPUs, <=240 GiB, <=8 h cumulative incl. retries.
STAGE_RESOURCES = {'pilot10': 8*3600, 'next90': 24*3600, 'rest200': 48*3600}
# Fixture 358449: with 6 concurrent live browsers on 8 CPUs a heavy page (flightaware.com) hung BrowserGym's
# post-action observation past the 180 s watchdog; the v6 OM2W pilot ran 4. The pilot uses 4.
STAGE_WORKERS = {'pilot10': 4, 'next90': 6, 'rest200': 6}
PAID_PER_TASK, LOCAL_PER_TASK, LOCAL_PREFLIGHT = 8*91+2*211, 2*30*(6+4+1), 11


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path); path.write_text(json.dumps(value, indent=1, sort_keys=True)+'\n'); os.chmod(path, 0o600)


def build(stage, api_usd, package, *, fixture_overrides=None, campaign_root=None, receipt=None, preproduction_ledger=None,
          actors=None, cell_ids=None, predecessor=None, candidate_controls=False, candidate_control_tasks=8,
          focused_controls=False):
    if focused_controls and (not candidate_controls or candidate_control_tasks != 100 or api_usd != 500):
        raise ValueError('Focused study requires explicit100-task candidate controls and USD500 cap')
    if candidate_control_tasks != 8 and not candidate_controls:
        raise ValueError('Control task count requires the explicit candidate-control study')
    if candidate_controls and (stage != 'rest200' or actors != ['gpt-4o-2024-05-13']
            or cell_ids is not None or predecessor is not None or preproduction_ledger is not None):
        raise ValueError('Candidate controls require a fresh GPT-4o rest200 development package')
    package = Path(package)
    if package.exists():
        raise SystemExit(f'{package} exists; refusing to overwrite')
    reference = json.loads((REFERENCE/'proposal.json').read_text())
    table5_template = json.loads((REFERENCE/'worker-template.json').read_text())
    package.mkdir(mode=0o700, parents=True)
    source = package/'source'
    shutil.copytree(REFERENCE/'source', source)
    for rel in OVERLAY:
        (source/rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(REPO/rel, source/rel)
    from scripts.launch_webprm_om2w_table5 import REQUIRED_CODE
    code = {str(source/rel): sha(source/rel) for rel in REQUIRED_CODE}
    template = om.worker_template(table5_template, table5_template['arbiter_source'], code, reference['env_file'])
    if actors is not None:
        # Single-actor stages keep the Table 5 actor order and every selector route
        # (the GPT-4o-mini selector still uses its own model) but drop the other actor's cells.
        if not actors or any(a not in template['actors'] for a in actors):
            raise ValueError('Actors must be a nonempty subset of the Table 5 actors')
        template['actors'] = [a for a in template['actors'] if a in actors]
    if candidate_controls:
        template['selectors'] = om.candidate_control_spec(candidate_control_tasks, focused=focused_controls)['selectors']
    put(package/'worker-template.json', template)
    task_file, cohorts_file = str(source/om.TASK_FILE), str(source/om.COHORTS_FILE)
    cohort = om.cohort(task_file, cohorts_file, template['actors'], template['selectors'])
    put(package/'cohort.json', cohort)
    broker_routes, cell_routes = om.routes(reference['broker_routes'], reference['cell_routes'], template['actors'],
                                          candidate_controls=candidate_controls)
    lo, hi = om.STAGES[stage]; tasks = hi-lo; cells = len(template['actors'])*len(template['selectors'])*tasks
    if candidate_controls:
        tasks = candidate_control_tasks
        envelope = om.candidate_control_envelope(tasks, focused=focused_controls)
        cells, cell_ids = tasks*len(template['selectors']), om.candidate_control_ids(cohort, tasks, focused=focused_controls)
    max_attempts = envelope['limits']['maximum_attempts'] if candidate_controls else 3
    limits = dict(api_usd=float(api_usd), paid_requests=PAID_PER_TASK*tasks,
        local_requests=LOCAL_PER_TASK*tasks+LOCAL_PREFLIGHT*max_attempts, max_workers=STAGE_WORKERS[stage],
        per_cell_memory_bytes=8*2**30, fixed_memory_bytes=160*2**30, min_start_seconds=1800, cleanup_seconds=300,
        retained_bytes=64*2**30, max_consecutive_invalid=25, maximum_attempts=max_attempts, min_cell_budget_usd=6.0)
    limits['total_requests'] = limits['paid_requests']+limits['local_requests']
    resources = dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=240, total_seconds=STAGE_RESOURCES[stage])
    if candidate_controls:
        resources = envelope['resources']
        limits.update(envelope['limits'])
    batch = package/'campaign.sbatch'
    batch.write_text(f'''#!/usr/bin/env bash
# OM2W stage {stage}: {cells} fixed cells ({tasks} tasks x {len(template['actors'])} actor(s) x {len(template['selectors'])} selectors).
# All attempts share the cumulative clock, API cap and dispatch caps. Usage: sbatch --time=MIN campaign.sbatch APPROVAL.json
#SBATCH --job-name=om2w-t5-{stage}
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@240
#SBATCH --export=ALL
#SBATCH --output={package}/campaign-%j.out
set -euo pipefail
: "${{1:?Provide the exact approved request path}}"
cd {source}
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 WANDB_PROJECT=openwebrl-evals WANDB_MODE=disabled
export BASH_ENV=/dev/null ENV=/dev/null HISTFILE=/dev/null PYTHONPATH={source}
exec {BROWSER} -B scripts/launch_webprm_om2w_table5.py --execute --proposal {package}/proposal.json --approval "$1"
''')
    os.chmod(batch, 0o700)
    bound = [p for p in sorted(source.rglob('*')) if p.is_file()]
    source_sha = {str(p): sha(p) for p in bound}
    for p in (package/'worker-template.json', package/'cohort.json', batch, REFERENCE/'proposal.json'):
        source_sha[str(p)] = sha(p)
    if preproduction_ledger is not None:
        preproduction_ledger = str(Path(preproduction_ledger).resolve())
        source_sha[preproduction_ledger] = sha(preproduction_ledger)
    estimate = None
    if not candidate_controls:
        try:
            from scripts.estimate_webprm_om2w_table5_cost import estimate as cost
            estimate = cost()
        except (OSError, ValueError, KeyError):
            estimate = None
    proposal = dict(schema_version=1, protocol=om.CAMPAIGN_PROTOCOL, stage=stage, stage_tasks=tasks, stage_cells=cells,
        campaign_root=str(campaign_root or package/'campaign'), task_file=task_file, cohorts_file=cohorts_file,
        cohort_path=str(package/'cohort.json'), cohort_sha256=digest(cohort),
        worker_template=str(package/'worker-template.json'), worker_template_sha256=sha(package/'worker-template.json'),
        reference_proposal=str(REFERENCE/'proposal.json'), protocol_sha256=digest(template['protocol_contract']),
        broker_routes=broker_routes, cell_routes=cell_routes, env_file=reference['env_file'], models=reference['models'],
        resources=resources, limits=limits, source_sha256=source_sha,
        **(dict(cell_ids=list(cell_ids), **(dict(predecessor=predecessor) if predecessor else {})) if cell_ids else {}),
        **(dict(candidate_controls=om.candidate_control_spec(tasks, focused=focused_controls)) if candidate_controls else {}),
        scientific_scope=dict(benchmark='Online-Mind2Web live, pinned 300-task file', task_file_sha256=om.TASK_FILE_SHA256,
            stages={k: list(v) for k, v in om.STAGES.items()}, stage=stage,
            stage_rule='first 10 / next 90 of the frozen repeated-100 stratum, then the 200-task holdout; disjoint',
            conditions='2 actor snapshots x 5 selectors, one primary attempt per cell, never rerun',
            judge=f'{om.JUDGE} OM2W/AgentTrek terminal success, 4096 completion tokens, one call',
            paper_exact_protocol=False,
            comparability='Same viewer actor/selector protocol as the WebArena Table 5 reconstruction; live sites drift; '
                          'LLM-judge success, not programmatic evaluators'),
        cost_estimate=estimate, status='prepared_pending_api_cap_and_launch',
        accounting=dict(allocations=0, provider_calls=0, browser_calls=0))
    if candidate_controls:
        proposal['scientific_scope'].update(conditions=f"GPT-4o x {len(template['selectors'])} selectors, {tasks} fixed tasks; random5 control, no first5; fresh paired cohort",
            stage_rule=om.candidate_control_spec(tasks, focused=focused_controls)['selection'],
            comparability='Controlled live-web diagnostic; not a full300 result or WebArena-Lite paper reproduction')
    if fixture_overrides is not None:
        proposal.update(plan_overrides=fixture_overrides, fixture_only=True)
    if preproduction_ledger is not None:
        proposal['preproduction_ledger'] = preproduction_ledger
    from scripts.launch_webprm_om2w_table5 import preproduction_seconds
    prior_seconds = preproduction_seconds(proposal)
    remaining_minutes = (resources['total_seconds'] - prior_seconds)//60
    if remaining_minutes <= 0:
        raise ValueError('Preproduction exhausted the approved cumulative allocation time')
    put(package/'proposal.json', proposal)
    pending = dict(status='pending_user_api_cap_approval', approved=False, submitted=False, stage=stage,
        proposal=str(package/'proposal.json'), proposal_sha256=digest(proposal), api_usd_cap=float(api_usd),
        resources=resources, limits=limits, protocol_sha256=proposal['protocol_sha256'],
        approval_template=dict(status='approved', proposal_sha256=digest(proposal), stage=stage, api_usd_cap=float(api_usd),
            resources=resources, limits=limits, user_approval_quote='<exact user words>',
            **(dict(user_approval_receipt=str(receipt), user_approval_receipt_sha256=sha(receipt)) if receipt else {})),
        preproduction_seconds=prior_seconds, remaining_seconds=resources['total_seconds']-prior_seconds,
        submit_command=f'sbatch --time={min(remaining_minutes, 1440)} {batch} <approval.json>',
        created_utc=datetime.now(timezone.utc).isoformat())
    put(package/'request-pending.json', pending)
    return proposal, pending


def tail_cells(root, stage, actors):
    """Unclaimed stage cells of a finished predecessor campaign (claims are never rerun)."""
    if root is None:
        return {}
    predecessor = json.loads((Path(root).parent/'cohort.json').read_text())
    cells = [c for c in predecessor['cells'] if c['stage'] == stage and (not actors or c['actor'] in actors)]
    claimed = {json.loads(p.read_text())['cell_id'] for p in (Path(root)/'queue'/'claims').glob('*.json')}
    remaining = [c['cell_id'] for c in cells if c['cell_id'] not in claimed]
    if not remaining:
        raise SystemExit('Predecessor claimed every stage cell; no tail campaign needed')
    return dict(cell_ids=remaining, predecessor=dict(campaign_root=str(root), claimed=len(claimed),
                                                     remaining=len(remaining)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=sorted(om.STAGES))
    parser.add_argument('--actors', nargs='+', help='subset of Table 5 actors (default: both)')
    parser.add_argument('--tail-of', type=Path, help='campaign root of a capped predecessor; run only its unclaimed stage cells')
    parser.add_argument('--api-usd', required=True, type=float)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--receipt', type=Path, help='user approval receipt to bind by sha256')
    parser.add_argument('--preproduction-ledger', type=Path, help='hash-bound earlier allocation accounting charged to this cap')
    parser.add_argument('--candidate-controls', action='store_true', help='Prepare the fixed, separate 8-task x 6-selector random-control diagnostic')
    parser.add_argument('--candidate-control-tasks', type=int, choices=(8,100), default=8,
                        help='Fixed fresh study:8 diagnostic tasks or100 disjoint paired tasks; no submission')
    parser.add_argument('--focused-controls', action='store_true', help='Approved four-method100-task study; requires USD500 API cap')
    args = parser.parse_args()
    package = args.package or RUNTIME/f'{args.stage}-v1'
    proposal, pending = build(args.stage, args.api_usd, package, receipt=args.receipt,
                              preproduction_ledger=args.preproduction_ledger, actors=args.actors,
                              candidate_controls=args.candidate_controls or args.focused_controls,
                              candidate_control_tasks=args.candidate_control_tasks,
                              focused_controls=args.focused_controls,
                              **tail_cells(args.tail_of, args.stage, args.actors))
    print(json.dumps(dict(package=str(package), proposal_sha256=pending['proposal_sha256'], limits=proposal['limits'],
                          resources=proposal['resources']), indent=1))


if __name__ == '__main__':
    main()
