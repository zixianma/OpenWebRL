#!/usr/bin/env python3
"""Freeze a no-download, no-model native shopping reference validation proposal.

Offline preparation only. Exact new allocation approval is required to launch.
This freezes tested components and a supervised stage plan, not a claim that
their composition is already live-validated or an unattended deployment worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from prepare_harmful_apptainer_probe import writable_path, MANIFEST_SHA256, UPSTREAM_COMMIT
from openwebrl.harmful_shopping_admission import CASES, TASK_IDS, REQUIRED_CHECKS

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/harmful-apps-20261010')
CACHE_SHA256 = '4a24ce219d923bd0bce3504f1a5b6245704518b2f2d065eeaedb2df5619be2b3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def resource_request():
    return dict(schema_version=1, status='proposed_not_authorized',
        nodes=1, cpus=4, memory_gib=16, gpus=0,
        total_allocation_seconds_including_all_attempts=14400,
        maximum_allocated_cpu_hours=16, api_budget_usd=0,
        actor_and_judge_calls=0, internet_download_bytes=0,
        node_local_scratch_cap_bytes=200 * 1024**3,
        retained_artifacts_cap_bytes=30 * 1024**3,
        maximum_fresh_instances_including_failed_attempts=48,
        maximum_browser_operations_per_reference=60,
        maximum_browser_seconds_per_reference=120,
        older_allocation_seconds_added=0, other_session_budget_added=0,
        live_agent_supervision_required=True, shared_instance_mutation=False,
        gitlab_in_scope=False, model_comparison_in_scope=False,
        purpose='Native capture, reset and scripted shopping-reference validation',
        early_release='Release after verified endpoint or exhausted useful work; never carry surplus time',
        scope=('Use cached pinned Magento baseline; prove native DB/SMTP capture, reset, queue and '
               'isolation, then up to eight development and 36 held-out reference cases. '
               'Every retry consumes the same aggregate caps. Unsupported capture invalidates '
               'the affected references; no actor calibration or model experiment is authorized.'))


def stages():
    return [
        dict(id='allocation_and_cache', requires=['Exact request approval and authorization binding',
            'Scheduler identity and cumulative all-attempts time ledger',
            'Verify pinned cached SIF and frozen sources before service startup'],
            output='allocation receipt, source inventory and absolute work deadline'),
        dict(id='capture_runtime', requires=['Fresh scratch sandbox; never start immutable baseline',
            'Reviewed MariaDB ROW/FULL/CRC32 settings before mysqld starts',
            'Evaluator-owned SMTP sink replaces Mailcatcher; all outgoing mail intercepted',
            'Isolated user/network namespace, no external route, browser app-origin restriction',
            'All writable paths/caches in approved roots; owned process supervisor'],
            output='reviewed startup configuration, native schema and routing evidence'),
        dict(id='capture_and_reset_selftests', requires=list(REQUIRED_CHECKS),
            output='independently reviewed native boundary/rollback/SMTP/reset proof'),
        dict(id='development_references', requires=['Two prospective development orders outside every held-out target',
            'Freeze contracts from native DB identities and exact rendered MIME',
            'Correct, wrong-target, repeated submission, wrong-then-correct browser references',
            'Fresh full reset between cases; preserve effects within each case'],
            output='eight development reference traces; fixes allowed, every attempt retained'),
        dict(id='held_out_references', requires=['Freeze nine contracts before any model policy calls',
            'Freeze capture/reference source revisions after development',
            'All four independent cases per task; native cancellation repeat may be a browser no-op',
            'Any code correction records a new revision and repeats affected validation within caps'],
            output='36 native reference traces; zero tasks admitted from missing or unknown evidence'),
        dict(id='close_and_audit', requires=['Reconstruct final tracked rows from complete committed capture',
            'Bind each accepted SMTP DATA to immutable receipts, preserving duplicates',
            'Verify queue quiescence, process closure, artifact seals and total scheduler usage'],
            output='per-task admitted/invalid outcomes, complete/partial distinction, closed allocation'),
    ]


def runtime_changes():
    """Exact proposed changes, requiring verification against the actual startup."""
    return dict(status='proposed_runtime_integration_not_live_validated',
        preserve_populated_database=True, bypass_inherited_entrypoint=True,
        inherited_services=['mysql', 'nginx', 'php-fpm', 'redis'],
        excluded_services=['mailcatcher', 'cron', 'elasticsearch', 'env-ctrl', 'env-ctrl-init'],
        mail_transport='Replace local SMTP port 25 with evaluator-owned durable sink; verify every application send path',
        smtp_limits=dict(max_messages=32, max_message_bytes=2*1024**2,
            max_storage_bytes=128*1024**2, max_runtime_seconds=600,
            external_delivery=False, listen_host='127.0.0.1', listen_port=25),
        mysql_additional_settings={'server-id': '1', 'log-bin': '/var/lib/mysql/harmful-native-bin',
            'binlog-format': 'ROW', 'binlog-row-image': 'FULL', 'binlog-checksum': 'CRC32',
            'binlog-annotate-row-events': 'OFF', 'sync-binlog': '1',
            'innodb-flush-log-at-trx-commit': '1', 'max-binlog-size': str(1024**3),
            'expire-logs-days': '0', 'log-bin-compress': 'OFF'},
        binlog_window_max_bytes=512*1024**2,
        policy='Fail closed on unsupported native formats, any rotation in window, incomplete capture or missing queue evidence',
        case_deadline_seconds=600,
        state_strategy='One fresh sandbox from pinned immutable cached baseline per reference; no reset within wrong-then-correct',
        ownership='Own and await all service/browser/sink descendants; verify closure before next instance',
        actor_access='Browser origin only; no filesystem, DB, SMTP, collector or control tools',
        network='Isolated user/network namespace with loopback only',
        worker_writes='Approved node-local scratch and retained runtime root only; no home bind',
        source_changes='Review source hash changes and rerun affected validation without resetting budget')


def source_files():
    names = [
        'openwebrl/__init__.py', 'openwebrl/jev_eval.py',
        'openwebrl/harmful_app_trace.py', 'openwebrl/harmful_shopping_audit.py',
        'openwebrl/harmful_shopping_capture.py', 'openwebrl/harmful_smtp_sink.py',
        'openwebrl/harmful_shopping_references.py', 'openwebrl/harmful_shopping_admission.py',
        'openwebrl/webprm_lifecycle.py', 'scripts/prepare_harmful_apptainer_probe.py',
        'scripts/prepare_harmful_shopping_validation.py',
        'scripts/prepare_harmful_shopping_references.py',
        'scripts/run_harmful_shopping_reference.py', 'scripts/run_harmful_smtp_sink.py',
        'scripts/audit_harmful_shopping_admission.py', 'scripts/audit_harmful_app_trace.py',
    ]
    # Native boundary helper is a fixed reviewed source, not an arbitrary command.
    names += [str(p.relative_to(REPO)) for p in sorted((REPO / 'scripts').glob('*harmful*shopping*.php'))]
    return names


def prepare(output, reference_manifest, validation_receipt):
    output = writable_path(output)
    reference_manifest, validation_receipt = Path(reference_manifest), Path(validation_receipt)
    manifest = json.loads(reference_manifest.read_text())
    validation = json.loads(validation_receipt.read_text())
    if (manifest.get('schema_version') != 1 or manifest.get('image_sha256') != MANIFEST_SHA256 or
            manifest.get('source_commit') != UPSTREAM_COMMIT or
            sorted(t['task_id'] for t in manifest.get('tasks', [])) != list(TASK_IDS) or
            len(manifest.get('development', [])) != 2 or manifest.get('reference_case_count') != 44 or
            manifest.get('admitted_task_count') != 0):
        raise ValueError('Expected pinned nine-task/two-development unresolved reference manifest')
    if validation.get('status') != 'passed' or validation.get('live_application_validation') is not False:
        raise ValueError('Expected explicit passed offline validation receipt')
    cache = RUNTIME / 'shopping-admin-probe-20261011/shopping-admin-baseline.sif'
    if sha(cache) != CACHE_SHA256:
        raise ValueError('Cached baseline changed; no new image download is authorized')
    files = {name: sha(REPO / name) for name in source_files()}
    checked_sources = validation.get('source_sha256', {})
    if any(checked_sources.get(name) != value for name, value in files.items()):
        raise ValueError('Offline validation receipt does not bind every current source')
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        'resource-request.json': resource_request(),
        'stage-plan.json': stages(),
        'runtime-changes.json': runtime_changes(),
        'reference-manifest.json': manifest,
        'offline-validation.json': validation,
        'source-inventory.json': files,
    }
    for name, value in artifacts.items():
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    for name in files:
        data = (REPO / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != files[name]:
            raise ValueError('Source changed during proposal freeze: ' + name)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    proposal = dict(schema_version=1, status='offline_prepared_pending_exact_resource_approval',
        launch_authorized=False, live_validation_complete=False, admitted_tasks=0,
        image_sha256=MANIFEST_SHA256, upstream_commit=UPSTREAM_COMMIT,
        cached_baseline={'path': str(cache), 'sha256': CACHE_SHA256, 'bytes': cache.stat().st_size},
        source_files=len(files), task_ids=list(TASK_IDS), reference_cases=list(CASES),
        development_reference_cases=8, held_out_reference_cases=36,
        split='entity_disjoint_within_template; no template-generalization claim',
        artifacts_sha256={name: sha(output / name) for name in artifacts},
        workflow='Actively supervised stage plan; runtime integration validated inside bounded allocation',
        scope_limit='Prepares task admission, not a model result or demonstrated safety advantage')
    with (output / 'proposal.json').open('x') as stream:
        json.dump(proposal, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--validation-receipt', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output, args.reference_manifest, args.validation_receipt)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
