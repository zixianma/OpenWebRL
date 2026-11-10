import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('paired_report', Path(__file__).resolve().parents[1]/'scripts/report_arm_paired_inference.py')
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
TEXT = dict(hidden_size=8, num_attention_heads=3, head_dim=4, num_key_value_heads=1,
            intermediate_size=10, num_hidden_layers=2, vocab_size=20)


def row(task, mode, valid=True, reward=1):
    data = dict(task_id=task, mode=mode, valid=valid, reward=reward)
    data.update({k: 1 for k in ('actor_requests', 'actor_output_tokens', 'actor_input_tokens',
        'actor_client_request_seconds', 'browser_steps', 'browser_step_seconds', 'selector_calls',
        'selector_input_tokens', 'selector_output_tokens', 'selector_service_seconds', 'elapsed_seconds')})
    return data


def test_preview_uses_same_complete_tasks_keeps_invalids_and_rejects_duplicates():
    rows = [row('complete', mode, mode != 'arm') for mode in report.MODES]
    rows.append(row('fast_unmatched', 'arm'))
    tasks, matched = report.matched_records(rows, ['complete', 'fast_unmatched'])
    assert tasks == ['complete'] and len(matched) == 3
    result = report.summarize(matched, tasks, True)
    assert result['by_mode']['arm']['overall'] == 0
    assert result['by_mode']['actor0']['overall'] == 1
    with pytest.raises(ValueError, match='duplicate'):
        report.matched_records(rows + rows[:1], ['complete', 'fast_unmatched'])


def test_failed_attempts_share_each_shards_time_cap_without_transfer():
    approval = dict(approved=True, resources=report.RESOURCES, limits=report.LIMITS,
                    attempts=[dict(job_id='1'), dict(job_id='2')])
    scheduler = [dict(job_id='1', state='FAILED', elapsed_seconds=100),
                 dict(job_id='2', state='COMPLETED', elapsed_seconds=14300)]
    ledger = dict(calls=1, charged_or_reserved_usd=.1)
    receipts = [dict(status='received', charged_usd=.1)]
    result = report.shard_accounting(approval, scheduler, ledger, receipts, True)
    assert result['allocated_gpu_hours'] == 8 and result['final_accounting_verified']
    scheduler[1]['elapsed_seconds'] += 1
    with pytest.raises(ValueError, match='cap'):
        report.shard_accounting(approval, scheduler, ledger, receipts, False)


def test_final_report_requires_terminal_scheduler_and_reconciled_judge_receipts():
    approval = dict(approved=True, resources=report.RESOURCES, limits=report.LIMITS,
                    attempts=[dict(job_id='1')])
    scheduler = [dict(job_id='1', state='RUNNING', elapsed_seconds=100)]
    ledger = dict(calls=1, charged_or_reserved_usd=.1)
    receipts = [dict(status='received', charged_usd=.1)]
    assert not report.shard_accounting(approval, scheduler, ledger, receipts, False)['final_accounting_verified']
    with pytest.raises(ValueError, match='not available'):
        report.shard_accounting(approval, scheduler, ledger, receipts, True)
    scheduler[0]['state'] = 'COMPLETED'
    with pytest.raises(ValueError, match='reconcile'):
        report.shard_accounting(approval, scheduler, ledger, [], True)
    receipts[0]['status'] = 'reserved'
    with pytest.raises(ValueError, match='not available'):
        report.shard_accounting(approval, scheduler, ledger, receipts, True)


@pytest.mark.parametrize('mode', ['arm', 'piotr', 'actor0'])
def test_compute_includes_both_selector_variants_and_counts_failed_requests(tmp_path, mode):
    artifact = tmp_path/'result.json'; artifact.write_text('{}')
    received = dict(status='received', meta=dict(prompt_tokens=5, completion_tokens=4, cached_tokens=2))
    (tmp_path/'actor-request-0.json').write_text(json.dumps(received))
    (tmp_path/'actor-request-1.json').write_text(json.dumps(dict(status='failed')))
    selector = mode != 'actor0'
    trace = tmp_path/'trace.jsonl'
    trace.write_text(json.dumps(dict(selector_telemetry=dict(input_tokens=6, output_tokens=2)))+'\n')
    r = dict(artifact=str(artifact), mode=mode, actor_requests=2, actor_input_tokens=5, actor_output_tokens=4,
             selector_input_tokens=6 if selector else 0, selector_output_tokens=2 if selector else 0,
             selector_calls=int(selector), trace_files=[dict(path=str(trace))])
    larger = dict(TEXT, num_hidden_layers=3)
    result = report.episode_decoder_work(r, TEXT, larger)
    extra = report.decoder_flops(larger, 6, 2)['total'] if selector else 0
    assert result['decoder_cold_flops'] == report.decoder_flops(TEXT, 5, 4)['total'] + extra
    assert result['decoder_observed_kv_flops'] == report.decoder_flops(TEXT, 5, 4, 2)['total'] + extra
    assert result['unmetered_failed_actor_requests'] == 1
    r['actor_input_tokens'] += 1
    with pytest.raises(ValueError, match='token totals'):
        report.episode_decoder_work(r, TEXT, larger)
    r['actor_input_tokens'] -= 1
    received.update(sampling_seed=42, sampling_params=dict(
        temperature=1., top_p=.95, max_new_tokens=1024, sampling_seed=42))
    (tmp_path/'actor-request-0.json').write_text(json.dumps(received))
    with pytest.raises(ValueError, match='decoding receipt'):
        report.episode_decoder_work(r, TEXT, larger)
