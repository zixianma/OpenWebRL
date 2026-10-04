#!/usr/bin/env python3
"""Bounded CPU audit of saved group metadata, never tensor/image storage records."""
import argparse
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import statistics
import time
import zipfile


class MetadataOnly(pickle.Unpickler):
    """Allow only the tensor stubs used by our own torch.save dictionaries."""
    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or pid[0] != 'storage':
            raise ValueError('Unexpected persistent object')
        return None

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if module == 'torch' and name in ('FloatStorage', 'LongStorage', 'BFloat16Storage', 'BoolStorage', 'ByteStorage'):
            return type(name, (), {})
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return lambda storage, offset, size, stride, *args: {'_tensor_shape': list(size)}
        raise ValueError('Unrecognized pickle global: ' + module + '.' + name)


def load_metadata(path):
    with zipfile.ZipFile(path) as archive:
        members = [i for i in archive.infolist() if i.filename.endswith('/data.pkl')]
        if len(members) != 1 or members[0].file_size > 64 * 1024**2:
            raise ValueError('Unexpected/oversized metadata member')
        data = archive.read(members[0])
    return MetadataOnly(io.BytesIO(data)).load(), len(data)


def reason_category(reason):
    if reason.startswith('generation_error:'):
        if '/reset' in reason:
            if 'ERR_NAME_NOT_RESOLVED' in reason: return 'reset_dns_error'
            if 'ERR_CONNECTION_CLOSED' in reason: return 'reset_connection_closed'
            if 'Timeout' in reason: return 'reset_navigation_timeout'
            return 'reset_other_error'
        if 'rollout_task_timeout' in reason: return 'rollout_task_timeout'
        return 'generation_other_error'
    return reason or 'missing_termination_reason'


def rejection(rows, excluded):
    """Audit the frozen validity predicate on scalar metadata, excluding tensors."""
    terminal = rows[-1]; metadata = terminal['metadata']; reward = metadata.get('reward', {})
    if any(s.get('remove_sample') or s['metadata'].get('arm_calibration_excluded') for s in rows):
        return 'removed_or_excluded'
    if any(s.get('reward') != 0 for s in rows): return 'missing_or_nonzero_reward'
    if any(s['response_length'] <= 0 or (s.get('loss_mask') is not None and not any(s['loss_mask'])) for s in rows):
        return 'no_trainable_response'
    if len({str(s['metadata'].get('trajectory_id', s['index'])) for s in rows}) != 1:
        return 'inconsistent_trajectory_identity'
    tasks = {str(s['metadata'].get('task_id', '')) for s in rows}
    if len(tasks) != 1 or not next(iter(tasks)): return 'inconsistent_trajectory_identity'
    if next(iter(tasks)) in excluded: return 'evaluation_overlap'
    if metadata.get('judge_timeout') or reward.get('judge_timeout'): return 'judge_timeout'
    if reward.get('combined') != 0 or reward.get('judge_prompt_variant') != 'action_history':
        return 'missing_native_reward_provenance'
    reason, status = metadata.get('terminate_reason'), terminal.get('status')
    if status == 'failed' and reason == 'max_steps_exhausted': return None
    if status != 'completed' or reason != 'task_completed': return 'invalid_termination'
    if reward.get('judge') != 0 or 'NOT SUCCESS' not in str(reward.get('judge_text', '')):
        return 'unverified_judge_failure'
    return None


def audit(root):
    started = time.monotonic()
    manifest = json.loads((root/'launch_manifest.json').read_text())
    configs = sorted((root/'iterations').glob('*/arm-config.json'))
    if len(configs) != 1: raise ValueError('Expected a single-collection pilot')
    folder = configs[0].parent
    reports = sorted((folder/'groups').glob('*/*.json'))
    config = json.loads(configs[0].read_text())
    excluded = set()
    for name in config.get('excluded_task_files', []):
        value = json.loads(Path(name).read_text())
        if isinstance(value,dict): excluded.update(value.get('task_ids', []))
    populations = defaultdict(list); groups=[]; metadata_bytes=0; aborted_zero_ids=set()
    for path in reports:
        journal = json.loads(path.read_text())
        payload, count = load_metadata(path.with_suffix('.pt')); metadata_bytes += count
        assert payload['policy_id'] == journal['policy_id']
        assert payload['filter_reason'] == journal['filter_reason']
        assert [s['index'] for s in payload['samples']] == [s['index'] for s in journal['rows']]
        trajectories = defaultdict(list)
        for sample in payload['samples']:
            trajectories[str(sample['metadata'].get('trajectory_id',sample['index']))].append(sample)
        group_rows=[]
        for identity, rows in trajectories.items():
            ts=rows[-1];md=ts['metadata'];reason=reason_category(str(md.get('terminate_reason','')))
            valid_failure_reason = rejection(rows, excluded)
            # Only scalar counts go to the report; no prompts, screenshots or API payloads.
            row=dict(turns=len(rows), response_lengths=[s['response_length'] for s in rows],
                last_response_length=ts['response_length'],status=ts['status'],reason=reason,
                removed=any(s['remove_sample'] for s in rows), invalid_reason=valid_failure_reason,
                reward=ts['reward'], sampled=sum(bool(s['metadata'].get('arm_turn_bonus',{}).get('sampled')) for s in rows),
                labeled=sum(bool(s['metadata'].get('arm_turn_bonus',{}).get('eligible')) for s in rows),
                closed_reasoning='</think>' in ts.get('response',''), complete_tool_call='</tool_call>' in ts.get('response',''))
            group_rows.append(row);populations['all'].append(row)
            if journal['filter_reason']=='zero_std_0.0':
                populations['zero_outcome_groups'].append(row)
                if valid_failure_reason is None: populations['valid_failures_in_zero_groups'].append(row)
                if ts['status']=='aborted':
                    aborted_zero_ids.update(str(md.get(k,'')) for k in ('task_id','old_task_id'))
            if journal['accepted']:populations['ordinary_retained'].append(row)
            if valid_failure_reason is None:populations['individually_valid_failures'].append(row)
        groups.append(dict(id=path.stem,accepted=journal['accepted'],filter_reason=journal['filter_reason'],
            first_rejection=next((r['invalid_reason'] for r in group_rows if r['invalid_reason']),None),
            valid_failure_trajectories=sum(r['invalid_reason'] is None for r in group_rows),
            reasons=sorted(set(r['reason'] for r in group_rows)),
            zero_outcome=journal['filter_reason']=='zero_std_0.0'))
        del payload, trajectories
    def summarize(rows):
        lengths=[r['turns'] for r in rows]
        return dict(trajectories=len(rows),turns=sum(lengths),
            length_histogram=dict(sorted(Counter(lengths).items())), median_turns=statistics.median(lengths) if lengths else None,
            statuses=dict(Counter(r['status'] for r in rows)),reasons=dict(Counter(r['reason'] for r in rows)),
            invalid_reasons=dict(Counter(str(r['invalid_reason']) for r in rows)),
            last_response_length_for_truncated=dict(Counter(r['last_response_length'] for r in rows if r['status']=='truncated')),
            truncated_response_markers=dict(closed_reasoning=sum(r['closed_reasoning'] for r in rows if r['status']=='truncated'), complete_tool_call=sum(r['complete_tool_call'] for r in rows if r['status']=='truncated')),
            expected_twenty_percent_states=.2*sum(lengths), four_turn_states=sum(min(4,t) for t in lengths),
            eight_turn_states=sum(min(8,t) for t in lengths), all_turn_states=sum(lengths),
            sampled_turns=sum(r['sampled'] for r in rows), admitted_turns=sum(r['labeled'] for r in rows))
    zero=[g for g in groups if g['zero_outcome']]
    log=(root/'collection.log').read_text(errors='replace');step_errors=Counter()
    for line in log.splitlines():
        m=re.search(r'Task (\S+) step (\d+): env.step\(\) failed, marking as ABORTED:(.*)', re.sub(r'\x1b\[[0-9;]*m','',line))
        if m and m[1] in aborted_zero_ids:step_errors[m[3].strip() or '(empty exception text)']+=1
    actual=json.loads((folder/'failure-coverage.json').read_text())['rejected_groups']
    observed=dict(Counter(g['first_rejection'] for g in zero))
    assert observed==actual,(observed,actual)
    return dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(root),checkpoint=manifest['checkpoint'],group_count=len(groups),
        filter_reasons=dict(Counter(str(g['filter_reason']) for g in groups)),
        populations={k:summarize(v) for k,v in populations.items()},
        zero_groups=dict(count=len(zero),first_rejection=observed,
            valid_failure_count_histogram=dict(sorted(Counter(g['valid_failure_trajectories'] for g in zero).items())),
            groups_containing_reason=dict(Counter(r for g in zero for r in g['reasons'])),
            all_five_valid=sum(g['valid_failure_trajectories']==5 for g in zero)),
        matched_zero_group_abort_messages=dict(step_errors), context_limit_log_events=log.count('exceeds max_context_len'),
        metadata_bytes_read=metadata_bytes,tensor_storage_bytes_read=0,
        elapsed_seconds=time.monotonic()-started,
        scope='Scalar metadata audit; four/eight-turn counts are potential selected states, not admitted labels. Individual-valid counts do not relax the five-valid-failure group rule.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
