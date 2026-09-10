"""Lossless text/raw-image archive of completed groups, including RL rejections.

Archival is independent of training selection. Processed image tensors are
deliberately excluded: retain original observations and reprocess them for SFT.
"""
import base64
import gzip
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import time
import uuid
from enum import Enum


class ArchiveEncoder:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / 'images').mkdir(parents=True, exist_ok=True)
        self.images = set()

    def image(self, payload, mime):
        digest = hashlib.sha256(payload).hexdigest()
        relative = f'images/{digest}.bin'
        path = self.root / relative
        if digest not in self.images:
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Image archive integrity failure: {relative}')
            else:
                temp = path.with_suffix(f'.{uuid.uuid4().hex}.partial')
                with temp.open('xb') as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, path)
            self.images.add(digest)
        return {'archive_image': relative, 'sha256': digest,
                'mime_type': mime, 'bytes': len(payload)}

    def encode(self, value):
        if isinstance(value, Enum):
            return self.encode(value.value)
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {'nonfinite_float': str(value)}
        if isinstance(value, str):
            if value.startswith('data:image/') and ';base64,' in value:
                header, encoded = value.split(',', 1)
                return self.image(base64.b64decode(encoded, validate=True), header[5:].split(';')[0])
            return value
        if isinstance(value, (bytes, bytearray)):
            return self.image(bytes(value), 'application/octet-stream')
        if isinstance(value, dict):
            return {str(k): self.encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.encode(v) for v in value]
        # PIL is optional; no image decoding is needed for existing data URLs.
        if type(value).__module__.startswith('PIL.'):
            stream = io.BytesIO()
            value.save(stream, format='PNG')
            return self.image(stream.getvalue(), 'image/png')
        if hasattr(value, 'tolist'):
            size = value.numel() if callable(getattr(value, 'numel', None)) else value.size
            if size > 4096:
                raise TypeError('Unexpected large array outside processed image tensors')
            return self.encode(value.tolist())
        raise TypeError(f'Unsupported archive value: {type(value).__name__}')


def _turns(trajectory):
    return trajectory if isinstance(trajectory, list) else [trajectory]


def _outcome(args, trajectory):
    turns = _turns(trajectory)
    if not turns:
        return {'valid': False, 'reward': None, 'success': False}
    last = max(turns, key=lambda s: (s.metadata or {}).get('turn_index', 0))
    reward = last.get_reward_value(args) if last.reward is not None else None
    valid = not any(s.remove_sample or getattr(s.status, 'value', s.status) == 'aborted' for s in turns)
    valid = valid and reward is not None and math.isfinite(float(reward))
    return {'valid': bool(valid), 'reward': float(reward) if valid else None,
            'success': bool(valid and reward == 1)}


def _classification(outcomes):
    if not outcomes or not all(x['valid'] for x in outcomes):
        return 'contains_invalid'
    rewards = [x['reward'] for x in outcomes]
    if all(r == 1 for r in rewards):
        return 'all_success'
    if all(r == 0 for r in rewards):
        return 'all_failure'
    if all(r <= 0 for r in rewards):
        return 'all_nonpositive'
    return 'mixed'


def archive_completed_groups(args, completed_groups, accepted_groups, root, iteration):
    """Write complete group records and a final manifest; never modify samples."""
    start = time.monotonic()
    root = Path(root)
    batch = root / f'iteration_{iteration:04d}_{uuid.uuid4().hex[:12]}'
    batch.mkdir(parents=True, exist_ok=False)
    encoder = ArchiveEncoder(root)
    accepted_turns = {id(s) for g in accepted_groups for t in g for s in _turns(t)}
    fields = ('group_index', 'index', 'prompt', 'response', 'tokens', 'loss_mask',
              'response_length', 'reward', 'status', 'remove_sample', 'metadata',
              'multimodal_inputs', 'weight_versions', 'rollout_log_probs')
    records = []
    counts = {}
    for ordinal, group in enumerate(completed_groups):
        outcomes = [_outcome(args, t) for t in group]
        classification = _classification(outcomes)
        trajectories = []
        for trajectory, outcome in zip(group, outcomes):
            turns = _turns(trajectory)
            trajectories.append({**outcome,
                'accepted_for_rl': any(id(s) in accepted_turns for s in turns),
                'turns': [{name: encoder.encode(getattr(s, name, None)) for name in fields} for s in turns]})
        accepted = any(t['accepted_for_rl'] for t in trajectories)
        disposition = 'accepted' if accepted else 'not_accepted'
        counts[f'{disposition}/{classification}'] = counts.get(f'{disposition}/{classification}', 0) + 1
        payload = {'schema_version': 1, 'reward_iteration': iteration,
                   'rollout_id_zero_based': iteration - 1,
                   'group_ordinal': ordinal, 'classification': classification,
                   'accepted_for_rl': accepted, 'trajectories': trajectories}
        path = batch / f'group_{ordinal:04d}.json.gz'
        temp = path.with_suffix('.partial')
        with gzip.open(temp, 'xt', encoding='utf-8', compresslevel=1) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        with temp.open('rb') as stream:
            os.fsync(stream.fileno())
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.replace(temp, path)
        records.append({'file': path.name, 'bytes': path.stat().st_size,
                        'classification': classification, 'accepted_for_rl': accepted})
    manifest = {'schema_version': 1, 'complete': True, 'reward_iteration': iteration,
                'run_directory': str(getattr(args, 'save', '')),
                'groups': len(records), 'trajectories': sum(map(len, completed_groups)),
                'classification_counts': counts, 'unique_images': len(encoder.images),
                'image_reference_root': str(root.resolve()), 'records': records,
                'elapsed_seconds': time.monotonic() - start,
                'coverage': 'All completed groups at collection cutoff; excludes canceled in-flight attempts.',
                'not_accepted_meaning': 'No turn entered the accepted batch; may include excess completed groups at cutoff.',
                'processed_multimodal_tensors_saved': False}
    with (batch / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def maybe_archive_completed_groups(args, completed_groups, accepted_groups):
    """Optional telemetry hook, enabled by a per-run control file or env var.

    Read the completed collection identity from its progress record because the
    existing telemetry interface does not receive rollout_id. Refuse ambiguity.
    Archive failures are visible but never change RL filtering or fail training.
    """
    save = getattr(args, 'save', None)
    if not save:
        return {}
    control = Path(save) / 'rollout_archive.enabled.json'
    enabled = os.environ.get('OPENWEBRL_ARCHIVE_COMPLETED_GROUPS') == '1' or control.exists()
    if not enabled:
        return {}
    try:
        progress = (Path(save) / 'progress.log').read_text()
        events = re.findall(r'\[GenerateProgress\] rollout=(\d+)/\d+ event=done ', progress)
        if not events:
            raise ValueError('No completed collection identity in progress.log')
        iteration = int(events[-1])
        report = archive_completed_groups(args, completed_groups, accepted_groups,
                                          Path(save) / 'completed_rollout_archive', iteration)
        logging.getLogger(__name__).info('[RolloutArchive] iteration=%s groups=%s images=%s seconds=%.1f',
            iteration, report['groups'], report['unique_images'], report['elapsed_seconds'])
        return {'rollout/archive/groups': report['groups'],
                'rollout/archive/trajectories': report['trajectories'],
                'rollout/archive/unique_images': report['unique_images'],
                'rollout/archive/seconds': report['elapsed_seconds'],
                'rollout/archive/errors': 0}
    except Exception as exc:
        logging.getLogger(__name__).exception('[RolloutArchive] failed')
        try:
            with (Path(save) / 'rollout_archive_errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'time': time.time(), 'error': str(exc)}) + '\n')
        except OSError:
            logging.getLogger(__name__).exception('[RolloutArchive] error receipt could not be written')
        return {'rollout/archive/errors': 1}
