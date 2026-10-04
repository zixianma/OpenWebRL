"""Restricted TP-only ARM auxiliary integration for the first GPU mechanics gate.

Auxiliary tensors travel through a per-collection, immutable file manifest,
outside outcome reward normalization. The pilot deliberately rejects DP/CP/PP
replication until those configurations receive their own transport tests.
"""
from copy import copy
import hashlib
import json
import math
import os
from pathlib import Path


FIELDS = ('arm_source', 'arm_scale', 'arm_advantage', 'arm_old_log_probs')


def enabled():
    return bool(os.environ.get('OPENWEBRL_ARM_AUX_MANIFEST'))


def check_topology(args, *, dp=1, cp=1, pp=1):
    if (dp, cp, pp) != (1, 1, 1) or args.micro_batch_size != 1:
        raise ValueError('ARM pilot requires DP1/CP1/PP1 and microbatch 1')
    if args.calculate_per_token_loss or args.entropy_coef != 0 or args.use_kl_loss:
        raise ValueError('ARM pilot requires baseline sample-mean loss, entropy=KL=0')
    if args.loss_type != 'policy_loss' or args.advantage_estimator != 'grpo':
        raise ValueError('ARM pilot requires baseline GRPO policy loss')
    if args.rollout_temperature != .8:
        raise ValueError('The audited candidate old-policy temperature is .8')
    if getattr(args, 'qkv_format', 'thd') != 'thd' or getattr(args, 'use_dynamic_batch_size', False):
        raise ValueError('ARM pilot requires the baseline thd/static-microbatch packing')


def prepare_auxiliary(actor, rollout_id):
    """Called before any update in this collection, with its actor restored."""
    if not enabled():
        return None
    import torch
    from megatron.core import mpu
    from slime.backends.megatron_utils.data import DataIterator

    check_topology(actor.args, dp=mpu.get_data_parallel_world_size(False),
                   cp=mpu.get_context_parallel_world_size(), pp=mpu.get_pipeline_model_parallel_world_size())
    if len(actor.model) != 1:
        raise ValueError('Virtual pipeline stages are not yet validated for ARM')
    path = Path(os.environ['OPENWEBRL_ARM_AUX_MANIFEST'].format(rollout_id=rollout_id))
    manifest = json.loads(path.read_text())
    expected_policy = f"{os.environ.get('OPENWEBRL_ARM_POLICY_ID')}:round{rollout_id}"
    if manifest['rollout_id'] != rollout_id or manifest['policy_id'] != expected_policy:
        raise ValueError('Stale auxiliary manifest/policy identity')
    if manifest.get('schema_version') != 1 or manifest.get('training_judge') != 'gpt-4.1/action_history':
        raise ValueError('Auxiliary manifest schema/judge mismatch')
    eta, lam = manifest['eta'], manifest['lambda']
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (eta, lam)):
        raise ValueError('Invalid auxiliary coefficients')
    if eta and lam:
        raise ValueError('The first two tracks are independent, not combined')
    if not eta and not lam:
        return None
    tensor_path = path.parent / manifest['tensor_file']
    # A pilot artifact is bounded; never deserialize full rollout recovery here.
    if tensor_path.stat().st_size > 2 * 1024**3:
        raise ValueError('Auxiliary pilot tensor file exceeds 2 GiB')
    h = hashlib.sha256()
    with tensor_path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    if h.hexdigest() != manifest['tensor_sha256']:
        raise ValueError('Auxiliary tensor hash mismatch')
    records = torch.load(tensor_path, map_location='cpu', weights_only=True)
    if len(records) != len(manifest['records']):
        raise ValueError('Auxiliary manifest/tensor row count mismatch')
    if not records:
        return None
    data = {k: [] for k in ['tokens', 'total_lengths', 'response_lengths', 'loss_masks', 'multimodal_train_inputs']}
    for meta, row in zip(manifest['records'], records, strict=True):
        if meta['source'] not in ('rescue_demo', 'arm_candidate'):
            raise ValueError('Unexpected auxiliary source')
        if row.get('terminal_reward') is not None:
            raise ValueError('Auxiliary payloads must not contain terminal returns')
        tokens, mask, n = row['tokens'], row['loss_mask'], row['response_length']
        if tokens.ndim != 1 or mask.ndim != 1 or len(mask) != n or not 0 < n < len(tokens):
            raise ValueError('Auxiliary response/token shape mismatch')
        if len(tokens) > 32768 or not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.sum() > 0):
            raise ValueError('Invalid auxiliary context or supervision mask')
        for key, value in dict(tokens=tokens.cuda(), total_lengths=len(tokens), response_lengths=n,
                               loss_masks=mask.cuda(), multimodal_train_inputs=row['multimodal_train_inputs']).items():
            data[key].append(value)
    # Raw SGLang records contain synthetic zero logps on appended end tokens.
    # Evaluate the complete response under the CURRENT frozen actor instead.
    iterator = [DataIterator(data, micro_batch_indices=[[i] for i in range(len(records))])]
    old = actor.compute_log_prob(iterator, [len(records)])['log_probs']
    if len(old) != len(records):
        raise ValueError('Missing candidate old-policy scores')
    if any(not bool(torch.isfinite(x).all()) for x in old):
        raise ValueError('Nonfinite frozen-actor candidate log probabilities')
    data['arm_old_log_probs'] = [x.detach() for x in old]
    return dict(data=data, manifest=manifest)


def attach_auxiliary(args, iterators, counts, prepared):
    """Append extra microbatches within each original optimizer window.

    The original outcome indices, their order, optimizer count and denominator
    remain fixed. Only the number of accumulated forward/backward passes grows.
    """
    if prepared is None:
        return iterators, counts
    from slime.backends.megatron_utils.data import DataIterator
    if len(iterators) != 1:
        raise ValueError('ARM pilot expects one iterator')
    base = iterators[0]
    data = base.rollout_data
    manifest, aux = prepared['manifest'], prepared['data']
    gbs = data.get('effective_global_batch_size', data.get('dynamic_global_batch_size', args.global_batch_size))
    base_indices = base.micro_batch_indices
    if base_indices is None:
        base_indices = [[i] for i in range(sum(counts))]
    if len(base_indices) != sum(counts) or any(len(x) != 1 for x in base_indices):
        raise ValueError('ARM pilot needs exactly one sequence per microbatch')
    # Validate action classes even though individual rows are packed separately.
    groups = {}
    for i, meta in enumerate(manifest['records']):
        if meta['source'] == 'arm_candidate':
            groups.setdefault(meta['state_id'], []).append(i)
    for indices in groups.values():
        a = [manifest['records'][i]['advantage'] for i in indices]
        if not math.isclose(sum(x for x in a if x > 0), 1.) or not math.isclose(sum(x for x in a if x < 0), -1.):
            raise ValueError('Incomplete local preference state')
    keys = ['tokens', 'multimodal_train_inputs', 'total_lengths', 'response_lengths', 'loss_masks',
            'sample_weights', 'log_probs', 'ref_log_probs', 'values', 'advantages', 'returns',
            'rollout_log_probs', 'max_seq_lens', 'teacher_log_probs', *FIELDS]
    mixed = {k: [] for k in keys}
    mixed['effective_global_batch_size'] = gbs
    cursor, new_counts = 0, []
    for window, count in enumerate(counts):
        original = [x[0] for x in base_indices[cursor:cursor + count]]
        cursor += count
        outcome_ids = {str(data['sample_indices'][i]) for i in original}
        selected = [i for i, m in enumerate(manifest['records'])
                    if m['source'] == 'rescue_demo' and m['optimizer_window'] == window]
        if len(selected) > 32:
            raise ValueError('Demo exposure exceeds 32 turns per outcome update')
        selected_groups = [indices for indices in groups.values()
                           if str(manifest['records'][indices[0]]['parent_sample_index']) in outcome_ids]
        for indices in selected_groups:
            if len({str(manifest['records'][i]['parent_sample_index']) for i in indices}) != 1:
                raise ValueError('Local state maps to multiple executed outcomes')
        for i in original:
            for key in keys:
                value = data.get(key)
                mixed[key].append(value[i] if value is not None else None)
            mixed['arm_source'][-1] = 'outcome'
            mixed['arm_scale'][-1] = 1.
            mixed['arm_advantage'][-1] = 0.
        addition = [(i, manifest['eta'] * gbs / len(selected)) for i in selected]
        addition += [(i, manifest['lambda'] * gbs / len(selected_groups))
                     for indices in selected_groups for i in indices
                     if manifest['records'][i]['advantage'] != 0]
        for i, scale in addition:
            meta = manifest['records'][i]
            for key in keys:
                values = aux.get(key)
                mixed[key].append(values[i] if values is not None else None)
            mixed['arm_source'][-1] = meta['source']
            mixed['arm_scale'][-1] = scale
            mixed['arm_advantage'][-1] = meta.get('advantage', 0.)
        new_counts.append(count + len(addition))
    # get_batch checks multimodal_train_inputs by value; optional outcome-only
    # fields remain None for auxiliary rows, whose custom branch never reads them.
    class AuxiliaryIterator(DataIterator):
        def get_next(self, keys):
            batch = super().get_next(keys)
            for key, value in batch.items():
                if isinstance(value, list) and value and all(x is None for x in value):
                    batch[key] = None
            return batch
    return [AuxiliaryIterator(mixed, micro_batch_size=1)], new_counts


def loss(args, batch, logits, reducer):
    """Called only for homogeneous microbatch=1 source rows."""
    import torch
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, policy_loss_function
    source = batch['arm_source'][0]
    if source == 'outcome':
        value, _ = policy_loss_function(args, batch, logits, reducer)
        return value, dict(arm_outcome_loss=value.detach(), arm_demo_loss=value.detach()*0,
                           arm_local_loss=value.detach()*0)
    local_args = copy(args)
    if source == 'rescue_demo':
        local_args.rollout_temperature = 1.0
    elif source != 'arm_candidate':
        raise ValueError('Unknown loss source')
    _, result = get_log_probs_and_entropy(logits, args=local_args,
        unconcat_tokens=batch['unconcat_tokens'], total_lengths=batch['total_lengths'],
        response_lengths=batch['response_lengths'], with_entropy=False,
        max_seq_lens=batch.get('max_seq_lens'))
    logp = torch.cat(result['log_probs'])
    if source == 'rescue_demo':
        value = -reducer(logp)
    else:
        old = torch.cat(batch['arm_old_log_probs']).detach()
        advantage = batch['arm_advantage'][0]
        ratio = (logp - old).exp()
        value = -reducer(torch.minimum(ratio * advantage, ratio.clamp(.8, 1.28) * advantage))
    value = value * batch['arm_scale'][0]
    zero = value.detach() * 0
    return value, dict(arm_outcome_loss=zero,
        arm_demo_loss=value.detach() if source == 'rescue_demo' else zero,
        arm_local_loss=value.detach() if source == 'arm_candidate' else zero)
