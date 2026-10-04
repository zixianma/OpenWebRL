"""Additive executed-turn failure PPO for TP-only OpenWebRL training.

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
    return bool(os.environ.get('OPENWEBRL_ARM_FAILURE_AUX_MANIFEST'))


def check_topology(args, *, dp=1, cp=1, pp=1):
    if (dp, cp, pp) != (1, 1, 1) or args.micro_batch_size != 1:
        raise ValueError('ARM pilot requires DP1/CP1/PP1 and microbatch 1')
    if args.calculate_per_token_loss or args.entropy_coef != 0 or args.use_kl_loss:
        raise ValueError('ARM pilot requires baseline sample-mean loss, entropy=KL=0')
    if args.loss_type != 'policy_loss' or args.advantage_estimator != 'grpo':
        raise ValueError('ARM pilot requires baseline GRPO policy loss')
    if args.rollout_temperature != .8:
        raise ValueError('The audited executed-turn old-policy temperature is .8')
    if getattr(args, 'qkv_format', 'thd') != 'thd' or getattr(args, 'use_dynamic_batch_size', False):
        raise ValueError('ARM pilot requires the baseline thd/static-microbatch packing')


def validate_auxiliary_advantage(meta, config):
    """Verify exact credit semantics before a saved auxiliary row reaches PPO."""
    from types import SimpleNamespace
    from openwebrl.arm_turn_bonus import candidate_minimum, credit_rule, unit_bonus
    from openwebrl.arm_failure_recipe import failure_beta
    beta = failure_beta(config)
    minimum, rule = candidate_minimum(config), credit_rule(config)
    label = meta.get('arm_turn_bonus')
    if label is None:
        if beta != .5 or minimum != 5 or rule != 'response_index' or meta['advantage'] not in (.4,-.1):
            raise ValueError('Missing auxiliary credit provenance')
        return  # historical five-distinct manifests
    if candidate_minimum(label) != minimum or credit_rule(label) != rule:
        raise ValueError('Auxiliary label/config credit mismatch')
    expected = beta*unit_bonus(SimpleNamespace(metadata={'arm_turn_bonus':label},remove_sample=False),config['policy_id'])
    if not expected or not math.isfinite(meta['advantage']) or meta['advantage'] != expected:
        raise ValueError('Invalid executed-turn advantage')


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
    path = Path(os.environ['OPENWEBRL_ARM_FAILURE_AUX_MANIFEST'].format(rollout_id=rollout_id))
    manifest = json.loads(path.read_text())
    config=json.loads(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']).read_text())
    if manifest['rollout_id'] != rollout_id or manifest['policy_id'] != config['policy_id'] or manifest['checkpoint'] != config['checkpoint']:
        raise ValueError('Stale additive failure manifest')
    if manifest['schema_version']!=1 or manifest['training_judge']!='gpt-4.1/action_history':
        raise ValueError('Auxiliary schema/judge mismatch')
    for key, default in [('candidate_gate','distinct5'),('credit_assignment','response_index')]:
        if manifest.get(key,default) != config.get(key,default):
            raise ValueError('Auxiliary manifest credit/gate mismatch')
    from openwebrl.arm_failure_recipe import failure_beta, coverage_budget, validate_recipe, failure_fraction, auxiliary_window_limit
    validate_recipe(config)
    if (manifest['coefficient']!=config['failure_loss_coefficient']*manifest['failure_groups']/8
            or manifest['beta']!=failure_beta(config) or manifest['q']!=failure_fraction(config)
            or manifest.get('failure_turn_budget',0)!=coverage_budget(config)
            or manifest.get('failure_ablation')!=config.get('failure_ablation')
            or manifest.get('max_auxiliary_per_window',32)!=auxiliary_window_limit(config)):
        raise ValueError('Auxiliary coefficients changed')
    tensor_path = path.parent / manifest['tensor_file']
    # A pilot artifact is bounded; never deserialize full rollout recovery here.
    from openwebrl.arm_gate_recovery import auxiliary_payload_limit
    limit = auxiliary_payload_limit(config)
    if tensor_path.stat().st_size > limit:
        raise ValueError(f'Auxiliary tensor file exceeds {limit//1024**3} GiB')
    h = hashlib.sha256()
    with tensor_path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    if h.hexdigest() != manifest['tensor_sha256']:
        raise ValueError('Auxiliary tensor hash mismatch')
    records = torch.load(tensor_path, map_location='cpu', weights_only=True, mmap=True)
    if len(records) != len(manifest['records']):
        raise ValueError('Auxiliary manifest/tensor row count mismatch')
    if not records:
        return None
    data = {k: [] for k in ['tokens', 'total_lengths', 'response_lengths', 'loss_masks', 'multimodal_train_inputs']}
    for meta, row in zip(manifest['records'], records, strict=True):
        validate_auxiliary_advantage(meta, config)
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
    # Evaluate the complete executed response under the frozen actor instead.
    iterator = [DataIterator(data, micro_batch_indices=[[i] for i in range(len(records))])]
    old = actor.compute_log_prob(iterator, [len(records)])['log_probs']
    if len(old) != len(records):
        raise ValueError('Missing executed-turn old-policy scores')
    if any(not bool(torch.isfinite(x).all()) for x in old):
        raise ValueError('Nonfinite frozen-actor log probabilities')
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
    from openwebrl.arm_failure_additive import window_schedule
    if len(counts)!=manifest['expected_windows']:raise ValueError('Outcome update count changed')
    additions=window_schedule(manifest['records'],manifest['total_failure_rows'],counts,gbs,manifest['coefficient'],manifest.get('max_auxiliary_per_window',32))
    keys = ['tokens', 'multimodal_train_inputs', 'total_lengths', 'response_lengths', 'loss_masks',
            'sample_weights', 'log_probs', 'ref_log_probs', 'values', 'advantages', 'returns',
            'rollout_log_probs', 'max_seq_lens', 'teacher_log_probs', *FIELDS]
    mixed = {k: [] for k in keys}
    mixed['effective_global_batch_size'] = gbs
    cursor, new_counts = 0, []
    for window, count in enumerate(counts):
        original = [x[0] for x in base_indices[cursor:cursor + count]]
        cursor += count
        for i in original:
            for key in keys:
                value = data.get(key)
                mixed[key].append(value[i] if value is not None else None)
            mixed['arm_source'][-1] = 'outcome'
            mixed['arm_scale'][-1] = 1.
            mixed['arm_advantage'][-1] = 0.
        addition=additions[window]
        for i, scale in addition:
            meta = manifest['records'][i]
            for key in keys:
                values = aux.get(key)
                mixed[key].append(values[i] if values is not None else None)
            mixed['arm_source'][-1] = 'arm_failure'
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
        value, metrics = policy_loss_function(args, batch, logits, reducer)
        zero=value.detach()*0
        return value,dict(arm_mixed_loss=value.detach(),arm_failure_loss=zero,
            arm_mixed_count=zero+1,arm_failure_count=zero,
            arm_mixed_kl_sum=metrics['ppo_kl'],arm_mixed_clip_sum=metrics['pg_clipfrac'],
            arm_failure_kl_sum=zero,arm_failure_clip_sum=zero)
    if source != 'arm_failure':raise ValueError('Unknown additive source')
    _, result = get_log_probs_and_entropy(logits,args=args,
        unconcat_tokens=batch['unconcat_tokens'],total_lengths=batch['total_lengths'],
        response_lengths=batch['response_lengths'],with_entropy=False,max_seq_lens=batch.get('max_seq_lens'))
    logp=torch.cat(result['log_probs']);old=torch.cat(batch['arm_old_log_probs']).detach()
    advantage=batch['arm_advantage'][0];ratio=(logp-old).exp()
    value=-reducer(torch.minimum(ratio*advantage,ratio.clamp(1-args.eps_clip,1+args.eps_clip_high)*advantage))*batch['arm_scale'][0]
    # Outcome telemetry must not pretend that auxiliary rows had terminal rewards.
    zero=value.detach()*0
    clipped=(ratio*advantage > ratio.clamp(1-args.eps_clip,1+args.eps_clip_high)*advantage).float()
    return value,dict(arm_mixed_loss=zero,arm_failure_loss=value.detach(),
        arm_mixed_count=zero,arm_failure_count=zero+1,arm_mixed_kl_sum=zero,arm_mixed_clip_sum=zero,
        arm_failure_kl_sum=reducer(old-logp).detach(),arm_failure_clip_sum=reducer(clipped).detach())


def normalize_metrics(log_dict):
    """Ratios cancel native averaging over mixed and auxiliary microbatches."""
    for source in ('mixed','failure'):
        count=log_dict.get(f'train/arm_{source}_count',0)
        if count:
            for suffix,target in [('kl_sum','ppo_kl'),('clip_sum','pg_clipfrac')]:
                log_dict[f'train/arm_{source}_{target}']=log_dict[f'train/arm_{source}_{suffix}']/count
            log_dict[f'train/arm_{source}_mean_loss']=log_dict[f'train/arm_{source}_loss']/count
    if log_dict.get('train/arm_mixed_count',0):
        log_dict['train/ppo_kl']=log_dict['train/arm_mixed_ppo_kl']
        log_dict['train/pg_clipfrac']=log_dict['train/arm_mixed_pg_clipfrac']
