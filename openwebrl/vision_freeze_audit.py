"""Fail closed if the frozen-backbone RL ablation freezes the wrong tensors."""
import json
import os
from pathlib import Path
import re


FREEZE_PATTERN = r'^vision_model\.(?!(?:merger|deepstack_merger_list)\.)'


def component(name):
    if name.startswith(('vision_model.merger.', 'vision_model.deepstack_merger_list.')):
        return 'projection'
    if name.startswith('vision_model.'):
        return 'backbone'
    if name.startswith('language_model.'):
        return 'language'
    raise ValueError(f'Unclassified model parameter: {name}')


def audit(model, args):
    """Run on the unwrapped TP-local model, before distributed optimizer setup."""
    if args.freeze_params_name_list != [FREEZE_PATTERN] or args.only_train_params_name_list:
        raise ValueError('Frozen vision requires the exact backbone-only mask')
    counts = {k: dict(tensors=0, parameters=0, trainable_parameters=0)
              for k in ('backbone', 'projection', 'language')}
    names = []
    for name, param in model.named_parameters():
        group = component(name)
        expected = group != 'backbone'
        if bool(param.requires_grad) != expected:
            raise ValueError(f'Unexpected requires_grad={param.requires_grad}: {name}')
        if bool(re.search(FREEZE_PATTERN, name)) != (group == 'backbone'):
            raise ValueError(f'Freeze mask does not match component: {name}')
        c = counts[group]
        c['tensors'] += 1
        c['parameters'] += param.numel()
        c['trainable_parameters'] += param.numel() if expected else 0
        names.append(dict(name=name, shape=list(param.shape), trainable=expected))
    if not all(c['tensors'] for c in counts.values()):
        raise ValueError('Missing backbone, projection or language parameters')
    result = dict(passed=True, counts=counts, parameters=names,
                  scope='TP-local counts; replicated vision must not be summed across ranks')
    destination = os.getenv('OPENWEBRL_VISION_FREEZE_AUDIT_DIR')
    if destination:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path/f'rank-{rank}-pid-{os.getpid()}.json').write_text(json.dumps(result, indent=2)+'\n')
    print('FROZEN_VISION_PARAMETER_AUDIT_OK '+json.dumps(counts), flush=True)
    return result
