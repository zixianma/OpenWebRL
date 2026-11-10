#!/usr/bin/env python3
"""Small CPU parity check against archived actor contexts; never loads weights."""
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/outcome-reward-20260927')
sys.path.insert(0, str(REPO))


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from slime.utils.processing_utils import build_processor_kwargs
    torch.set_num_threads(1)
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT', local_files_only=True)
    checks=[]
    for split in ['train', 'dev', 'future']:
        rows = [json.loads(s) for s in (ROOT/f'{split}.jsonl').read_text().splitlines()]
        selected = [min(rows, key=lambda r:r['maximum_context_tokens']), max(rows, key=lambda r:r['maximum_context_tokens'])]
        for row in selected:
            path=Path(row['source_group']); raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['source_sha256']: raise ValueError('Archive changed')
            tr=json.loads(gzip.decompress(raw))['trajectories'][row['trajectory_offset']]
            extremes=[min(tr['turns'],key=lambda s:len(s['tokens'])),max(tr['turns'],key=lambda s:len(s['tokens']))]
            turns=list({s['metadata']['turn_index']:s for s in extremes}.values())
            for turn in turns:
                images=[]
                for ref in turn['multimodal_inputs']['images']:
                    image_path=path.parent.parent/ref['archive_image']
                    if hashlib.sha256(image_path.read_bytes()).hexdigest()!=ref['sha256']: raise ValueError('Image changed')
                    with Image.open(image_path) as im: images.append(im.convert('RGB'))
                features=processor(text=[turn['prompt']], **build_processor_kwargs({'images':images}))
                n=turn['response_length']; ids=turn['tokens']; prefix=features['input_ids'][0]
                prefix_equal=prefix==ids[:-n]
                grid_equal=features['image_grid_thw'].tolist()==turn['metadata']['image_grid_thw']
                mask_valid=len(turn['loss_mask'])==n and set(turn['loss_mask'])<={0,1} and any(turn['loss_mask'])
                decoded=processor.tokenizer.decode(ids[-n:],skip_special_tokens=False)
                checks.append(dict(split=split,id=row['id'],turn=turn['metadata']['turn_index'],
                    context_tokens=len(ids),images=len(images),prefix_equal=prefix_equal,grid_equal=grid_equal,
                    mask_valid=mask_valid,response_text_equal=decoded==turn['response']))
                if not (prefix_equal and grid_equal and mask_valid):
                    break
    passed=all(c['prefix_equal'] and c['grid_equal'] and c['mask_valid'] for c in checks)
    report=dict(passed=passed,checks=checks,seconds=time.monotonic()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpus=0,api_calls=0,
        scope='Unique boundary turns from six shortest/longest-context trajectories: CPU actor processor prompt-token/grid parity and response-mask shape. No model weights or GPU kernels.',
        response_policy='Use archived executed token IDs, not re-encoded visible response text; special stop tokens may be absent from text.')
    (ROOT/'readiness-20260928/processor-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not passed: raise SystemExit(2)


if __name__=='__main__': main()
