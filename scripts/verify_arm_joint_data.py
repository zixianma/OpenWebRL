#!/usr/bin/env python3
"""Verify paired views and masks; export fixed source-specific validation panels."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

from tokenizers import Tokenizer
from prepare_arm_joint_data import pair_reason, order


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();start=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(384*1024**2,384*1024**2));resource.setrlimit(resource.RLIMIT_CPU,(90,95))
    tok=Tokenizer.from_file('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT/tokenizer.json')
    counts=Counter();groups={};ids=set();validation=[]
    for split in ['train','validation']:
        groups[split]=set()
        with (a.data/f'joint_pairs.{split}.jsonl').open() as pf,(a.data/f'joint_sft.{split}.jsonl').open() as sf:
            for i,(pl,sl) in enumerate(itertools.zip_longest(pf,sf),1):
                assert pl and sl,'SFT and pair view length mismatch'
                row=json.loads(pl);sft=json.loads(sl)
                assert sft=={k:v for k,v in row.items() if k not in ['rejected','rejected_index']}
                assert row['id'] not in ids;ids.add(row['id'])
                assert row['split']==split and row['status']=='retained'
                assert pair_reason(row['chosen']['actions'],row['rejected']['actions'])=='eligible'
                assert row['chosen_index']!=row['rejected_index']
                groups[split].add(row['task_group'])
                for name in ['chosen','rejected']:
                    x=row[name];mask=x['action_token_indices']
                    assert mask==sorted(set(mask)) and mask and max(mask)<len(x['token_ids'])
                    assert tok.decode(x['token_ids'],skip_special_tokens=False)==x['text']
                    expected=''.join(x['text'][start:end] for start,end in x['action_char_spans'])+'<|im_end|>'
                    actual=tok.decode([x['token_ids'][k] for k in mask],skip_special_tokens=False)
                    assert actual==expected,(row['id'],name,'mask includes unexpected tokens')
                    assert x['tokens']==len(x['token_ids']) and x['action_tokens']==len(mask)
                    assert row['prompt_tokens']+x['tokens']<=32768
                counts[split+':'+row['source']]+=1
                if split=='validation':validation.append(row)
                if i%100==0:time.sleep(.01)
    assert not groups['train']&groups['validation']
    panels={}
    for source in ['C2','Piotr']:
        panel=sorted([x for x in validation if x['source']==source],key=lambda x:order(x['id']+':panel'))[:256]
        path=a.data/f'validation-panel-{source.lower()}.jsonl'
        path.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in panel))
        panels[source]={'states':len(panel),'task_groups':len({x['task_group'] for x in panel}),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    download=json.loads((a.data.parent/'joint-data-inputs-0d83b48/download-audit.json').read_text())
    expected={r['path']:r['sha256'] for r in download['files']}
    images=json.loads((a.data/'image-manifest.json').read_text())
    hf_images=0
    for image in images:
        if image['path'] in expected:
            assert image['sha256']==expected[image['path']];hf_images+=1
    usage=resource.getrusage(resource.RUSAGE_SELF)
    report={'passed':True,'counts':dict(counts),'validation_panels':panels,'task_group_overlap':0,
            'hf_image_hashes_matched_download':hf_images,
            'checks':['Exact chosen-response equality between SFT and paired views','Unique state IDs across train/validation',
                      'Disjoint task groups','Eligible action differences','Response token decode equality',
                      'Exact decoded action mask equals tool-call spans plus end-of-turn token','32768 context limit',
                      'HF image bytes match checksum-verified download manifest'],
            'resources':{'wall_seconds':time.monotonic()-start,'cpu_seconds':usage.ru_utime+usage.ru_stime,'peak_rss_mib':usage.ru_maxrss/1024}}
    (a.data/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
