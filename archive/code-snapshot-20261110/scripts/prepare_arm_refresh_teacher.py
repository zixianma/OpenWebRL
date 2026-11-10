#!/usr/bin/env python3
"""Build private GPT-5.5 labeling requests and a conservative cost reservation.

This command is CPU-only and never uploads files, calls an API or submits Slurm.
"""
import argparse
import json
import math
from pathlib import Path

from prepare_arm_forward_transfer import CONTROL, SOURCE, rows, sha, stable, write
from openwebrl.arm_inference import load_selection_builder, selection_messages

MODEL = 'gpt-5.5'
OUTPUT_CAP = 2048
CHUNK_BYTES = 180_000_000  # Provider limit is 200 MB, including inline images.


def cost_reservation(messages, size, batch=True):
    # A UTF-8 byte ceiling avoids assuming GPT-5.5 uses the student's tokenizer.
    texts=[]
    for message in messages:
        content=message['content']
        texts.extend([content] if isinstance(content,str) else
                     [p['text'] for p in content if p['type']=='text'])
    text_ceiling=sum(len(t.encode('utf-8')) for t in texts)+256
    # GPT-5.5 auto/original: <=10K 32px patches, multiplier1.2. No enlargement.
    image_ceiling=math.ceil(min(10000,math.ceil(size[0]/32)*math.ceil(size[1]/32))*1.2)+256
    factor=.5 if batch else 1.
    return dict(input_tokens_reserved=text_ceiling+image_ceiling,
                max_completion_tokens=OUTPUT_CAP,
                usd_reserved=factor*((text_ceiling+image_ceiling)*5+OUTPUT_CAP*30)/1e6)


def prepare(data,output):
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['passed'] is not True:raise ValueError('Data preparation did not pass')
    for name,digest in manifest['data_hashes'].items():
        if sha((data/name).read_bytes())!=digest:raise ValueError('Data changed')
    output.mkdir(parents=True,exist_ok=False)
    builder=load_selection_builder(SOURCE)
    states=sum([list(rows(data/f'{s}.jsonl')) for s in ['train','dev','future']],[])
    # Preflight spans splits/diversity and is also reused in the final label set.
    states=sorted(states,key=lambda r:stable('pilot:'+r['id']))
    passes=[(r,'main',r['permutation']) for r in states]
    future=sorted([r for r in states if r['split']=='future'],key=lambda r:stable('order:'+r['id']))[:100]
    passes.extend((r,'reverse',list(reversed(r['permutation']))) for r in future)
    index=[];chunk=None;chunk_bytes=0;chunk_number=0
    for i,(row,treatment,order) in enumerate(passes):
        raw=Path(row['screenshot']).read_bytes()
        if sha(raw)!=row['screenshot_sha256']:raise ValueError('Screenshot changed')
        messages=selection_messages(builder,row['task'],row['url'],row['history'],
                                    [row['candidates'][j] for j in order],raw)
        pilot=i<20
        request=dict(custom_id='refresh-'+sha((row['id']+':'+treatment).encode())[:32],
                     method='POST',url='/v1/chat/completions',
                     body=dict(model=MODEL,messages=messages,max_completion_tokens=OUTPUT_CAP))
        encoded=(json.dumps(request,ensure_ascii=False)+'\n').encode()
        if len(encoded)>=CHUNK_BYTES:raise ValueError('One request exceeds chunk size')
        if pilot:
            dest=output/'pilot.jsonl'
        else:
            if chunk is None or chunk_bytes+len(encoded)>CHUNK_BYTES:
                chunk_number+=1;chunk=output/f'batch-{chunk_number:03d}.jsonl';chunk_bytes=0
            dest=chunk;chunk_bytes+=len(encoded)
        with dest.open('ab') as f:f.write(encoded)
        index.append(dict(custom_id=request['custom_id'],state_id=row['id'],split=row['split'],
                          treatment=treatment,permutation=order,file=dest.name,pilot=pilot,
                          body_sha256=sha(json.dumps(request['body'],sort_keys=True).encode()),
                          **cost_reservation(messages,row['image_size'],batch=not pilot)))
    (output/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
    summary=dict(model=MODEL,requests=len(index),pilot_requests=20,batch_requests=len(index)-20,
                 data=str(data),data_manifest_sha256=sha((data/'manifest.json').read_bytes()),
                 max_completion_tokens=OUTPUT_CAP,reasoning_effort='provider default (Piotr script omits it)',
                 temperature='provider default (Piotr script omits it)',
                 cost_ceiling_usd=sum(r['usd_reserved'] for r in index),
                 input_tokens_reserved=sum(r['input_tokens_reserved'] for r in index),
                 pricing=dict(standard_input_per_million=5,standard_output_per_million=30,batch_discount=.5),
                 files={p.name:dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in output.glob('*.jsonl')},
                 submitted=False,
                 notes=['No original replay/retention labels are repurchased.',
                        '20 standard preflight calls, then Batch; no GPU waits for labeling.',
                        '100 future states have a reversed-order teacher stability audit.',
                        'Preserve raw responses, model snapshot, usage, errors and custom IDs.',
                        'Do not infer an action winner from truncation, refusal or request failure.'])
    write(output/'manifest.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=CONTROL/'data-v2')
    p.add_argument('--output',type=Path,default=CONTROL/'teacher-v1')
    a=p.parse_args();prepare(a.data,a.output)
