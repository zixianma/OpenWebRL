#!/usr/bin/env python3
"""Record numeric device memory during an authorized first-batch replay."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first-iteration',type=int,required=True);p.add_argument('--minutes',type=int,default=60)
    a=p.parse_args();end=time.monotonic()+min(a.minutes,60)*60
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('a') as out:
        while time.monotonic()<end:
            record=dict(epoch=time.time())
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits'],text=True,timeout=8)
                record['gpu_rows']=raw.strip().splitlines()
                path=a.root/'status.json'
                if path.exists():
                    state=json.loads(path.read_text());record['phase']=state.get('stage');record['iteration']=state.get('iteration')
            except Exception as exc:record['error_type']=type(exc).__name__
            complete=(a.root/f'iterations/{a.first_iteration-1:04d}/checkpoint-validation.json').exists()
            record['first_checkpoint_validated']=complete
            out.write(json.dumps(record)+'\n');out.flush()
            if complete:break
            time.sleep(10)


if __name__=='__main__':main()
