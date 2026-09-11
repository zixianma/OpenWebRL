#!/usr/bin/env python3
"""Inventory raw candidate IDs before joining teacher labels; no model loads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(384*1024**2,384*1024**2));resource.setrlimit(resource.RLIMIT_CPU,(60,65))
    seen={};duplicates=[];temperatures=Counter();start=time.monotonic()
    source=a.inputs/'openwebrl_actor/candidates_merged.jsonl';file_hash=hashlib.sha256()
    with source.open('rb') as f:
        for n,line in enumerate(f,1):
            file_hash.update(line);r=json.loads(line);sid=r['state_id']
            digest=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            temperatures[str(r.get('temperature'))]+=1
            if sid in seen:duplicates.append({'id':sid,'line':n,'identical':seen[sid]==digest})
            else:seen[sid]=digest
            if n%1000==0:time.sleep(.01)
    report={'rows':n,'unique_draw_ids':len(seen),'duplicate_records':len(duplicates),
            'conflicting_duplicate_records':sum(not x['identical'] for x in duplicates),'duplicates':duplicates,
            'temperatures':dict(temperatures),'candidate_file_sha256':file_hash.hexdigest(),
            'wall_seconds':time.monotonic()-start,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime}
    (a.inputs/'candidate-identity-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='duplicates'},indent=2))


if __name__=='__main__':main()
