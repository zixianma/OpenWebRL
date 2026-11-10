#!/usr/bin/env python3
"""CPU-only initial-page checks. Page availability never establishes task success."""
import argparse
import asyncio
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
POOL = RUNTIME/'arm-turn-bonus-preparation/task-pool-expansion-20260922/curation-v3-20260929'
ROOT = POOL/'live-browser-20260929'
INPUT = POOL/'semantic-qwen8b/kept-337317.jsonl'
SOURCE = RUNTIME/'reference-arm-mixed-reweight-replay-20260928-v1'
SUPERVISOR = RUNTIME/'arm-turn-bonus-preparation/mixed-reweight-20260927/supervisor'
RESOURCES = dict(gpus=0, cpus=8, memory_gib=32, hours=2, browser_workers=8)
SOURCE_FILES = ('openwebrl/env/web_env.py', 'openwebrl/env/base_env.py',
                'openwebrl/env/browser_runtime.py', 'openwebrl/env/config.yaml')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def choose_urls(by_url, insta_count=124):
    pae = {url for url, item in by_url.items() if item['sources'].get('pae-webvoyager')}
    insta = [url for url, item in by_url.items()
             if item['sources'].get('insta-v3') and url not in pae]
    rank = lambda u: hashlib.sha256(('arm-live-urls-20260929\n'+u).encode()).hexdigest()
    return sorted(pae)+sorted(insta, key=rank)[:insta_count]


def prepare():
    from rebuild_arm_task_pool import load_upstream
    upstream, upstream_manifest = load_upstream(POOL/'upstream')
    by_url = {}
    for line in INPUT.open():
        row = json.loads(line)
        url = upstream.normalize_start_url(row['website'])
        item = by_url.setdefault(url, dict(sources=Counter(), task_ids=[], examples=[]))
        item['sources'][row['benchmark_name']] += 1
        item['task_ids'].append(str(row['task_id']))
        if len(item['examples']) < 2:
            item['examples'].append(dict(task_id=str(row['task_id']), task=row['task_name']))
    urls = choose_urls(by_url)
    probes = [dict(probe_id=hashlib.sha256(u.encode()).hexdigest()[:24], url=u,
                   selection='all_pae_start_urls' if by_url[u]['sources'].get('pae-webvoyager')
                   else 'uniform_hash_sample_insta_urls', **by_url[u]) for u in urls]
    manifest = dict(schema=1, source=str(SOURCE), source_hashes={n:sha(SOURCE/n) for n in SOURCE_FILES},
        input=str(INPUT), input_sha256=sha(INPUT), upstream_commit=upstream_manifest['commit'],
        resources=RESOURCES, probes=probes, total_candidate_tasks=sum(len(x['task_ids']) for x in by_url.values()),
        total_unique_start_urls=len(by_url), covered_candidate_tasks=sum(len(x['task_ids']) for x in probes),
        coverage_by_source=dict(sum((Counter(x['sources']) for x in probes), Counter())),
        interpretation='Initial-page availability only; shared URL evidence does not verify individual task targets or solvability.',
        policy=dict(manual_review_required=False, api_calls=0, stealth_sessions=0,
                    maximum_attempts_per_url=2, worker_hard_timeout_seconds=75,
                    navigation_timeout_ms=20000, screenshot_timeout_ms=10000,
                    no_clicks_or_form_submissions=True, fresh_browser_per_attempt=True,
                    unavailable_scope='Exact start URL in this browser at collection time; no automatic host blacklist.'))
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/'manifest.json'
    if path.exists() and read(path) != manifest:
        raise ValueError('Frozen pilot differs; use a new version rather than replacing it')
    write(path, manifest)
    write(ROOT/'request.json', dict(status='prepared_awaiting_exact_cpu_allocation_approval',
        resources=RESOURCES, max_cpu_hours=16, urls=len(probes), maximum_browser_attempts=len(probes)*2,
        expected_runtime='Roughly15–40min;2h cap includes retries/startup, release on completion.',
        no_gpu_or_model_api=True, manifest_sha256=sha(path), submission_script='scripts/check_arm_task_urls_cpu.sbatch'))
    return {k:manifest[k] for k in ['total_candidate_tasks','total_unique_start_urls','covered_candidate_tasks','coverage_by_source']} | dict(urls=len(probes))


def classify(row):
    """Conservative machine triage; uncertain pages remain eligible for diagnosis."""
    if row.get('worker_error') or row.get('setup_error'):
        return 'inconclusive', 'browser_infrastructure'
    status = row.get('http_status')
    title = (row.get('title') or '').lower().strip()
    text = (row.get('text') or '').strip()
    prefix = (title+'\n'+text[:1500]).lower()
    if any(x in title for x in ('just a moment', 'checking your browser', 'security verification')) or (
            len(text) < 2500 and any(x in prefix for x in ('verify you are human', 'verify that you are human', 'performing security verification'))):
        return 'inconclusive', 'browser_challenge'
    if status in (401, 403, 429):
        return 'inconclusive', f'http_{status}_access_or_rate_limit'
    if status in (404, 410):
        return 'unavailable', f'http_{status}'
    if status is not None and status >= 500:
        return 'unavailable', 'server_error'
    if len(text) < 1500 and any(x in title for x in ('domain for sale', 'website for sale', 'buy this domain')):
        return 'unavailable', 'suspected_parked_domain'
    if len(text) < 1000 and title in ('404', '404 not found', 'page not found', '410 gone'):
        return 'unavailable', 'missing_page'
    if row.get('navigation_error'):
        # A timeout can leave a useful rendered page; preserve the evidence.
        if len(text) >= 100 or row.get('interactive_count', 0) >= 3:
            return 'inconclusive', 'navigation_error_with_content'
        return 'unavailable', 'navigation_error'
    if row.get('password_fields', 0):
        return 'inconclusive', 'authentication_form_present'
    if row.get('evidence_error') or not row.get('screenshot_bytes') or not row.get('dom_bytes'):
        return 'inconclusive', 'incomplete_page_evidence'
    if len(text) < 100 and row.get('interactive_count', 0) < 3:
        return 'inconclusive', 'sparse_or_loading_page'
    return 'page_available', 'rendered_content'


def decision(attempts):
    if any(a['classification']=='page_available' for a in attempts):
        return 'page_available'
    if len(attempts)==2 and all(a['classification']=='unavailable' for a in attempts):
        return 'unavailable_after_retry'
    return 'inconclusive'


def approved_seconds_left():
    approval=read(ROOT/'approval.json');job=os.environ.get('SLURM_JOB_ID')
    ids=approval.get('job_ids',[])
    if not (approval.get('approved') and approval.get('resources')==RESOURCES and job in ids):
        raise ValueError('Exact CPU allocation approval and registered attempt required')
    if any(not str(x).isdecimal() for x in ids):
        raise ValueError('Expected numeric approved job IDs')
    with (ROOT/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
            '--format=JobID,State,ElapsedRaw,ExitCode'],text=True)
        attempts=[dict(job_id=f[0],state=f[1],elapsed_seconds=int(f[2]),exit_code=f[3])
                  for line in raw.splitlines() if (f:=line.split('|'))[0] in ids]
        if {a['job_id'] for a in attempts}!=set(ids):
            raise ValueError('Missing accounting for an approved attempt')
        used=sum(a['elapsed_seconds'] for a in attempts)
        write(ROOT/'attempts.json',dict(approved_seconds=7200,consumed_seconds=used,
            remaining_seconds=max(0,7200-used),attempts=attempts,checked_epoch=time.time()))
    return max(0,7200-used)


async def worker(manifest_path, probe_id, output):
    manifest = read(manifest_path)
    source = Path(manifest['source'])
    for name, expected in manifest['source_hashes'].items():
        if sha(source/name) != expected:
            raise ValueError('Frozen browser source changed')
    sys.path.insert(0, str(source))
    import yaml
    from openwebrl.env.web_env import WebEnv
    cfg = yaml.safe_load((source/'openwebrl/env/config.yaml').read_text())
    fields = ('width','height','dpr','max_retries','wait_timeout','screenshot_timeout',
              'resize_output_coords','resize_scale','image_patch_size')
    kwargs = {k:cfg[k] for k in fields}
    kwargs.update(max_retries=1, screenshot_timeout=10000)
    probe = next(p for p in manifest['probes'] if p['probe_id']==probe_id)
    if urlsplit(probe['url']).scheme not in ('http','https') or not urlsplit(probe['url']).hostname:
        raise ValueError('Expected a public HTTP(S) task start URL')
    env = WebEnv(**kwargs, start_url='about:blank', tool_list=None, policy=None)
    row = dict(probe_id=probe_id, requested_url=probe['url'], started_epoch=time.time())
    output.mkdir(parents=True, exist_ok=True)
    try:
        try:
            await env.setup()
        except Exception as exc:
            row['setup_error']=str(exc)[:1000]
        if not row.get('setup_error'):
            try:
                response = await env.page.goto(probe['url'], wait_until='load', timeout=20000)
                row['http_status'] = response.status if response else None
            except Exception as exc:
                row['navigation_error']=str(exc)[:1000]
            await asyncio.sleep(2)
            row['final_url']=env.page.url
            try:
                row['title']=await asyncio.wait_for(env.page.title(), 3)
                row['text']=await env.page.inner_text('body', timeout=3000)
                counts=await asyncio.wait_for(env.page.evaluate('''() => ({
                  interactive_count: [...document.querySelectorAll('a,button,input,select,textarea')].filter(e=>e.getBoundingClientRect().width>0&&e.getBoundingClientRect().height>0).length,
                  password_fields: [...document.querySelectorAll('input[type=password]')].filter(e=>e.getBoundingClientRect().width>0&&e.getBoundingClientRect().height>0).length
                })'''), 3)
                row.update(counts)
                dom=await asyncio.wait_for(env.page.content(), 5)
                (output/'page.html').write_text(dom[:2000000])
                row['dom_bytes']=(output/'page.html').stat().st_size
                row['dom_truncated']=len(dom)>2000000
                (output/'page.txt').write_text(row['text'][:200000])
                screenshot=await env.get_screenshot()
                (output/'screenshot.png').write_bytes(screenshot)
                row['screenshot_bytes']=len(screenshot)
            except Exception as exc:
                row['evidence_error']=str(exc)[:1000]
    finally:
        try:
            await asyncio.wait_for(env.exit(), 8)
        except Exception as exc:
            row['cleanup_error']=str(exc)[:500]
        row['classification'],row['reason']=classify(row)
        row['text']=row.get('text','')[:8000]
        row['finished_epoch']=time.time()
        write(output/'result.json',row)


async def attempt(manifest_path, probe, directory):
    directory.mkdir(parents=True, exist_ok=True)
    env=dict(os.environ, PLAYWRIGHT_BROWSERS_PATH=str(RUNTIME/'browsers'),
             OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (directory/'worker.log').open('w') as log:
        proc=await asyncio.create_subprocess_exec(sys.executable,'-B',str(Path(__file__).resolve()),
            '--worker','--manifest',str(manifest_path),'--probe-id',probe['probe_id'],
            '--output',str(directory),env=env,stdout=log,stderr=log,start_new_session=True)
        try:
            await asyncio.wait_for(proc.wait(),75)
        except asyncio.TimeoutError:
            pass
        finally:
            # Terminate only this attempt's owned process group, including
            # Chromium children, even if the worker exited before cleaning up.
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            if proc.returncode is None:
                try: await asyncio.wait_for(proc.wait(),3)
                except asyncio.TimeoutError:
                    try: os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    await proc.wait()
    path=directory/'result.json'
    if path.exists():
        row=read(path)
    else:
        row=dict(probe_id=probe['probe_id'],requested_url=probe['url'],worker_error='worker_timeout_or_error',
                 classification='inconclusive',reason='browser_infrastructure',returncode=proc.returncode)
        write(path,row)
    return row


async def execute(manifest_path, output, smoke=False):
    manifest=read(manifest_path)
    if smoke:
        # Bounded sequential live smoke; no model/API requests or paid session.
        probes=([p for p in manifest['probes'] if p['selection']=='all_pae_start_urls'][:1]+
                [p for p in manifest['probes'] if p['selection']=='uniform_hash_sample_insta_urls'][:2])
        concurrency=1
    else:
        approval=read(ROOT/'approval.json');job=os.environ.get('SLURM_JOB_ID')
        if not (approval.get('approved') and approval.get('resources')==RESOURCES and
                job in approval.get('job_ids',[]) and approval.get('manifest_sha256')==sha(manifest_path)):
            raise ValueError('Exact CPU allocation approval, registered job and frozen manifest required')
        remaining=approved_seconds_left()-120
        if remaining<180:
            raise ValueError('Insufficient unused approved time for a safe attempt')
        live=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        fields=dict(x.split('=',1) for x in live.split() if '=' in x)
        if int(fields['NumCPUs'])!=8 or 'gres/gpu' in fields.get('AllocTRES',''):
            raise ValueError('Live resources differ from the approved CPU-only request')
        probes=manifest['probes'];concurrency=RESOURCES['browser_workers']
    output.mkdir(parents=True,exist_ok=True)
    with (output/'active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signature=sha(manifest_path)
        if (output/'run.json').exists() and read(output/'run.json')['manifest_sha256']!=signature:
            raise ValueError('Refuse resume with a different manifest')
        write(output/'run.json',dict(manifest_sha256=signature,source=manifest['source'],smoke=smoke,
              probes=[p['probe_id'] for p in probes],concurrency=concurrency))
        if not smoke:
            with (SUPERVISOR/'registry-update.lock').open('a') as registry_lock:
                fcntl.flock(registry_lock,fcntl.LOCK_EX)
                registry=read(SUPERVISOR/'registry.json')
                key='task-pool-live-browser-200'
                existing=next((x for x in registry['jobs'] if x['key']==key),None)
                if existing is None:
                    existing=dict(key=key);registry['jobs'].append(existing)
                existing.update(job_id=job,controller_root=str(output),training_root=str(output),
                    evaluations=[],approved_total_seconds=7200,budget_receipt=str(ROOT/'approval.json'),
                    verified_complete=False,requires_user=False,require_agent_completion_review=True,
                    agent_completion_reviewed=False,requested_endpoint='200 exact-start-URL evidence/decisions; initial-page availability only')
                write(SUPERVISOR/'registry.json',registry)
        done=[];gate=asyncio.Semaphore(concurrency)
        async def one(probe):
            async with gate:
                directory=output/probe['probe_id'];path=directory/'decision.json'
                if path.exists():
                    row=read(path)
                else:
                    attempts=[]
                    for i in range(1,3):
                        saved=directory/f'attempt{i}'/'result.json'
                        if i==2 and not saved.exists():await asyncio.sleep(2)
                        row=read(saved) if saved.exists() else await attempt(manifest_path,probe,directory/f'attempt{i}')
                        attempts.append(row)
                        if row['classification']=='page_available':break
                    row=dict(probe_id=probe['probe_id'],url=probe['url'],decision=decision(attempts),
                        attempts=len(attempts),reasons=[a['reason'] for a in attempts],
                        covered_candidate_tasks=len(probe['task_ids']),sources=probe['sources'])
                    write(path,row)
                done.append(row)
                write(output/'status.json',dict(complete=False,stage='collection',completed_urls=len(done),
                      expected_urls=len(probes),counts=dict(Counter(x['decision'] for x in done)),updated_epoch=time.time()))
                progress=output/'runtime/progress.log';progress.parent.mkdir(exist_ok=True)
                with progress.open('a') as f:f.write(f'{time.time()} completed_urls={len(done)}/{len(probes)}\n')
        try:
            await asyncio.wait_for(asyncio.gather(*(one(p) for p in probes)),
                                   timeout=500 if smoke else remaining)
        except BaseException as exc:
            write(output/'status.json',dict(complete=False,stage='failed',error=type(exc).__name__,
                  detail=str(exc)[:500],completed_urls=len(done),expected_urls=len(probes),updated_epoch=time.time()))
            raise
        finally:
            if not smoke:approved_seconds_left()
        result=dict(complete=True,returncode=0,stage='artifacts_ready',urls=len(done),
            counts=dict(Counter(x['decision'] for x in done)),attempts=sum(x['attempts'] for x in done),
            interpretation=manifest['interpretation'],smoke=smoke,updated_epoch=time.time())
        write(output/'status.json',result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    for flag in ['prepare','execute','smoke','worker']:group.add_argument('--'+flag,action='store_true')
    p.add_argument('--manifest',type=Path,default=ROOT/'manifest.json')
    p.add_argument('--output',type=Path,default=ROOT/'run')
    p.add_argument('--probe-id');args=p.parse_args()
    if args.prepare:print(json.dumps(prepare(),indent=2))
    elif args.worker:asyncio.run(worker(args.manifest,args.probe_id,args.output))
    else:print(json.dumps(asyncio.run(execute(args.manifest,args.output,smoke=args.smoke)),indent=2))
