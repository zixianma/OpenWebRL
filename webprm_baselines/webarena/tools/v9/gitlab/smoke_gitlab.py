"""CPU-only live smoke of the v9 GitLab cell path (no GPU, no model/provider calls).

Runs the package's generated stage functions directly (build -> copy -> namespace-inside serve), then, inside the
same owned user+net namespace: readiness, the adapter's released login, and a native-browser-v3 observation of
three released GitLab task start URLs with the login storage state (1024x2490 compatibility geometry), including a
`new_tab` + `goto` step. Then stop.json -> recorded controller shutdown -> owned closure.
The campaign context/approval checks of the real stage entry point are bypassed here (smoke only; no cell result).
Usage: smoke_gitlab.py PACKAGE OUT        (outer, inside a CPU Slurm job)
       smoke_gitlab.py --inside CONTEXT   (namespace child)
"""
import asyncio, json, os, sys, threading, time
from pathlib import Path

PY = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python'
D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')


def setup(package):
    sys.path.insert(0, str(Path(package)/'source')); os.environ['WEBPRM_NATIVE_SOURCE_ROOT'] = str(Path(package)/'source')
    from scripts import run_webprm_table5_gitlab as g
    return g


def outer(package, out):
    g = setup(package)
    from openwebrl.webprm_lifecycle import _process
    out = Path(out); out.mkdir(mode=0o700, parents=True, exist_ok=False)
    job = os.environ['SLURM_JOB_ID']; scratch = Path('/tmp')/('gsmoke-'+job); scratch.mkdir(mode=0o700)
    (scratch/'states').mkdir(mode=0o700)
    owned = dict(proposal=json.loads((Path(package)/'proposal.json').read_text()), scratch=str(scratch), output=str(out),
                 deadline=time.time()+5400, job_id=job, owner_pid=os.getpid(), owner_start_ticks=_process(os.getpid())['start_ticks'])
    timing = {}; t0 = time.time()
    g.build(owned, scratch/'baseline', out/'original-build'); timing['build'] = time.time()-t0
    (out/'cells/smoke').mkdir(mode=0o700, parents=True)
    state = scratch/'states'/'smoke'; t0 = time.time()
    g.copy(owned, state, out/'cells/smoke/copy'); timing['copy'] = time.time()-t0
    ctx = out/'smoke-context.json'
    ctx.write_text(json.dumps(dict(owned=owned, state=str(state), site=str(out/'cells/smoke/site'), package=str(package),
                                   controls=str(g.controls_path(owned, state)), timing=timing)))
    import subprocess
    t0 = time.time()
    r = subprocess.run(['unshare', '--user', '--map-root-user', '--net', PY, '-B', __file__, '--inside', str(ctx)],
                       stdout=open(out/'inside.log', 'wb'), stderr=subprocess.STDOUT, timeout=4000)
    timing['inside'] = time.time()-t0
    result = json.loads((out/'smoke-result.json').read_text()) if (out/'smoke-result.json').exists() else dict(status='no_result')
    result.update(inside_returncode=r.returncode, timing=dict(timing, **result.get('timing', {})))
    (out/'smoke-summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1)[:3000])


def probe(c, result_path):
    """Forked before the service owner starts; checks the live instance, then requests stop."""
    from openwebrl import webprm_gitlab_site as site
    out = Path(c['site']); owned = c['owned']; result = dict(status='failed', timing={}); t0 = time.time()
    try:
        while not (out/'instance.json').exists():
            if time.time()-t0 > 1500: raise RuntimeError('instance timeout')
            if (out/'service-closure.json').exists(): raise RuntimeError('service closed before instance')
            time.sleep(2)
        result['timing']['instance_ready'] = time.time()-t0
        instance = json.loads((out/'instance.json').read_text())
        site.website_ready()
        site.validate_instance(instance)
        native_source = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter'
        chromium = '/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome'
        auth = out.parent/'auth'; auth.mkdir(mode=0o700)
        t1 = time.time(); site.authenticate(native_source, instance, chromium, auth); result['timing']['login'] = time.time()-t1
        result['authentication'] = json.loads((auth/'authentication.json').read_text())
        sys.path.insert(0, str(D/'native-browser-v3'))
        from native_episode import NativeBrowser, load_native, stage_chromium
        native = load_native(); staged = stage_chromium(cache_root=Path(owned['scratch'])/'chromium-cache')
        tasks = json.loads((Path(native_source)/'viewer/config_files/wa/test_webarena_lite.raw.json').read_text())
        observed = []
        for tid in (9, 112, 130):
            task = dict(tasks[tid]); task['start_url'] = task['start_url'].replace('__GITLAB__', site.ORIGIN)
            task['storage_state'] = str(auth/'storage-state-private.json')
            cfg = auth/f'task-{tid}.json'; cfg.write_text(json.dumps(task))
            browser = NativeBrowser(native, auth/f'browser-{tid}', viewport=dict(width=1024, height=2490), compatibility_geometry=True,
                                    allowed_origins=[site.ORIGIN], chromium=staged['executable'], launch_parent=Path(owned['scratch']))
            async def go():
                try:
                    rows = [await browser.start(cfg)]
                    if tid == 9:
                        rows.append((await browser.execute('new_tab'))['observation'])
                        rows.append((await browser.execute(f'goto [{site.ORIGIN}/byteblaze]'))['observation'])
                    return rows
                finally:
                    await browser.close()
            t1 = time.time(); rows = asyncio.run(go())
            observed.append(dict(task=tid, start_url=task['start_url'], steps=len(rows), seconds=round(time.time()-t1, 1),
                                 text_chars=[len(r['text']) for r in rows], sign_in_prompt=('Sign in' in rows[0]['text'][:4000]),
                                 byteblaze=('byteblaze' in rows[-1]['text']), geometry=rows[-1].get('geometry'),
                                 lazy_page_clients=browser.lazy_page_clients))
        result.update(status='passed', observations=observed, instance_id=instance['instance_id'])
    except BaseException as e:
        result['error'] = repr(e)[:2000]
    finally:
        Path(result_path).write_text(json.dumps(result, indent=1))
        (out/'stop.json').write_text(json.dumps(dict(reason='smoke closed')))


def inside(ctx_path):
    import multiprocessing
    c = json.loads(Path(ctx_path).read_text()); g = setup(c['package'])
    owned, state, out = c['owned'], Path(c['state']), Path(c['site'])
    out.mkdir(mode=0o700)
    g.validate_handoff = lambda *a, **k: dict(controls=c['controls'])  # smoke only: no campaign launch handoff
    probe_path = out.parent/'probe.json'
    p = multiprocessing.get_context('fork').Process(target=probe, args=(c, str(probe_path))); p.start()
    errors = []
    try: g.inside(owned, str(state), out, None)
    except BaseException as e: errors.append(repr(e)[:2000])
    p.join(timeout=600)
    result = json.loads(probe_path.read_text()) if probe_path.exists() else dict(status='failed', error='no probe result')
    logs = state/'var/log/webprm-probe'
    result['controller_log_tails'] = {x.name: x.read_text(errors='replace')[-1500:] for x in sorted(logs.iterdir()) if x.is_file() and x.stat().st_size < 2**22} if logs.is_dir() else None
    closure = json.loads((out/'service-closure.json').read_text()) if (out/'service-closure.json').exists() else None
    result.update(serve_errors=errors, service_closure=closure)
    if errors or closure is None or closure.get('remaining_owned_processes') or not closure.get('controller_stopped'):
        result['status'] = 'failed'
    (out.parent.parent.parent/'smoke-result.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == '--inside':
        inside(sys.argv[2])
    else:
        outer(sys.argv[1], sys.argv[2])
