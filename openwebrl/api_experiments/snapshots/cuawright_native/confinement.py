"""Inherited native-agent write confinement and physical browser reservations."""
import ctypes
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request


def restrict(output, temporary, chromium, read_scope=True, private_proc=False):
    libc = ctypes.CDLL(None, use_errno=True)
    def call(number, *args):
        value = libc.syscall(number, *args)
        if value < 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        return value
    abi = call(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError('Landlock ABI >= 3 required')
    handled = 1 | sum(1 << n for n in range(1, 15) if n not in (2, 3))
    if read_scope:
        handled |= (1 << 2) | (1 << 3)
    class Rules(ctypes.Structure):
        _fields_ = [('access', ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('access', ctypes.c_uint64), ('parent', ctypes.c_int32)]
    attr = Rules(handled)
    fd = call(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    def allow(path, access):
        path = Path(path).resolve()
        if not path.exists():
            return
        parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = Beneath(access, parent)
            call(445, fd, 1, ctypes.byref(rule), 0)
        finally:
            os.close(parent)
    if read_scope:
        read=(1 << 2) | (1 << 3)
        # No directory grant reaches project .env, any home, physical Chrome in
        # actor domain, or /proc process credentials/memory/file descriptors.
        for path in ['/usr','/lib','/lib64','/bin','/sbin','/etc','/opt',
                     '/dev/null','/dev/urandom','/dev/random',
                     '/proc/sys','/proc/cpuinfo','/proc/meminfo','/proc/uptime']:
            resolved=Path(path).resolve()
            allow(resolved, read if resolved.is_dir() else 1 << 2)
        root=Path(__file__).resolve().parent
        # No root READ_DIR grant: Landlock directory rights recurse and would
        # expose private group/evidence filenames. Fixed modules load by path.
        allow(root/'upstream',read)
        allow(root/'confinement_support',read)
        for path in root.iterdir():
            if path.is_file() and path.suffix in ('.py','.yaml'):
                allow(path,1 << 2)
        import sys
        allow(Path(sys.prefix).resolve(),read)
        if chromium is not None:
            allow(Path(chromium).parent,read)
            # Read-only device/sysfs discovery for unchanged Chromium GPU startup.
            allow('/sys',read)
            allow('/dev/zero',1 << 2)
            # Physical Chrome needs its own maps/fd metadata; this rule is added
            # in its preexec child, and never granted to actor subprocesses.
            allow('/proc/self',read)
            allow('/proc/thread-self',read)
            if private_proc:
                # Caller must have established a separate PID namespace and /proc.
                allow('/proc',read)
    for value in [output, temporary]:
        path = Path(value).resolve()
        if path == Path('/gpfs/home/zixianma') or Path('/gpfs/home/zixianma') in path.parents:
            raise ValueError('Forbidden writable root')
        allow(path, handled & ~1)
    allow('/dev/null', 1 << 1)
    # Individual executable files deliberately exclude rm; do not grant /usr or /bin.
    executables = set()
    for directory in ['/usr/bin', '/bin', '/usr/local/bin']:
        for path in Path(directory).iterdir():
            if path.is_file() and os.access(path, os.X_OK) and path.name not in ('rm','chrome','chromium','chromium-browser','headless_shell','chrome-headless-shell'):
                resolved=path.resolve()
                # ELF interpreter execute is required for ordinary shell/Python.
                # Loader bypass is stopped by denying READ_FILE on physical Chrome.
                executables.add(resolved)
    for path in executables:
        if path.name not in ('rm','chrome','chromium','chromium-browser','headless_shell','chrome-headless-shell'):
            allow(path, 1)
    if chromium is not None:
        allow(Path(chromium).parent, 1)
    else:
        # Actor domain: physical Chromium has no execute grant, including shell calls.
        allow(Path(__file__).with_name('browser_launcher.py'), 1)
    import playwright
    allow(Path(playwright.__file__).parent / 'driver/node', 1)
    if libc.prctl(38, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'no_new_privs')
    call(446, fd, 0)
    os.close(fd)
    return abi


def assert_no_rm(command):
    if re.search(r'\brm\b', str(command)):
        raise PermissionError('rm commands require exact separate user approval')
    text = str(command)
    protected = r'(?:HOME|home|CODEX_HOME)'
    patterns = [r'(?<![\w])'+protected+r'\s*=',
                r'''["']'''+protected+r'''["']\s*\]\s*=''',
                r'''["']'''+protected+r'''["']\s*:''']
    if any(re.search(pattern,text) for pattern in patterns):
        raise PermissionError('Reassigning protected home environment variables is forbidden')


def nonlogin_args(args):
    """Suppress host profile side effects while preserving the shell command."""
    if isinstance(args, str):
        return re.sub(r'(?<![\w/])((?:/usr/bin/|/bin/)?bash)\s+-lc\b',r'\1 --noprofile --norc -c',args)
    if isinstance(args,(tuple,list)) and args and Path(str(args[0])).name=='bash':
        values=list(args)
        for index,value in enumerate(values[1:],1):
            if value=='-lc':
                values[index]='-c'
                values[1:1]=['--noprofile','--norc']
                break
            if value in ('-l','--login'):
                values[index]='--noprofile'
                values.insert(index+1,'--norc')
                break
        return values
    return args


def actor_headed_display_failure(argv, returncode, stderr):
    """Recognize an unsupported headed launch, without changing actor arguments.

    A failed headless launch or any other startup error remains an infrastructure
    halt. Only Chromium's exact missing-display error from a headed invocation
    is a local tool capability error that the native agent may observe.
    """
    headed = (bool(argv) and Path(str(argv[0])).name not in ('headless_shell', 'chrome-headless-shell')
              and not any(str(arg) == '--headless' or str(arg).startswith('--headless=')
                          for arg in argv))
    return (headed and returncode == 1 and
            'Missing X server or $DISPLAY' in stderr and
            'The platform failed to initialize.  Exiting.' in stderr)


def actor_missing_sandbox_failure(argv, returncode, stderr):
    """Recognize unsupported native sandbox selection; never rewrite actor args.

    Upstream BooleanOptionalAction for --no-sandbox can select the unsupported
    sandboxed launch. Preserve that native tool error without halting other tasks.
    All other startup failures retain the infrastructure halt.
    """
    return (bool(argv) and Path(str(argv[0])).name in ('chrome','chromium','chromium-browser')
            and returncode == -6
            and not any(str(arg).split('=',1)[0] == '--no-sandbox' for arg in argv)
            and 'FATAL:content/browser/zygote_host/zygote_host_impl_linux.cc:' in stderr
            and 'No usable sandbox!' in stderr
            and 'DevTools listening on' not in stderr)


def install_hooks():
    if not os.environ.get('CUAWRIGHT_NATIVE_GUARD_CONFIG'):
        return
    if getattr(subprocess, '_cuawright_guard_installed', False):
        return
    cfg = json.loads(Path(os.environ['CUAWRIGHT_NATIVE_GUARD_CONFIG']).read_text())
    def reserve_browser():
        if getattr(subprocess,'_cuawright_startup_failed',False) or (Path(cfg['output'])/'startup-local-halt.json').exists():
            raise RuntimeError('Browser startup previously failed; automatic operational halt')
        req = urllib.request.Request(cfg['browser_reserve_url'], data=b'{}', headers={'Content-Type':'application/json'}, method='POST')
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=30) as response:
            reservation = json.loads(response.read())
        if reservation.get('reserved') is not True:
            raise RuntimeError('Browser reservation denied')
        return reservation
    original = subprocess.Popen
    class GuardedPopen(original):
        def __init__(self, args, *positional, **kwargs):
            assert_no_rm(args)
            for key,value in cfg.get('protected_env',{}).items():
                if os.environ.get(key)!=value or (kwargs.get('env') is not None and kwargs['env'].get(key)!=value):
                    raise PermissionError('Protected home environment changed')
            if cfg.get('private_temp') and kwargs.get('env') is not None and kwargs['env'].get('TMPDIR')!=cfg['private_temp']:
                child_env=dict(kwargs['env'])
                old_temp=child_env.get('TMPDIR')
                child_env['TMPDIR']=cfg['private_temp']
                kwargs['env']=child_env
                with (Path(cfg['output'])/'temporary-adaptations.jsonl').open('a') as handle:
                    handle.write(json.dumps({'reason':'Retain per-episode private local temporary storage across native workspace subprocesses','native_TMPDIR':old_temp,'executed_TMPDIR':cfg['private_temp']})+'\n')
            original_args=args
            args=nonlogin_args(args)
            if args!=original_args:
                with (Path(cfg['output'])/'shell-adaptations.jsonl').open('a') as handle:
                    handle.write(json.dumps({'reason':'suppress host login profiles','original':original_args,'executed':args})+'\n')
            argv = args if isinstance(args, (list, tuple)) else [args]
            binary = Path(str(argv[0])).name if argv else ''
            is_browser = binary in ('chrome', 'chromium', 'chromium-browser', 'headless_shell', 'chrome-headless-shell')
            if is_browser:
                reservation = None
                args = [cfg['browser_launcher']] + list(argv[1:])
                if '--disable-dev-shm-usage' not in argv:
                    args = list(args) + ['--disable-dev-shm-usage']
                    with (Path(cfg['output'])/'browser-adaptations.jsonl').open('a') as handle:
                        handle.write(json.dumps({'reason':'Use confined private temporary storage instead of denied /dev/shm','original':argv,'executed':args,'TMPDIR':(kwargs.get('env') or os.environ).get('TMPDIR')})+'\n')
            super().__init__(args, *positional, **kwargs)
            self._cua_browser_reservation=None  # Physical reservation and startup owned by broker
            self._cua_startup_confirmed=False
            self._cua_browser_argv=list(args) if is_browser else []
            self._cua_failure_reported=False
            record = {'pid':self.pid,'process_group':os.getpgid(self.pid)}
            try:
                record['start_ticks'] = Path('/proc/' + str(self.pid) + '/stat').read_text().rsplit(')',1)[1].split()[19]
            except OSError:
                record['start_ticks'] = None
            with (Path(cfg['output'])/'owned-child-processes.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            if is_browser:
                record['proxy_only'] = True
                record['reservation'] = reservation
                with (Path(cfg['output'])/'owned-browser-proxies.jsonl').open('a') as handle:
                    handle.write(json.dumps(record)+'\n')
                if self.stderr is not None:
                    raw=self.stderr
                    class StderrTee:
                        def __getattr__(self,name):
                            return getattr(raw,name)
                        def save(self,data):
                            if data:
                                if 'DevTools listening on' in (data.decode(errors='replace') if isinstance(data,bytes) else data):
                                    self_process._cua_startup_confirmed=True
                                with (Path(cfg['output'])/('chromium-stderr-'+str(record['pid'])+'.log')).open('a') as handle:
                                    handle.write(data.decode(errors='replace') if isinstance(data,bytes) else data)
                            return data
                        def read(self,*a,**kw):
                            return self.save(raw.read(*a,**kw))
                        def readline(self,*a,**kw):
                            return self.save(raw.readline(*a,**kw))
                    self_process=self
                    self.stderr=StderrTee()
        def _cua_report_failed_startup(self,returncode):
            reservation=getattr(self,'_cua_browser_reservation',None)
            if returncode is None or returncode==0 or not reservation or getattr(self,'_cua_startup_confirmed',False) or getattr(self,'_cua_failure_reported',False):
                return
            self._cua_failure_reported=True
            stderr_path=Path(cfg['output'])/('chromium-stderr-'+str(self.pid)+'.log')
            stderr=stderr_path.read_text() if stderr_path.exists() else ''
            capability = ('actor_headed_missing_display' if actor_headed_display_failure(self._cua_browser_argv,returncode,stderr)
                          else 'actor_missing_sandbox' if actor_missing_sandbox_failure(self._cua_browser_argv,returncode,stderr) else None)
            if capability:
                with (Path(cfg['output'])/'browser-capability-errors.jsonl').open('a') as handle:
                    handle.write(json.dumps({'classification':capability,'reserved_browser_id':reservation['id'],'pid':self.pid,'exit_code':int(returncode),'argv':self._cua_browser_argv,'stderr_path':stderr_path.name,'global_halt':False})+'\n')
                return
            subprocess._cuawright_startup_failed=True
            payload={'reserved_browser_id':reservation['id'],'exit_code':int(returncode),'startup_error':'Chromium exited before an observed DevTools readiness line','devtools_observed':False}
            (Path(cfg['output'])/'startup-local-halt.json').write_text(json.dumps(payload,indent=2)+'\n')
            endpoint=cfg['browser_reserve_url'].removesuffix('/reserve')+'/failure'
            request=urllib.request.Request(endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=10) as response:
                response.read()
        def poll(self):
            result=super().poll()
            self._cua_report_failed_startup(result)
            return result
        def wait(self,*args,**kwargs):
            result=super().wait(*args,**kwargs)
            self._cua_report_failed_startup(result)
            return result
    subprocess.Popen = GuardedPopen
    subprocess._cuawright_guard_installed = True
    import httpx
    original_post = httpx.AsyncClient.post
    async def guarded_post(self, url, *args, **kwargs):
        url = str(url)
        if (getattr(subprocess,'_cuawright_startup_failed',False) or (Path(cfg['output'])/'startup-local-halt.json').exists()) and url in (cfg['model_url'],cfg['helper_url']):
            raise RuntimeError('Browser startup failure halted paid model dispatch')
        if url == cfg['model_url']:
            helper = any(frame.filename.endswith(('/tools/self_reflection.py', '/tools/image_qa.py')) for frame in inspect.stack())
            if helper:
                url = cfg['helper_url']
        return await original_post(self, url, *args, **kwargs)
    httpx.AsyncClient.post = guarded_post
    # Native persistent CLI uses Popen above. Also account for optional direct
    # Playwright launches, whose real Chrome spawn occurs in the Node driver.
    from playwright.async_api import BrowserType as AsyncBrowserType
    from playwright.sync_api import BrowserType as SyncBrowserType
    for cls in [AsyncBrowserType, SyncBrowserType]:
        for name in ['launch', 'launch_persistent_context']:
            original_launch = getattr(cls, name)
            if cls is AsyncBrowserType:
                async def async_launch(self, *args, _original=original_launch, **kwargs):
                    kwargs['executable_path']=cfg['browser_launcher']
                    return await _original(self,*args,**kwargs)
                setattr(cls,name,async_launch)
            else:
                def sync_launch(self,*args,_original=original_launch,**kwargs):
                    kwargs['executable_path']=cfg['browser_launcher']
                    return _original(self,*args,**kwargs)
                setattr(cls,name,sync_launch)


def owned_browser_members(output):
    records_path = Path(output)/'owned-browser-processes.jsonl'
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    child_path = Path(output)/'owned-child-processes.jsonl'
    child_rows = [json.loads(line) for line in child_path.read_text().splitlines()] if child_path.exists() else []
    identities = {int(row['pid']):row['start_ticks'] for row in child_rows}
    processes = {}
    try:
        process_paths=list(Path('/proc').iterdir())
    except PermissionError:
        process_paths=[]
    for path in process_paths:
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            parts = (path/'stat').read_text().rsplit(')',1)[1].split()
            if parts[0] != 'Z':
                processes[int(path.name)] = {'pid':int(path.name),'parent':int(parts[1]),'group':int(parts[2]),'start_ticks':parts[19]}
        except (OSError,ValueError):
            continue
    groups = {}
    for record in records:
        pid = int(record['pid'])
        if int(record['process_group']) != pid:
            raise RuntimeError('Recorded Chromium does not own its process group')
        if record.get('start_ticks') is None:
            raise RuntimeError('Cannot verify browser process start identity; refusing group cleanup')
        if pid in processes and processes[pid]['start_ticks'] != record['start_ticks']:
            raise RuntimeError('Browser root PID was reused; refusing group cleanup')
        groups[pid] = int(record['start_ticks'])
    owned = {pid for pid,row in processes.items() if (row['group'] in groups and int(row['start_ticks'])>=groups[row['group']]) or (identities.get(pid) is not None and identities.get(pid)==row['start_ticks'])}
    while True:
        expanded = owned | {pid for pid,row in processes.items() if row['parent'] in owned}
        if expanded == owned:
            break
        owned = expanded
    return [{'pid':pid,'start_ticks':processes[pid]['start_ticks']} for pid in owned]


def terminate_recorded_members(members):
    import signal
    import time
    def alive(row):
        try:
            parts = Path('/proc/'+str(row['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
            return parts[0]!='Z' and parts[19]==row['start_ticks']
        except OSError:
            return False
    for signum in [signal.SIGTERM, signal.SIGKILL]:
        for row in members:
            if alive(row):
                try:
                    os.kill(row['pid'],signum)
                except ProcessLookupError:
                    pass
        time.sleep(0.3)
    return {'verified':not any(alive(row) for row in members),'owned_members_checked':len(members)}
