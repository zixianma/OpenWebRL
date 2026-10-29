#!/usr/bin/env python3
"""Physical browser broker outside actor execution confinement.

Two flock leases are shared by every episode using one shard gateway. Physical
starts are reserved centrally before exec, including failed exec/startup attempts.
The actor cannot execute the real Chromium inode, even through a shell or copied
binary, because its Landlock domain grants execute only to fixed system tools and
our immutable launcher. All real Chrome groups are recorded and reaped here.
"""
import argparse
import base64
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request


def physical_arguments(args):
    """Filesystem/crash-reporting adaptation; preserve all actor browser flags."""
    result=list(args)
    for flag in ('--disable-dev-shm-usage',):
        if flag not in result:
            result.append(flag)
    return result


def identity(pid):
    try:
        fields = Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid, 'process_group':int(fields[2]), 'start_ticks':fields[19], 'state':fields[0]}
    except OSError:
        return None


def stop_group(proc, original):
    current = identity(proc.pid)
    if current and original and current['start_ticks'] != original['start_ticks']:
        raise RuntimeError('Broker browser PID reused')
    # Group identity was established at spawn; a reaped parent can leave children.
    members = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row = identity(int(path.name))
            if row and original and row['process_group'] == proc.pid and int(row['start_ticks']) >= int(original['start_ticks']) and row['state'] != 'Z':
                members.append(row)
    child=getattr(proc,'_cua_namespace_child',None)
    if child and not any(row['pid']==child['pid'] for row in members):
        now=identity(child['pid'])
        if now and now['start_ticks']==child['start_ticks'] and now['state']!='Z':
            members.append(now)
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for row in members:
            now = identity(row['pid'])
            if now and now['start_ticks'] == row['start_ticks'] and now['state'] != 'Z':
                try:
                    os.kill(row['pid'], signum)
                except ProcessLookupError:
                    pass
        time.sleep(.1)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    def remaining():
        return [row['pid'] for row in members if (now:=identity(row['pid'])) and now['start_ticks']==row['start_ticks'] and now['state']!='Z']
    # SIGKILL may be pending while namespace init finishes kernel/filesystem
    # teardown. The supervisor exiting alone does not prove descendants gone.
    # Keep the lease until every captured identity settles, with a hard bound.
    post_kill_started=time.monotonic()
    post_kill_deadline=post_kill_started+5.0
    checks=0
    alive=remaining()
    while alive and time.monotonic()<post_kill_deadline:
        time.sleep(.05)
        checks+=1
        alive=remaining()
    return {'verified':not alive,'members_checked':len(members),'remaining':alive,'post_kill_rechecks':checks,'post_kill_wait_seconds':time.monotonic()-post_kill_started}


class Broker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output = Path(cfg['broker_evidence'])
        self.lock = threading.RLock()
        from command_timeout_guard import CommandOwnership
        self.command_ownership=CommandOwnership(os.getppid(),self.output)
        self.group_relay=None
        if cfg.get('private_group_config'):
            from native_group_relay import GroupRelay
            self.group_relay=GroupRelay(cfg['private_group_config'],self.command_ownership)
        self.children = {}; self.capture_proxies=[]
        self.stopping = False
        self.cleanup_failures = []
        self.failed_leases = []
        root=Path(__file__).resolve().parent
        try:
            shard=Path(cfg['output']).resolve().relative_to(root).parts[0]
            if not __import__('re').fullmatch(r'shard[0-9]+',shard):
                raise ValueError('No registered logical shard')
            self.slots=root/'browser-leases'/shard
            lease_scope='campaign-logical-shard'
        except ValueError:
            if cfg['physical_chromium']=='/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome':
                raise RuntimeError('Live browser lease requires registered logical-shard output')
            from urllib.parse import urlparse
            key=hashlib.sha256(urlparse(cfg['browser_reserve_url']).netloc.encode()).hexdigest()[:20]
            self.slots=Path('/tmp')/('cuawright-fixture-slots-'+str(os.getuid())+'-'+key)
            lease_scope='offline-fixture-only'
        self.slots.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.record({'event':'broker_started','uid':os.getuid(),'broker_pid':os.getpid(),'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'physical_chromium':cfg['physical_chromium'],'browser_reserve_url':cfg['browser_reserve_url'],'output':cfg['output'],'private_temp':cfg['private_temp'],'concurrency_limit':2,'lease_scope':lease_scope,'lease_directory':str(self.slots),'host_pid_namespace':os.readlink('/proc/self/ns/pid'),'runtime_child_source_binding':cfg.get('runtime_child_source_binding')})

    def record(self, row):
        with self.lock, (self.output/'browser-broker-events.jsonl').open('a') as handle:
            handle.write(json.dumps({'time':time.time(), **row})+'\n')

    def slot(self):
        for index in range(2):
            handle = (self.slots/str(index)).open('a+')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return index, handle
            except BlockingIOError:
                handle.close()
        raise RuntimeError('Shard physical browser concurrency cap (2) reached')

    def reserve(self):
        req = urllib.request.Request(self.cfg['browser_reserve_url'], data=b'{}', headers={'Content-Type':'application/json'}, method='POST')
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=30) as response:
            row = json.loads(response.read())
        if row.get('reserved') is not True:
            raise RuntimeError('Physical browser start reservation denied')
        return row

    def serve_launch(self, connection, request):
        slot_handle = None
        proc = None
        original = None
        reservation = None
        advertiser = None
        def send(row):
            connection.sendall(json.dumps(row).encode()+b'\n')
        try:
            if self.stopping:
                raise RuntimeError('Episode broker is stopping')
            args = request['args']
            if not isinstance(args,list) or not all(isinstance(a,str) and '\0' not in a for a in args):
                raise ValueError('Invalid Chromium arguments')
            cwd = Path(request['cwd']).resolve()
            roots = [Path(self.cfg[k]).resolve() for k in ('output','private_temp')]
            if not any(cwd==p or p in cwd.parents for p in roots):
                raise PermissionError('Browser cwd outside episode roots')
            env = request['env']
            if not isinstance(env,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in env.items()):
                raise ValueError('Invalid browser environment')
            for key, value in self.cfg['protected_env'].items():
                if env.get(key)!=value:
                    raise PermissionError('Protected environment changed')
            env['TMPDIR'] = self.cfg['private_temp']
            env.pop('CUAWRIGHT_NATIVE_GUARD_CONFIG',None)
            if '--remote-debugging-pipe' in args:
                raise PermissionError('Direct Playwright pipe launch unsupported by mediated local Chromium; use native CDP persistent browser')
            original_args=list(args)
            args=physical_arguments(args)
            slot_index, slot_handle = self.slot()
            reservation = self.reserve()
            self.record({'event':'reserved', 'reservation':reservation, 'slot':slot_index, 'argv':[self.cfg['physical_chromium']]+args,'original_argv':[self.cfg['physical_chromium']]+original_args,'adaptations':['confined-private-temp-instead-of-dev-shm','private-pid-namespace-and-proc']})
            namespace_cfg={**self.cfg,'argv':[self.cfg['physical_chromium']]+args,'reservation_id':reservation['id'],'expected_host_pid_namespace':os.readlink('/proc/self/ns/pid'),'namespace_proof':str(self.output/('namespace-proof-'+str(reservation['id'])+'.json'))}
            namespace_path=self.output/('namespace-config-'+str(reservation['id'])+'.json')
            namespace_path.write_text(json.dumps(namespace_cfg))
            namespace_command=['/usr/bin/unshare','--user','--map-root-user','--pid','--fork','--kill-child=SIGKILL','--mount-proc',sys.executable,'-B',str(Path(__file__).with_name('browser_namespace.py')),'--config',str(namespace_path)]
            with self.lock:
                if self.stopping:
                    raise RuntimeError('Broker stopped before exec; start remains charged')
                if self.cfg.get('offline_fixture_skip_namespace') is True:
                    if Path(self.cfg['physical_chromium']).resolve()==Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome').resolve():
                        raise PermissionError('Cannot disable private PID namespace for real Chrome')
                    from confinement import restrict
                    def fixture_domain():
                        restrict(self.cfg['output'],self.cfg['private_temp'],self.cfg['physical_chromium'])
                    proc=subprocess.Popen([self.cfg['physical_chromium']]+args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,preexec_fn=fixture_domain)
                else:
                    proc = subprocess.Popen(namespace_command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
                original = identity(proc.pid)
                if not original or original['process_group'] != proc.pid:
                    raise RuntimeError('Cannot establish physical browser group identity')
                self.children[proc.pid]=(proc,original)
            owned = {**original, 'reservation':reservation, 'broker_pid':os.getpid(), 'slot':slot_index}
            with (self.output/'owned-browser-processes.jsonl').open('a') as handle:
                handle.write(json.dumps(owned)+'\n')
            namespace_child=None
            deadline=time.monotonic()+5
            while proc.poll() is None and time.monotonic()<deadline:
                try:
                    child_pids=Path('/proc/'+str(proc.pid)+'/task/'+str(proc.pid)+'/children').read_text().split()
                    if child_pids:
                        child_pid=int(child_pids[0]);namespace_child=identity(child_pid)
                        if namespace_child:
                            status=Path('/proc/'+str(child_pid)+'/status').read_text()
                            namespace_child['nspid']=[int(v) for line in status.splitlines() if line.startswith('NSpid:') for v in line.split()[1:]]
                            namespace_child['pid_namespace']=os.readlink('/proc/'+str(child_pid)+'/ns/pid')
                            break
                except (OSError,ValueError):
                    pass
                time.sleep(.005)
            proc._cua_namespace_child=namespace_child
            self.record({'event':'spawned', **owned,'namespace_child':namespace_child,'namespace_command':namespace_command,'namespace_proof':namespace_cfg['namespace_proof']})
            from native_capture import DevtoolsAdvertiser, CDPCaptureProxy
            def capture_proxy(target):
                proxy=CDPCaptureProxy(target,self.output,owned)
                with self.lock:self.capture_proxies.append(proxy)
                return proxy
            advertiser=DevtoolsAdvertiser(capture_proxy)
            selected = selectors.DefaultSelector()
            selected.register(connection,selectors.EVENT_READ,'client')
            for stream,name in ((proc.stdout,'stdout'),(proc.stderr,'stderr')):
                os.set_blocking(stream.fileno(),False)
                selected.register(stream,selectors.EVENT_READ,name)
            while True:
                for item, _ in selected.select(.2):
                    if item.data=='client':
                        if not connection.recv(4096):
                            raise ConnectionError('Launcher disconnected')
                    else:
                        data = os.read(item.fileobj.fileno(),65536)
                        if data:
                            if item.data=='stderr':
                                with (self.output/('chromium-stderr-'+str(proc.pid)+'.log')).open('ab') as handle:
                                    handle.write(data)
                                data=advertiser.feed(data)
                            if data:send({'type':item.data,'data':base64.b64encode(data).decode()})
                        else:
                            if item.data=='stderr':
                                tail=advertiser.flush()
                                if tail:send({'type':'stderr','data':base64.b64encode(tail).decode()})
                            selected.unregister(item.fileobj)
                if proc.poll() is not None:
                    send({'type':'exit','returncode':proc.returncode})
                    break
        except (BrokenPipeError, ConnectionError):
            pass
        except BaseException as exc:
            self.record({'event':'launch_error','error':str(exc),'reservation':reservation})
            try:
                send({'type':'error','message':str(exc)})
            except OSError:
                pass
        finally:
            if proc:
                stderr_path=self.output/('chromium-stderr-'+str(proc.pid)+'.log')
                stderr=stderr_path.read_text(errors='replace') if stderr_path.exists() else ''
                returncode=proc.poll()
                if returncode not in (None,0) and 'DevTools listening on' not in stderr:
                    from confinement import actor_headed_display_failure,actor_missing_sandbox_failure
                    argv=[self.cfg['physical_chromium']]+request['args']
                    capability=('actor_headed_missing_display' if actor_headed_display_failure(argv,returncode,stderr) else 'actor_missing_sandbox' if actor_missing_sandbox_failure(argv,returncode,stderr) else None)
                    self.record({'event':'startup_failure','pid':proc.pid,'reservation':reservation,'capability':capability,'exit_code':returncode})
                    if capability is None:
                        payload={'reserved_browser_id':reservation['id'],'exit_code':returncode,'startup_error':'Physical Chromium exited without DevTools readiness','devtools_observed':False}
                        (Path(self.cfg['output'])/'startup-local-halt.json').write_text(json.dumps(payload)+'\n')
                        req=urllib.request.Request(self.cfg['browser_reserve_url'].removesuffix('/reserve')+'/failure',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
                        try:
                            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=10) as response:
                                response.read()
                        except OSError as exc:
                            self.record({'event':'halt_report_error','error':str(exc)})
                cleanup=stop_group(proc,original)
                with self.lock:
                    if cleanup['verified']:
                        self.children.pop(proc.pid,None)
                    else:
                        self.stopping=True
                        self.cleanup_failures.append(cleanup)
                        self.failed_leases.append(slot_handle)
                        slot_handle=None
                if advertiser is not None and advertiser.proxy is not None:
                    try:advertiser.proxy.close()
                    except BaseException as exc:
                        self.cleanup_failures.append({'verified':False,'remaining':['owned-cdp-capture-tee'],'error':str(exc)})
                self.record({'event':'reaped','pid':proc.pid,'start_ticks':(original or {}).get('start_ticks'),'cleanup':cleanup})
            if slot_handle:
                slot_handle.close()

    def shutdown(self):
        command_checks=[]
        for token in list(self.command_ownership.commands):
            command_checks.append(self.command_ownership.cleanup(token))
        with self.lock:
            self.stopping=True
            rows=list(self.children.values())
        checks=command_checks+[stop_group(proc,original) for proc,original in rows]
        for proxy in list(self.capture_proxies):
            try:checks.append(proxy.close())
            except BaseException as exc:checks.append({'verified':False,'remaining':['owned-cdp-capture-tee'],'error':str(exc)})
        checks=list(self.cleanup_failures)+checks
        self.record({'event':'shutdown','verified':all(c['verified'] for c in checks),'checks':checks})
        return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    # Parent death must close physical browsers even if the native agent is killed.
    libc=ctypes.CDLL(None)
    parent=os.getppid()
    if libc.prctl(1,signal.SIGTERM,0,0,0):
        raise RuntimeError('Cannot establish broker parent-death signal')
    import playwright  # Preload before threaded launch preexec confinement.
    broker=Broker(cfg)
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            from native_group_transport import MAX_FRAME
            line=self.rfile.readline(MAX_FRAME+1)
            if len(line)>MAX_FRAME or not line.endswith(b'\n'):
                return
            request=json.loads(line)
            if request.get('op')=='launch':
                broker.serve_launch(self.connection,request)
            elif request.get('op') in ('command_register','command_cleanup','command_finish'):
                try:
                    operation=request['op']
                    import struct
                    peer_pid,peer_uid,_=struct.unpack('3i',self.connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i')))
                    broker.command_ownership.authorize(peer_pid,peer_uid,request['token'],operation)
                    if operation=='command_register':result=broker.command_ownership.register(request['token'],request['pid'],request['command_sha256'])
                    elif operation=='command_cleanup':result=broker.command_ownership.cleanup(request['token'])
                    else:result=broker.command_ownership.finish(request['token'])
                    self.wfile.write(json.dumps(result).encode()+b'\n')
                except BaseException as error:
                    self.wfile.write(json.dumps({'verified':False,'error':str(error)}).encode()+b'\n')
            elif request.get('op') in ('group_open','group_create','group_finish','group_record'):
                try:
                    import struct
                    peer_pid,peer_uid,_=struct.unpack('3i',self.connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i')))
                    if broker.group_relay is None:raise PermissionError('No approved N3 group relay')
                    result=broker.group_relay.handle(peer_pid,peer_uid,request)
                    self.wfile.write(json.dumps({'verified':True,'result':result}).encode()+b'\n')
                except BaseException as error:
                    self.wfile.write(json.dumps({'verified':False,'error':str(error)}).encode()+b'\n')
            elif request.get('op')=='shutdown':
                checks=broker.shutdown()
                self.wfile.write(json.dumps({'verified':all(c['verified'] for c in checks),'checks':checks}).encode()+b'\n')
                threading.Thread(target=server.shutdown,daemon=True).start()
    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads=False
    server=Server(cfg['browser_broker_socket'],Handler)
    os.chmod(cfg['browser_broker_socket'],0o600)
    def interrupted(signum,frame):
        broker.shutdown()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    if os.getppid()!=parent:
        broker.shutdown()
        return 1
    Path(cfg['broker_ready']).write_text(json.dumps({'pid':os.getpid(),'start_ticks':identity(os.getpid())['start_ticks']})+'\n')
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        broker.shutdown()
        server.server_close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
