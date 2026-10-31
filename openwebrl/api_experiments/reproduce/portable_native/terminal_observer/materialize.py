"""Install the validated trusted terminal observer into a new isolated runtime.

Archived CUAWright files are never changed in the source bundle. This opt-in
source adaptation adds evidence capture and a zero-model-call explicit-close
probe; it leaves actor prompts, models, actions and scientific settings intact.
"""
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Pinned source insertion differs: ' + old[:100])
    return source.replace(old, new, 1)


def materialize(runtime):
    runtime = Path(runtime).resolve()
    if (runtime / 'trusted_terminal_capture.py').exists():
        raise ValueError('Observer already materialized; use a new isolated runtime')
    upgrade = False
    broker_path = runtime / 'browser_broker.py'
    episode_path = runtime / 'native_episode.py'
    broker = broker_path.read_text()
    episode = episode_path.read_text()
    if not upgrade:
        broker = replace_once(broker, "            if request.get('op')=='launch':", """            if request.get('op')=='terminal_capture':
                try:
                    import struct
                    from trusted_terminal_capture import capture
                    peer_pid,peer_uid,_=struct.unpack('3i',self.connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i')))
                    result=capture(broker,peer_pid,peer_uid,request['actor_finished_epoch'])
                    self.wfile.write(json.dumps(result).encode()+b'\\n')
                except BaseException as error:
                    self.wfile.write(json.dumps({'verified':False,'error':str(error)}).encode()+b'\\n')
                return
            if getattr(broker,'terminal_observer_started',False) and request.get('op') in ('launch','command_register','group_open','group_create','group_finish','group_record'):
                self.wfile.write(json.dumps({'verified':False,'error':'Actor admission closed after terminal observation'}).encode()+b'\\n')
                return
            if request.get('op')=='launch':""")
    broker = replace_once(broker, "            if request.get('op')=='terminal_capture':", """            if request.get('op')=='terminal_remember_session':
                try:
                    import struct
                    from trusted_terminal_capture import remember_session
                    peer_pid,peer_uid,_=struct.unpack('3i',self.connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i')))
                    broker.command_ownership.authorize(peer_pid,peer_uid,'terminal-remember','terminal_remember_session')
                    result=remember_session(broker)
                    self.wfile.write(json.dumps(result).encode()+b'\\n')
                except BaseException as error:
                    self.wfile.write(json.dumps({'verified':False,'error':str(error)}).encode()+b'\\n')
                return
            if request.get('op')=='terminal_capture':""")
    broker = replace_once(broker, "if getattr(broker,'terminal_observer_started',False) and request.get('op')", "if (getattr(broker,'terminal_observer_started',False) or getattr(broker,'terminal_observer_requested',False)) and request.get('op')")
    broker = replace_once(broker, '                proxy=CDPCaptureProxy(target,self.output,owned)', '                proxy=CDPCaptureProxy(target,self.output,owned)\n                proxy.terminal_preclose_done=threading.Event()')
    broker = replace_once(broker, "                    else:result=broker.command_ownership.finish(request['token'])", """                    else:
                        result=broker.command_ownership.finish(request['token'])
                        try:
                            from trusted_terminal_capture import remember_session
                            remember_session(broker)
                        except BaseException as exc:
                            broker.record({'event':'terminal_session_snapshot_unavailable','error':str(exc)})""")
    broker = replace_once(broker, '                cleanup=stop_group(proc,original)', """                if advertiser is not None and advertiser.proxy is not None and proc.poll() is None:
                    try:
                        from trusted_terminal_capture import preclose_capture
                        preclose_capture(self,advertiser.proxy)
                    except BaseException as exc:
                        self.record({'event':'terminal_preclose_capture_unqualified','error':str(exc),'browser_identity':original})
                cleanup=stop_group(proc,original)""")
    if not upgrade:
        episode = replace_once(episode, '    cleanup = {}\n    started = time.time()',
                           '    cleanup = {}\n    terminal_capture = {}\n    terminal_observer_seconds = 0.0\n    started = time.time()')
        episode = replace_once(episode, "    finally:\n        try:\n            cleanup = asyncio.run(_close_persistent_session_files(", """    finally:
        observer_started = time.time()
        try:
            if error is None:
                from trusted_terminal_capture import request_capture
                terminal_capture = request_capture(guard_cfg, observer_started)
            else:
                terminal_capture = {'verified':False,'not_attempted':'actor_execution_failed'}
        except BaseException as exc:
            terminal_capture = {'verified':False,'error':str(exc)}
        terminal_observer_seconds = time.time() - observer_started
        try:
            cleanup = asyncio.run(_close_persistent_session_files(""")
        episode = replace_once(episode, "'task_latency_seconds':time.time()-started,", "'task_latency_seconds':time.time()-started-terminal_observer_seconds,'terminal_capture':terminal_capture,'terminal_observer_seconds':terminal_observer_seconds,")
    episode = replace_once(episode, '    # Preload fixed transport before runtime-root directory discovery is denied.',
                           '    from trusted_terminal_capture import request_capture\n    # Preload fixed transport before runtime-root directory discovery is denied.')
    episode = replace_once(episode, '                from trusted_terminal_capture import request_capture\n                terminal_capture = request_capture',
                           '                terminal_capture = request_capture')
    compile(broker, str(broker_path), 'exec')
    compile(episode, str(episode_path), 'exec')
    helper = (HERE / 'trusted_terminal_capture.py').read_bytes()
    (runtime / 'trusted_terminal_capture.py').write_bytes(helper)
    broker_path.write_text(broker)
    episode_path.write_text(episode)
    return {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (broker_path, episode_path, runtime / 'trusted_terminal_capture.py')}


def install_probe(runtime):
    """Adapt existing metered diagnostic entrypoint; preserves parent cleanup ABI."""
    runtime = Path(runtime).resolve()
    target = runtime / 'diagnostic_browser.py'
    source = target.read_text()
    source = replace_once(source, '    from browser_broker_support import start_broker,stop_broker',
                          '    from diagnostic_terminal_probe import exercise\n    from browser_broker_support import start_broker,stop_broker')
    source = replace_once(source, 'started=time.time();error=None;created=False;cleanup={};kernel_guard={}',
                          'started=time.time();error=None;created=False;cleanup={};kernel_guard={};terminal_probe={}')
    source = replace_once(source, '        created=True\n', """        created=True
        if create_function is None:
            terminal_probe=exercise(environment,output,cfg)
""")
    source = replace_once(source, "        (output/'startup-diagnostic.json').write_text", """        record.update({'terminal_capture_probe':terminal_probe,'synthetic_fixture_only':True,'scientific_result_eligible':False,'synthetic_ungraded':True,'terminal_observer_preclose_verified':terminal_probe.get('terminal_capture',{}).get('verified') is True})
        if create_function is None and terminal_probe.get('terminal_capture',{}).get('verified') is not True:
            record['actor_release_permitted']=False
        (output/'startup-diagnostic.json').write_text""")
    compile(source, str(target), 'exec')
    target.write_text(source)
    (runtime / 'diagnostic_terminal_probe.py').write_bytes((HERE / 'diagnostic_terminal_probe.py').read_bytes())
    return {name: hashlib.sha256((runtime / name).read_bytes()).hexdigest()
            for name in ('diagnostic_browser.py', 'diagnostic_terminal_probe.py')}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    print(json.dumps(materialize(parser.parse_args().runtime), indent=2))
