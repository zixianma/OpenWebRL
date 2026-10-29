#!/usr/bin/env python3
"""A separately reserved local startup probe. Never invokes an actor model."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import shlex
import tempfile
import time
import traceback
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parent


def startup_create_command(python,output):
    # Diagnostic only: shared-filesystem startup can outlast native30s readiness.
    # LocalWorkspaceEnvironment retains its existing240s command deadline.
    return shlex.join([str(python),'-m','cuawright.webwright.tools.browser_session','create','--workspace-dir',str(output),'--start-url','about:blank','--startup-timeout','180'])


def run(output,endpoint,approved_browser_remaining,create_function=None):
    if approved_browser_remaining<=0:
        raise PermissionError('No approved browser starts remain; probe is forbidden')
    parsed=urlparse(endpoint)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or not endpoint.endswith('/reserve'):
        raise ValueError('A loopback gateway browser reservation endpoint is required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='cua-startup-',dir='/tmp'))
    for key in list(os.environ):
        if any(value in key.upper() for value in ['API_KEY','TOKEN','SECRET','CREDENTIAL']):
            os.environ.pop(key,None)
    for key in ['BASH_ENV','ENV','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
        os.environ.pop(key,None)
    os.environ.update({'TMPDIR':str(temporary),'PLAYWRIGHT_BROWSERS_PATH':'/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers','BROWSER_MODE':'local','WEBWRIGHT_BROWSER_BACKEND':'local','PYTHONDONTWRITEBYTECODE':'1','NO_PROXY':'127.0.0.1,localhost,::1','no_proxy':'127.0.0.1,localhost,::1'})
    for key in ['XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','PYTHONPYCACHEPREFIX']:
        path=output/'.cache'/key.lower();path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ['PYTHONPATH']=':'.join([str(ROOT/'confinement_support'),str(ROOT),str(ROOT/'upstream/src')])
    os.environ['PATH']=str(Path(sys.executable).parent)+':'+os.environ.get('PATH','/usr/bin:/bin')
    sys.path.insert(0,str(ROOT/'upstream/src'))
    from browser_broker_support import start_broker,stop_broker
    cfg={'model_url':'disabled','helper_url':'disabled','browser_reserve_url':endpoint,'output':str(output),'private_temp':str(temporary),'protected_env':{name:os.environ.get(name) for name in ['HOME','home','CODEX_HOME']}}
    broker=start_broker(cfg)
    guard=output/'guard-config.json';guard.write_text(json.dumps(cfg));os.environ['CUAWRIGHT_NATIVE_GUARD_CONFIG']=str(guard)
    from confinement import install_hooks,restrict,owned_browser_members,terminate_recorded_members
    install_hooks();abi=restrict(output,temporary,None)
    from cuawright.webwright.tools import browser_session as browser
    from cuawright.webwright.models.base import BaseModel
    model_dispatches=[]
    async def reject_model(*args,**kwargs):
        model_dispatches.append(time.time())
        raise AssertionError('Browser diagnostic must never dispatch a model')
    BaseModel._post_with_retries=reject_model
    started=time.time();error=None;created=False;cleanup={};kernel_guard={}
    # These are denied before exec: no physical browser start and no reservation.
    import subprocess
    physical=cfg['physical_chromium']
    try:
        Path(physical).read_bytes();kernel_guard['physical_chromium_read_denied']=False
    except PermissionError:
        kernel_guard['physical_chromium_read_denied']=True
    for name,path in [('broker_config_read_denied',str(Path(cfg['browser_broker_socket']).parent/'config.json')),('parent_environ_read_denied','/proc/'+str(os.getppid())+'/environ')]:
        try:
            Path(path).read_bytes();kernel_guard[name]=False
        except PermissionError:
            kernel_guard[name]=True
    # Bypass Python hook intentionally, so the kernel (not monkeypatch) is tested.
    original=next(cls for cls in subprocess.Popen.__mro__[1:] if cls.__module__=='subprocess')
    try:
        process=original([physical,'--version'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        process.wait(timeout=5);kernel_guard['physical_chromium_exec_denied']=False
    except PermissionError:
        kernel_guard['physical_chromium_exec_denied']=True
    shell=subprocess.run(['/bin/bash','--noprofile','--norc','-c',shlex.quote(physical)+' --version'],capture_output=True)
    kernel_guard['shell_physical_exec_denied']=shell.returncode==126
    if not all(kernel_guard.values()):
        raise RuntimeError('Kernel browser/credential guard failed: '+str(kernel_guard))
    try:
        if create_function is not None:
            create_function()
        else:
            from cuawright.webwright.config import get_config_from_spec
            from cuawright.webwright.environments import get_environment
            config=get_config_from_spec(str(ROOT/'resolved-native-config.yaml'))['environment']
            config['output_dir']=str(output)
            config['start_url']='about:blank'
            config['env'].update({key:os.environ[key] for key in config['env'] if key in os.environ})
            config['env']['PYTHONPATH']=os.environ['PYTHONPATH']
            environment=get_environment(config)
            environment.prepare(task='Native local browser startup health probe',task_id='startup-probe',start_url='about:blank')
            command=startup_create_command(sys.executable,output)
            executed=environment.execute({'command':command,'bash_command':command})
            observation=executed.get('observation',{})
            (output/'native-workspace-observation.json').write_text(json.dumps(observation,indent=2,default=str)+'\n')
            if not observation.get('success'):
                raise RuntimeError('Native workspace browser create failed with returncode '+str(observation.get('returncode'))+': '+str(observation.get('command_output',''))[-4000:])
            if not browser._workspace_file(output,browser.DEFAULT_PERSISTENT_SESSION_FILE).exists():
                raise RuntimeError('Native workspace create returned success without a session file')
        created=True
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    finally:
        try:
            cleanup=asyncio.run(browser._close_persistent_session_files(session_path=browser._workspace_file(output,browser.DEFAULT_PERSISTENT_SESSION_FILE),events_path=browser._workspace_file(output,browser.DEFAULT_SESSION_EVENTS_FILE),delete_user_data=False,delete_session_file=False))
        except BaseException as exc:
            cleanup={'success':False,'identity_error':str(exc),'process_verification':{'verified':False}}
        try:
            cleanup['broker_verification']=stop_broker(broker,cfg)
            verification=cleanup['broker_verification']
            cleanup['process_verification']={'verified':verification.get('verified') is True and verification.get('exit_code')==0,'source':'physical-browser-broker','checks':verification.get('checks',[]),'evidence':verification.get('evidence')}
        except BaseException as exc:
            cleanup['broker_verification']={'verified':False,'error':str(exc)}
        record={'startup_healthy':created and error is None,'actor_release_permitted':created and error is None and cleanup.get('process_verification',{}).get('verified',False) and cleanup.get('broker_verification',{}).get('verified',False),'error':error,'model_dispatches':len(model_dispatches),'paid_model_calls':0,'owned_teardown':cleanup,'elapsed_seconds':time.time()-started,'landlock_abi':abi,'stderr_files':[str(path) for path in output.glob('chromium-stderr-*.log')],'temporary_directory':str(temporary),'offline_mock':create_function is not None,'kernel_guard':kernel_guard,'diagnostic_startup_timeout_seconds':180}
        (output/'startup-diagnostic.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--endpoint',required=True);parser.add_argument('--approved-browser-remaining',type=int,required=True);args=parser.parse_args()
    record=run(args.output,args.endpoint,args.approved_browser_remaining)
    print(json.dumps(record));return 0 if record['actor_release_permitted'] else 1


if __name__=='__main__':
    raise SystemExit(main())
