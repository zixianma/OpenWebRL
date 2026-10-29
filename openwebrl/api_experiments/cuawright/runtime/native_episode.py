#!/usr/bin/env python3
"""Run the frozen upstream native agent; local adaptation and confinement only."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'upstream'


def install_workspace_listing_race_guard():
    """Keep pinned file-list semantics while omitting paths gone before stat.

    Chromium rotates profile files between is_file() and the original sort's
    stat(). Only that FileNotFoundError is recoverable; every other error
    propagates. The upstream source and model configuration stay unchanged.
    """
    import hashlib
    import inspect
    from cuawright.webwright.environments.local_workspace import LocalWorkspaceEnvironment
    original = LocalWorkspaceEnvironment._recent_workspace_files
    if getattr(original, "_cuawright_listing_race_v1", None) is not None:
        return original._cuawright_listing_race_v1
    expected = "1da7ee42dfc4a89de5d57f57c35a3896c5c2b7c31d13e1c189135819497119e9"
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != expected:
        raise RuntimeError("Pinned recent-workspace-files implementation changed")
    evidence = {"adapter": "omit_disappeared_workspace_files_v1",
                "upstream_method_sha256": expected, "omitted_paths": []}
    def recent_workspace_files(self):
        workspace_dir = self._workspace_dir()
        files = [path for path in workspace_dir.rglob("*") if path.is_file()]
        present = []
        for path in files:
            try:
                modified = path.stat().st_mtime
            except FileNotFoundError:
                evidence["omitted_paths"].append(str(path.relative_to(workspace_dir)))
                continue
            present.append((path, modified))
        present.sort(key=lambda row: row[1], reverse=True)
        recent = present[:self.config.recent_files_limit]
        return [str(path.relative_to(workspace_dir)) for path, _ in recent]
    recent_workspace_files._cuawright_listing_race_v1 = evidence
    LocalWorkspaceEnvironment._recent_workspace_files = recent_workspace_files
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-json', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--arm', required=True, choices=['luna_high_single','luna_high_n3_select_high','luna_high_n5_select_high','sol_high_single','sol_high_n3_select_high'])
    parser.add_argument('--group-config')
    parser.add_argument('--absolute-deadline',type=float,required=True)
    args = parser.parse_args()
    from native_proposal_adapter import ARMS
    model_name,candidate_count=ARMS[args.arm]
    if candidate_count>1 and not args.group_config:raise ValueError('Proposal selection requires private host group configuration')
    deadline_monotonic=time.monotonic()+args.absolute_deadline-time.time()
    if deadline_monotonic<=time.monotonic():raise ValueError('Native allocation deadline expired')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    task = json.loads(Path(args.task_json).read_text())
    endpoint = args.endpoint.rstrip('/')
    if not endpoint.endswith('/responses'):
        endpoint += '/responses'
    parsed = urlparse(endpoint)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or '/model/' not in endpoint:
        raise ValueError('A local, metered /model/{case}/responses endpoint is required')
    helper = endpoint.replace('/model/', '/helper/', 1)
    reserve = endpoint.replace('/model/', '/browser/', 1).removesuffix('/responses') + '/reserve'
    temporary = Path(tempfile.mkdtemp(prefix='cua-native-', dir='/tmp'))
    for key in list(os.environ):
        if any(part in key.upper() for part in ['API_KEY', 'TOKEN', 'SECRET', 'CREDENTIAL']):
            os.environ.pop(key, None)
    for key in ['BASH_ENV', 'ENV', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
        os.environ.pop(key, None)
    os.environ.update({'OPENAI_API_KEY':'local-metered-dummy','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(temporary),'NO_PROXY':'127.0.0.1,localhost,::1','no_proxy':'127.0.0.1,localhost,::1','PLAYWRIGHT_BROWSERS_PATH':'/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers','WEBWRIGHT_BROWSER_BACKEND':'local','BROWSER_MODE':'local'})
    for name in ['XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','PYTHONPYCACHEPREFIX','PIP_CACHE_DIR','UV_CACHE_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DIR']:
        path = output / '.cache' / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['PATH'] = str(Path(sys.executable).parent) + ':' + os.environ.get('PATH', '/usr/bin:/bin')
    os.environ['PYTHONPATH'] = ':'.join([str(ROOT / 'confinement_support'), str(ROOT), str(UPSTREAM / 'src')])
    sys.path.insert(0, str(UPSTREAM / 'src'))
    from cuawright.webwright.config import get_config_from_spec
    import yaml
    override = get_config_from_spec(str(ROOT / 'local_override.yaml'))
    override['environment']['env'].update({name: os.environ[name] for name in override['environment']['env'] if name in os.environ})
    override['environment']['env']['PYTHONPATH'] = os.environ['PYTHONPATH']
    override.setdefault('model', {}).update({'responses_url':endpoint, 'openai_endpoint':endpoint, 'openai_api_key':'local-metered-dummy','model_name':model_name,'reasoning_effort':'high','max_output_tokens':20000})
    runtime_config = output / 'runtime-override.yaml'
    runtime_config.write_text(yaml.safe_dump(override, sort_keys=False))
    from command_timeout_guard import bind_runtime_inventory
    runtime_child_source_binding=bind_runtime_inventory(ROOT,output)
    from browser_broker_support import start_broker, stop_broker
    guard_cfg = {'model_url':endpoint,'helper_url':helper,'browser_reserve_url':reserve,'output':str(output),'private_temp':str(temporary),'protected_env':{name:os.environ.get(name) for name in ['HOME','home','CODEX_HOME']},'runtime_child_source_binding':runtime_child_source_binding}
    workspace_listing_guard = install_workspace_listing_race_guard()
    from command_timeout_guard import install as install_command_timeout_guard
    command_timeout_guard=install_command_timeout_guard(guard_cfg)
    host_cfg=dict(guard_cfg)
    if candidate_count>1:host_cfg['private_group_config']=str(Path(args.group_config).resolve())
    broker = start_broker(host_cfg)
    # Only public relay socket/lifecycle metadata reaches actor-readable config.
    guard_cfg.update({k:v for k,v in host_cfg.items() if k!='private_group_config'})
    guard = output/'guard-config.json'
    guard.write_text(json.dumps(guard_cfg))
    os.environ['CUAWRIGHT_NATIVE_GUARD_CONFIG'] = str(guard)
    from confinement import install_hooks, restrict, owned_browser_members, terminate_recorded_members
    install_hooks()
    # Preload fixed transport before runtime-root directory discovery is denied.
    from native_group_transport import BrokerTransport
    abi = restrict(output, temporary, None)
    import cuawright.webwright.run.cli as native_cli
    adapter_metadata={'arm':args.arm,'native_query_adapter_enabled':False,'native_agent_query_unmodified':True,'upstream_source_files_modified':False,'scientific_protocol_intervention':None}
    if candidate_count>1:
        from native_group_transport import BrokerTransport
        from native_proposal_adapter import NativeProposalAdapter,install_agent_decision_hook
        transport=BrokerTransport(guard_cfg['browser_broker_socket'],str(task.get('task_id') or task.get('id')))
        adapter=NativeProposalAdapter(args.arm,Path(guard_cfg['broker_evidence'])/'private-proposals',transport,deadline_monotonic=deadline_monotonic,persist=transport.persist)
        original_get_agent=native_cli.get_agent
        def hooked_get_agent(*values,**kwargs):
            agent=original_get_agent(*values,**kwargs)
            adapter_metadata.update(install_agent_decision_hook(agent,adapter))
            return agent
        native_cli.get_agent=hooked_get_agent
    run_one=native_cli.run_one
    from cuawright.webwright.tools.browser_session import _close_persistent_session_files, _workspace_file, DEFAULT_PERSISTENT_SESSION_FILE, DEFAULT_SESSION_EVENTS_FILE
    import asyncio
    native = {}
    error = None
    cleanup = {}
    started = time.time()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Native episode received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        native = run_one(task=task.get('intent') or task.get('task') or task.get('instruction'), task_id=task.get('task_id') or task.get('id'), start_url=task.get('start_url') or task.get('url'), config_spec=[str(UPSTREAM/'src/cuawright/webwright/config/best_default_judge_json_persistent_cli.yaml'),str(UPSTREAM/'src/cuawright/webwright/config/model_openai.yaml'),str(runtime_config)], resolved_output_dir=output)
    except BaseException as exc:
        error = {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    finally:
        try:
            cleanup = asyncio.run(_close_persistent_session_files(session_path=_workspace_file(output,DEFAULT_PERSISTENT_SESSION_FILE), events_path=_workspace_file(output,DEFAULT_SESSION_EVENTS_FILE),delete_user_data=False,delete_session_file=False))
        except BaseException as exc:
            cleanup = {'success':False,'error':str(exc)}
        try:
            cleanup['broker_verification']=stop_broker(broker,guard_cfg)
        except BaseException as exc:
            cleanup['broker_verification']={'verified':False,'error':str(exc)}
        verification=cleanup.get('broker_verification',{})
        cleanup['process_verification']={'verified':verification.get('verified') is True and verification.get('exit_code')==0,'source':'physical-browser-broker','checks':verification.get('checks',[]),'evidence':verification.get('evidence')}
        trajectory = output/'trajectory.json'
        saved = json.loads(trajectory.read_text()) if trajectory.exists() else {}
        info = saved.get('info', {})
        if not native and saved.get('messages'):
            native = saved['messages'][-1].get('extra', {})
        final = str(native.get('final_response') or native.get('submission') or '')
        (output/'final-answer.json').write_text(json.dumps({'final_answer':final},indent=2)+'\n')
        result = {'native_result':native,'exit_status':native.get('exit_status') or info.get('exit_status') or (error or {}).get('type','unknown'),'final_response':final,'decisions':info.get('api_calls',0),'api_calls':info.get('api_calls',0),'error':error,'native_execution_complete':error is None,'valid':error is None,'task_latency_seconds':time.time()-started,'owned_teardown':cleanup,'landlock_abi':abi,'temporary_directory':str(temporary),'scientific_source_modified':candidate_count>1,'native_protocol':adapter_metadata,'runtime_workspace_listing_guard':workspace_listing_guard,'runtime_command_timeout_guard':command_timeout_guard,'runtime_child_source_binding':runtime_child_source_binding}
        (output/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps({'exit_status':result['exit_status'],'decisions':result['decisions'],'cleanup':cleanup},default=str))
    return 0 if error is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
