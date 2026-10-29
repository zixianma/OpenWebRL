"""Host-side broker lifecycle, called before actor confinement is installed."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def start_broker(cfg):
    directory=Path(tempfile.mkdtemp(prefix='cua-broker-',dir='/tmp'))
    cfg.update({'physical_chromium':'/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome','browser_launcher':str(Path(__file__).with_name('browser_launcher.py')),'browser_broker_socket':str(directory/'broker.sock'),'broker_ready':str(directory/'ready.json'),'broker_evidence':str(directory/'evidence')})
    evidence_root=Path(cfg['output']).parent/'.broker-evidence'
    evidence_root.mkdir(exist_ok=True)
    cfg['broker_evidence']=tempfile.mkdtemp(prefix=Path(cfg['output']).name+'-',dir=str(evidence_root))
    path=directory/'config.json';path.write_text(json.dumps(cfg))
    log=(directory/'broker.log').open('w')
    env=dict(os.environ);env.pop('CUAWRIGHT_NATIVE_GUARD_CONFIG',None)
    process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('browser_broker.py')),'--config',str(path)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    deadline=time.monotonic()+30
    while not Path(cfg['broker_ready']).exists():
        if process.poll() is not None or time.monotonic()>deadline:
            if process.poll() is None:
                process.terminate();process.wait(timeout=10)
            raise RuntimeError('Browser broker failed to start; log='+str(directory/'broker.log'))
        time.sleep(.05)
    return process


def stop_broker(process,cfg):
    connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    connection.settimeout(20)
    try:
        connection.connect(cfg['browser_broker_socket'])
        connection.sendall(b'{"op":"shutdown"}\n')
        result=json.loads(connection.makefile('rb').readline())
    finally:
        connection.close()
    process.wait(timeout=20)
    return {**result,'exit_code':process.returncode,'evidence':cfg['broker_evidence']}
