#!/usr/bin/python3
"""Inherited filesystem confinement for generated commands and owned Chromium."""
import ctypes,os,re,sys,json,tempfile
from pathlib import Path

def restrict(workspace,executables):
 libc=ctypes.CDLL(None,use_errno=True)
 def call(n,*args):
  v=libc.syscall(n,*args)
  if v<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
  return v
 abi=call(444,0,0,1)
 if abi<3:raise RuntimeError('Landlock ABI>=3 required; never execute without write confinement')
 # Execute plus every mutation supported through ABI3, including rename and truncate.
 handled=1|sum(1<<x for x in range(1,15) if x not in (2,3))
 class Rules(ctypes.Structure):_fields_=[('access',ctypes.c_uint64)]
 class Beneath(ctypes.Structure):_pack_=1;_fields_=[('access',ctypes.c_uint64),('parent',ctypes.c_int32)]
 attr=Rules(handled);fd=call(444,ctypes.byref(attr),ctypes.sizeof(attr),0)
 def allow(path,access):
  p=Path(path).resolve()
  if not p.exists():return
  f=os.open(p,os.O_PATH|os.O_CLOEXEC)
  try:
   rule=Beneath(access,f);call(445,fd,1,ctypes.byref(rule),0)
  finally:os.close(f)
 workspace=Path(workspace).resolve()
 if str(workspace).startswith('/gpfs/home/zixianma'):raise ValueError('Forbidden writable root')
 allow(workspace,handled & ~1)
 # Character devices require write-open permission but are not filesystem storage.
 allow('/dev/null',1<<1)
 allow(os.environ['TMPDIR'],handled & ~1)
 for path in executables:allow(path,1)
 if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'no_new_privs')
 call(446,fd,0);os.close(fd)

def main():
 workspace=os.environ['WORKSPACE_DIR'];chrome=os.environ.get('CUA_CHROME_EXECUTABLE','')
 browser=sys.argv[1:2]==['--browser'];argv=sys.argv[2:] if browser else ['/bin/bash','--noprofile','--norc',*sys.argv[1:]]
 if not browser:
  command=sys.argv[-1] if sys.argv else ''
  if re.search(r'\brm\b',command):raise ValueError('rm commands require separate exact user approval; not permitted by this pilot')
 for k in ['BASH_ENV','ENV']:os.environ.pop(k,None)
 os.environ['NO_PROXY']='127.0.0.1,localhost,::1'
 os.environ['no_proxy']=os.environ['NO_PROXY']
 for k in list(os.environ):
  if any(x in k.upper() for x in ['API_KEY','TOKEN','SECRET','CREDENTIAL']):os.environ.pop(k,None)
 marker=Path(workspace)/'.private-temp.json'
 if marker.exists():
  temporary=Path(json.loads(marker.read_text())['path'])
  if not str(temporary).startswith('/tmp/cua-') or temporary.stat().st_uid!=os.getuid() or temporary.stat().st_mode & 0o077:raise ValueError('Unsafe private temporary directory')
 else:
  temporary=Path(tempfile.mkdtemp(prefix='cua-',dir='/tmp'));marker.write_text(json.dumps({'path':str(temporary)}))
 os.environ['TMPDIR']=str(temporary)
 for k in ['XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','PYTHONPYCACHEPREFIX']:
  p=Path(workspace)/'.cache'/k.lower();p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
 allowed=['/usr','/bin','/lib','/lib64','/gpfs/scrubbed/zixianma/openwebrl-runtime/venv']
 if browser:allowed.append(str(Path(chrome).parent))
 restrict(workspace,allowed)
 os.execvpe(argv[0],argv,os.environ)
if __name__=='__main__':main()
