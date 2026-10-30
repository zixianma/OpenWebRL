"""Fail closed if inherited native subprocess mediation cannot be installed."""
import os
import sys

if os.environ.get('CUAWRIGHT_NATIVE_GUARD_CONFIG'):
    try:
        import importlib.util
        from pathlib import Path
        # Fixed immutable path avoids Python FileFinder listing runtime root.
        path=Path(__file__).resolve().parents[1]/'confinement.py'
        spec=importlib.util.spec_from_file_location('confinement',path)
        module=importlib.util.module_from_spec(spec)
        sys.modules['confinement']=module
        spec.loader.exec_module(module)
        module.install_hooks()
    except BaseException as exc:
        sys.stderr.write('Native guard initialization failed: '+str(exc)+'\n')
        sys.stderr.flush()
        os._exit(126)
