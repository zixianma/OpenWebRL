"""Expose this runtime's frozen support when the public package is loaded.

The module entry point has already imported the repository's ``openwebrl``.
Adding a directory to sys.path alone cannot extend that package's search path.
"""
import importlib
import hashlib
import json
from pathlib import Path
import sys


def load(runtime):
    runtime = Path(runtime).resolve()
    names = {path.stem for path in runtime.glob("*.py")} | {"bootstrap", "worker", "environments"}
    for name in sorted(names):
        cached = sys.modules.get(name)
        if cached is not None:
            original = getattr(cached, "__file__", None)
            if original is None or Path(original).resolve() != runtime / (name + ".py"):
                raise ValueError("Different runtime module already loaded; use a fresh process: " + name)
    config = json.loads((runtime / "runtime-config.json").read_text())
    support = config.get("frozen_support_source")
    if support is not None:
        support = Path(support).resolve()
        package_root = support / "openwebrl"
        if not (package_root / "__init__.py").is_file():
            raise ValueError("Runtime frozen support package is missing")
        for path in package_root.rglob("*.py"):
            if path.name == "__init__.py":
                continue
            name = "openwebrl." + ".".join(path.relative_to(package_root).with_suffix("").parts)
            cached = sys.modules.get(name)
            if cached is not None:
                original = getattr(cached, "__file__", None)
                if (original is None or Path(original).resolve() != path.resolve()
                        or hashlib.sha256(Path(original).read_bytes()).digest() != hashlib.sha256(path.read_bytes()).digest()):
                    raise ValueError("Different frozen support module already loaded; use a fresh process: " + name)
        sys.path.insert(0, str(support))
        package = importlib.import_module("openwebrl")
        package.__path__ = [str(package_root), *[p for p in package.__path__ if p != str(package_root)]]
        # The frozen eval directory is a namespace package. An existing regular
        # repository package otherwise wins even after extending the root.
        for directory in sorted(package_root.iterdir()):
            if directory.is_dir() and any(directory.glob("*.py")):
                child = importlib.import_module("openwebrl." + directory.name)
                child.__path__ = [str(directory), *[p for p in child.__path__ if p != str(directory)]]
    sys.path.insert(0, str(runtime))
    return importlib.import_module("worker"), importlib.import_module("environments")
