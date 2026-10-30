"""Install a pinned, CPU-only API experiment environment in a new directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parent
PINS = {
    "cuawright_source": {"url": "https://github.com/microsoft/CUAWright.git", "commit": "ff44a0524e0812b69814efcec5e583f0d8cdde0c", "paths": ["src", "licenses"]},
    "browsercode_source": {"url": "https://github.com/browser-use/browsercode.git", "commit": "b200643fe0eb93ce5a3aadd4bb21cb6cceb2ed04", "paths": ["packages/bcode-browser"]},
    "official_webjudge": {"url": "https://github.com/OSU-NLP-Group/Online-Mind2Web.git", "commit": "f0d805ee0e9e0b3ea70911e45e5264b72968f3dc", "paths": ["src"]},
}
BUN = {"url": "https://github.com/oven-sh/bun/releases/download/bun-v1.4.3/bun-linux-x64-baseline.zip", "version": "1.4.3", "archive_sha256": "1fc2edac843102909e3a1be1d8d9802cc6071cf074e67e88231f4ffe0f8b397b", "binary_sha256": "7ce7d6b654eddeec20afddf2396317fba6dc19ab52788a27e894e1806bf8f24b"}
NPM = {"effect": "4.0.0-beta.83", "zod": "4.1.8"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args, *, env, cwd=None):
    subprocess.run([str(value) for value in args], env=env, cwd=cwd, check=True)


def checkout(pin, output, env):
    """Fetch one immutable commit; a partial failed install can be retried."""
    if not output.exists():
        output.mkdir(parents=True)
        run(["git", "init", "--quiet", output], env=env)
        run(["git", "-C", output, "remote", "add", "origin", pin["url"]], env=env)
        run(["git", "-C", output, "sparse-checkout", "init", "--cone"], env=env)
        run(["git", "-C", output, "sparse-checkout", "set", *pin["paths"]], env=env)
    origin = subprocess.check_output(["git", "-C", str(output), "remote", "get-url", "origin"], text=True, env=env).strip()
    if origin != pin["url"]:
        raise ValueError("Unexpected existing dependency checkout: " + str(output))
    head = subprocess.run(["git", "-C", str(output), "rev-parse", "HEAD"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if head.returncode or head.stdout.strip() != pin["commit"]:
        run(["git", "-C", output, "fetch", "--filter=blob:none", "--depth", "1", "origin", pin["commit"]], env=env)
        run(["git", "-C", output, "checkout", "--quiet", "--detach", "FETCH_HEAD"], env=env)
    actual = subprocess.check_output(["git", "-C", str(output), "rev-parse", "HEAD"], text=True, env=env).strip()
    if actual != pin["commit"]:
        raise ValueError("Dependency commit mismatch: " + str(output))
    dirty = subprocess.check_output(["git", "-C", str(output), "status", "--porcelain", "--untracked-files=no"], text=True, env=env)
    if dirty:
        raise ValueError("Dependency checkout contains modified tracked source: " + str(output))


def install(prefix, *, legacy=False):
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Run the installer with Python 3.12 (the executed runtime version)")
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise ValueError("This pinned browser/runtime profile requires Linux x86_64")
    prefix = prefix.resolve()
    if prefix.is_relative_to(Path("/gpfs/home/zixianma")):
        raise ValueError("Select a writable project or temporary installation prefix")
    marker = prefix / "installation-plan.json"
    plan = {"schema": "openwebrl-api-dependencies-v1", "python": "3.12", "sources": PINS, "bun": BUN, "npm": NPM, "requirements_sha256": digest(ROOT / "requirements.txt"), "bun_lock_sha256": digest(ROOT / "bun.lock"), "legacy_requirements_sha256": digest(ROOT / "requirements-legacy.txt") if legacy else None}
    if prefix.exists() and not marker.is_file():
        raise ValueError("Prefix must be new or an earlier install.py prefix")
    prefix.mkdir(parents=True, exist_ok=True)
    if marker.exists() and json.loads(marker.read_text()) != plan:
        previous = json.loads(marker.read_text())
        identity = lambda sources: {key: (value["url"], value["commit"]) for key, value in sources.items()}
        if identity(previous["sources"]) != identity(PINS):
            raise ValueError("Existing installation has different source pins; choose a new prefix")
        # Resume an interrupted install or add the optional legacy CPU profile.
        # All package versions are pinned and pip checks their consistency.
    marker.write_text(json.dumps(plan, indent=2) + "\n")
    env = os.environ.copy()
    for name in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "PIP_CACHE_DIR", "BUN_INSTALL_CACHE_DIR", "MPLCONFIGDIR", "HF_HOME", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        path = prefix / "cache" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1", GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    environment = prefix / "venv"
    if not (environment / "bin/python").is_file():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    run([python, "-m", "pip", "install", "--requirement", ROOT / "requirements.txt"], env=env)
    if legacy:
        run([python, "-m", "pip", "install", "--no-deps", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.9.1+cpu", "torchvision==0.24.1+cpu"], env=env)
        run([python, "-m", "pip", "install", "--constraint", ROOT / "requirements.txt", "--requirement", ROOT / "requirements-legacy.txt"], env=env)
    sources = {}
    for key, pin in PINS.items():
        destination = prefix / "sources" / key
        checkout(pin, destination, env)
        sources[key] = str(destination)
    # No editable installation or upstream dependency resolver is needed: source
    # builders copy only the pinned import trees into their isolated runtimes.
    bun = prefix / "bin/bun"
    if not bun.is_file():
        archive = prefix / "bun.zip"
        urllib.request.urlretrieve(BUN["url"], archive)
        if digest(archive) != BUN["archive_sha256"]:
            raise ValueError("Bun release archive hash differs from the executed binary")
        with zipfile.ZipFile(archive) as package:
            entries = [entry for entry in package.infolist() if not entry.is_dir() and Path(entry.filename).name == "bun"]
            if len(entries) != 1:
                raise ValueError("Unexpected Bun archive layout")
            bun.parent.mkdir(parents=True, exist_ok=True)
            with bun.open("xb") as stream:
                stream.write(package.read(entries[0]))
            bun.chmod(0o755)
    if digest(bun) != BUN["binary_sha256"]:
        raise ValueError("Bun executable hash mismatch")
    javascript = prefix / "javascript"
    javascript.mkdir(exist_ok=True)
    (javascript / "package.json").write_text(json.dumps({"private": True, "type": "module", "dependencies": NPM}, indent=2) + "\n")
    (javascript / "bun.lock").write_bytes((ROOT / "bun.lock").read_bytes())
    run([bun, "install", "--frozen-lockfile", "--ignore-scripts"], env=env, cwd=javascript)
    source_modules = prefix / "sources/node_modules"
    if not source_modules.exists():
        source_modules.symlink_to(javascript / "node_modules", target_is_directory=True)
    browser_cache = prefix / "browsers"
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_cache)
    run([python, "-m", "playwright", "install", "chromium"], env=env)
    executable = subprocess.check_output([str(python), "-c", "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); print(p.chromium.executable_path); p.stop()"], env=env, text=True).strip()
    chromium = Path(executable)
    if not chromium.is_file():
        raise ValueError("Pinned Chromium executable missing after installation")
    output = {**sources, "cuawright": sources["cuawright_source"], "python": str(python), "bun": str(bun), "chromium": str(chromium), "node_modules": str(javascript / "node_modules"), "browser_cache": str(browser_cache), "pins": plan}
    if legacy:
        tokenizer = prefix / "tokenizer"
        snippet = "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-VL-4B-Thinking',revision='1de27d8c51f12e819435303b9e84c4e25ba8401e',allow_patterns=['*.json','*.txt','*.model','*.jinja'],local_dir=" + repr(str(tokenizer)) + ")"
        run([python, "-c", snippet], env=env)
        output["tokenizer_source"] = str(tokenizer)
        output["tokenizer_files_sha256"] = {str(path.relative_to(tokenizer)): digest(path) for path in tokenizer.rglob("*") if path.is_file() and ".cache" not in path.parts}
    run([python, "-m", "pip", "check"], env=env)
    (prefix / "dependencies.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"dependencies": str(prefix / "dependencies.json"), "python": str(python)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--install", action="store_true", help="Create the environment and download pinned dependencies; default prints the plan")
    parser.add_argument("--legacy", action="store_true", help="Also install CPU vision/tokenizer support for historical coordinate-browser experiments; no model weights")
    args = parser.parse_args()
    if not args.install:
        print(json.dumps({"prefix": str(args.prefix.resolve()), "sources": PINS, "bun": BUN, "npm": NPM, "requirements": str(ROOT / "requirements.txt"), "legacy_requirements": str(ROOT / "requirements-legacy.txt") if args.legacy else None, "install": False}, indent=2))
        return
    try:
        install(args.prefix, legacy=args.legacy)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
