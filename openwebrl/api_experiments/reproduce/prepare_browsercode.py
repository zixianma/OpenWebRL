"""Assemble an isolated BrowserCode runtime from public sources; never call APIs.

Archived source bytes stay in api_experiments. Adaptations are written only to a
new output tree and recorded as before/after hashes in portable-build.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "portable_assets"
CAMPAIGNS = ("browsercode_luna_effort", "browsercode_claude", "browsercode_gpt_high", "browsercode_luna_single_high", "browsercode_gpt_high_wv_ds", "browsercode_sol61_pair20", "browsercode_sol61_pair80", "browsercode_luna_high100", "browsercode_luna_high100_initial", "browsercode_luna_high100_recovery4096", "browsercode_sol61_onecall20", "browsercode_sol61_onecall80", "browsercode_sol61_low_medium30")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def materializer():
    spec = importlib.util.spec_from_file_location("api_source_materializer", ROOT / "materialize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.materialize



def verify_dependencies(cuawright_source, browsercode_source, bun, chromium, node_modules):
    """Reject supplied overrides that change the pinned code or tool runtime."""
    spec = importlib.util.spec_from_file_location('portable_dependency_pins', Path(__file__).with_name('install.py'))
    pins = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pins)
    checkout = cuawright_source.parent
    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}
    head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], env=env, text=True).strip()
    expected = pins.PINS['cuawright_source']['commit']
    if head != expected:
        raise ValueError('CUAWright must be the pinned checkout ' + expected)
    clean = subprocess.run(['git', '-C', str(checkout), 'diff', '--quiet', expected, '--', 'src'], env=env)
    if clean.returncode:
        raise ValueError('CUAWright source differs from its pinned revision')
    manifest = json.loads((ASSETS / 'browsercode-source-manifest.json').read_text())
    for name, expected_hash in manifest['files'].items():
        if digest(browsercode_source / name) != expected_hash:
            raise ValueError('BrowserCode source differs from its frozen source: ' + name)
    # Playwright1.58.0 Linux Chromium1208 used in the recorded experiments.
    if digest(chromium) != '481fea1516a1f2b76454664272f12cd9dd1f20117b21e1f1498e08bc7f872c00':
        raise ValueError('Chromium binary differs from frozen Playwright1208 build')
    if digest(bun) != pins.BUN['binary_sha256']:
        raise ValueError('Bun binary differs from the pinned build')
    for package, version in pins.NPM.items():
        if json.loads((node_modules / package / 'package.json').read_text())['version'] != version:
            raise ValueError('JavaScript dependency differs from pinned version: ' + package)


def build(campaign, output, *, cuawright_source, browsercode_source, bun, chromium, node_modules, tasks_json=None):
    output = Path(output).resolve()
    if output.is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError("Use a writable project directory or /tmp")
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    cuawright_source, browsercode_source, bun, chromium, node_modules = map(
        lambda p: Path(p).resolve(), (cuawright_source, browsercode_source, bun, chromium, node_modules))
    if (cuawright_source / "src").is_dir():
        cuawright_source /= "src"
    for path in (cuawright_source / "cuawright/webwright/agents/default.py", browsercode_source / "packages/bcode-browser/src/browser-execute.ts", bun, chromium, node_modules / "effect/package.json"):
        if not path.is_file():
            raise ValueError(f"Missing dependency: {path}")
    verify_dependencies(cuawright_source, browsercode_source, bun, chromium, node_modules)
    spec_dir = ASSETS / "campaign_specs" / campaign
    config = json.loads((spec_dir / "runtime-config.json").read_text())
    plan = json.loads((spec_dir / "campaign-plan.json").read_text())
    protocol = plan.setdefault('protocol', dict(plan.get('actor_protocol', {})))
    protocol['max_decisions'] = protocol.get('max_decisions', protocol.get('max_steps', protocol.get('decision_cap', plan.get('decision_cap_per_episode'))))
    if 'arms' not in plan:
        if plan.get('policy') == 'onecall3_low':
            plan['arms'] = [dict(arm='onecall3_low', model=protocol['model'], proposal_count=3,
                reasoning_effort=protocol['reasoning_effort'], output_tokens=protocol['max_output_tokens'],
                selector_reasoning_effort=protocol['reasoning_effort'], selector_in_same_response=True)]
        else:
            cells = {cell['policy']: cell for cell in plan['cells']}
            plan['arms'] = [dict(arm=arm, model=protocol['actor'], proposal_count=1,
                reasoning_effort=cell['effort'], output_tokens=cell['max_output_tokens']) for arm, cell in cells.items()]
    if isinstance(plan['arms'][0], str):
        plan['arms'] = [dict(arm=arm,
            model=plan.get('arm_protocols', {}).get(arm, {}).get('actor_model', protocol.get('actor_model')),
            proposal_count=plan.get('arm_protocols', {}).get(arm, {}).get('candidates', protocol.get('candidate_counts', {}).get(arm)),
            reasoning_effort=protocol['actor_effort'], selector_reasoning_effort=protocol['selector_effort'],
            output_tokens=protocol['actor_max_output_tokens'], selector_output_tokens=protocol.get('selector_max_output_tokens')) for arm in plan['arms']]
    if tasks_json is None:
        # Uses only checked-in public benchmark data and the frozen ID roster.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tasks import load_tasks
        tasks = load_tasks(campaign)
    else:
        tasks_json = Path(tasks_json).resolve()
        tasks = json.loads(tasks_json.read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task file must be a nonempty normalized public benchmark list")
    for item in tasks:
        if not all(isinstance(item.get(k), str) and item[k] for k in ("task_id", "intent", "start_url")):
            raise ValueError("Each public task needs task_id, intent and start_url strings")
    materialize = materializer()
    materialize(campaign, output)
    adaptations = []
    def overlay(name, source, old, new, *, required=True):
        path = output / name
        text = source if isinstance(source, str) else source.read_text()
        count = text.count(old)
        if required and count != 1:
            raise ValueError(f"Overlay expected one match in {name}; got {count}")
        if count:
            after = text.replace(old, new)
            path.write_text(after)
            adaptations.append({"file": name, "reason": old, "original_sha256": hashlib.sha256(text.encode()).hexdigest(), "adapted_sha256": digest(path)})
    bridge_source = spec_dir / "browser_bridge.ts"
    shutil.copyfile(bridge_source if bridge_source.exists() else ASSETS / "browser_bridge.ts", output / "browser_bridge.ts")
    bridge_guard = "if (!workspace.startsWith('/gpfs/scrubbed/zixianma/openwebrl-runtime/')) throw new Error('Unexpected workspace');"
    overlay("browser_bridge.ts", output / "browser_bridge.ts", bridge_guard,
        "const allowedRoot=fs.realpathSync(process.env.OPENWEBRL_REPRO_OUTPUT!);\nconst actualWorkspace=fs.realpathSync(workspace);\nif (!actualWorkspace.startsWith(allowedRoot+'/')) throw new Error('Workspace outside isolated reproduction output');")
    overlay("sandbox.py", output / "sandbox.py", "'/gpfs/scrubbed/zixianma/openwebrl-runtime/venv'", repr(str(Path(sys.executable).resolve().parent.parent)))
    overlay("sandbox.py", output / "sandbox.py", "str(Path(chrome).parent)", "str(Path(chrome).resolve().parent)")
    (output / "sandbox.py").chmod(0o755)
    worker = output / "worker.py"
    if 'def stop(proc):' in (output / 'environments.py').read_text():
        # Historic workers use the older environment adapter. Add the same
        # identity-bound teardown already used by the newer frozen campaigns;
        # retain their original observations, model payloads and judge paths.
        ownership = (ROOT / 'browsercode/runtime/environments.py').read_text().split('def child_env(', 1)[0]
        (output / 'portable_processes.py').write_text(ownership)
        environment_path = output / 'environments.py'
        old_environment = environment_path.read_text()
        old_stop = old_environment[old_environment.index('def stop(proc):'):old_environment.index('def child_env(')]
        overlay('environments.py', environment_path, old_stop,
            'from portable_processes import stop,record_owned,cleanup_registered_groups\n\n')
        overlay('environments.py', environment_path,
            "  result=self.call({'op':'init','wsUrl':ws,'targetId':target},30)",
            "  record_owned(self.proc,workspace,'browsercode-bridge')\n  result=self.call({'op':'init','wsUrl':ws,'targetId':target},30)")
        overlay('environments.py', environment_path,
            "  deadline=time.monotonic()+45;port=profile/'DevToolsActivePort'",
            "  record_owned(self.proc,d,'chromium')\n  deadline=time.monotonic()+45;port=profile/'DevToolsActivePort'")
        overlay('worker.py', worker, "PLAN['plan_sha256']", "hashlib.sha256((ROOT/'campaign-plan.json').read_bytes()).hexdigest()", required=False)
    if campaign == 'browsercode_luna_single_high':
        original = worker.read_text()
        start = original.index('import importlib.util\n_spec=')
        end = original.index("if __name__=='__main__':main()", start)
        overlay('worker.py', worker, original[start:end],
            '# Portable runner retains the actor halt on all provider failures; private Slurm isolation is omitted.\n')
    if (spec_dir / 'native_judge_sources').is_dir():
        shutil.copytree(spec_dir / 'native_judge_sources', output / 'native_judge_sources')
    overlay("worker.py", worker, "from preparation_gate import validate\n", "# Private campaign registration is intentionally not imported by the portable episode runner.\n", required=False)
    if "'PYTHONPATH':" not in worker.read_text():
        overlay('worker.py', worker, "'WEBWRIGHT_BROWSER_BACKEND':'cdp'",
            "'WEBWRIGHT_BROWSER_BACKEND':'cdp','PYTHONPATH':os.pathsep.join([str(ROOT),CONFIG['cuawright_source'],CONFIG['frozen_support_source']]),'PYTHONDONTWRITEBYTECODE':'1'")
    config.update(cuawright_source=str(cuawright_source), frozen_support_source=str(ASSETS / "support"), chromium=str(chromium), python=sys.executable)
    config["portable_campaign"] = campaign
    config.pop("support_source_hashes", None)
    write_json(output / "runtime-config.json", config)
    write_json(output / "tasks.json", tasks)
    plan["task_manifest_sha256"] = digest(output / "tasks.json")
    write_json(output / "campaign-plan.json", plan)
    for name, target in (("browsercode", browsercode_source), ("node_modules", node_modules)):
        (output / name).symlink_to(target, target_is_directory=True)
    (output / "javascript-runtime").mkdir()
    (output / "javascript-runtime/bun").symlink_to(bun)
    # Include imported frozen support, prompts, bridge, and actual external code
    # in the same verification set as archived Python. No old ledgers copied.
    hashes = {str(p): digest(p) for p in output.rglob("*") if p.is_file() and not p.is_symlink() and p.name != "source-manifest.json"}
    for base in (ASSETS / "support", cuawright_source / "cuawright", browsercode_source / "packages/bcode-browser"):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".ts", ".yaml", ".json", ".md") and '__pycache__' not in path.parts:
                hashes[str(path)] = digest(path)
    hashes[str(bun)] = digest(bun)
    # chrome itself plus its installed directory remain visible in the manifest.
    hashes[str(chromium)] = digest(chromium)
    manifest = {"schema": "portable-browsercode-runtime-v1", "plan_sha256": digest(output / "campaign-plan.json"), "source_hashes": hashes}
    write_json(output / "source-manifest.json", manifest)
    shutil.copyfile(output / "source-manifest.json", output / "execution-source-manifest.json")
    write_json(output / "portable-build.json", {"campaign": campaign, "adaptations": adaptations, "archive_policy": "Original repository snapshots remain byte-identical; only this generated runtime is adapted.", "tasks_sha256": digest(output / "tasks.json"), "isolated_budgets": True})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, choices=CAMPAIGNS)
    parser.add_argument('--dependencies', type=Path, help='dependencies.json emitted by install.py')
    for name in ('output', 'cuawright-source', 'browsercode-source', 'bun', 'chromium', 'node-modules', 'tasks-json'):
        parser.add_argument('--' + name, type=Path, required=name == 'output')
    args = parser.parse_args()
    try:
        values = vars(args).copy()
        dependencies = values.pop('dependencies')
        if dependencies:
            installed = json.loads(dependencies.read_text())
            for name in ('cuawright_source', 'browsercode_source', 'bun', 'chromium', 'node_modules'):
                if values[name] is None:
                    values[name] = installed[name]
        missing = [name for name, value in values.items() if value is None and name != 'tasks_json']
        if missing:
            raise ValueError('Supply --dependencies or explicit paths: ' + ', '.join(missing))
        target = build(**values)
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"Prepared {target}; no browser or API calls started.")


if __name__ == '__main__':
    main()
