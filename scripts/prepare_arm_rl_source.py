#!/usr/bin/env python3
"""Isolate the ARM mechanics implementation on the preserved RL recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from evaluate_baseline_checkpoint import REPO, RUNTIME
from resume_baseline import validate_source, write_json


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_source(source)
    if output.exists() or not output.is_relative_to(RUNTIME):
        raise ValueError('Use a new isolated runtime source')
    core = ['slime/backends/megatron_utils/actor.py', 'slime/backends/megatron_utils/model.py',
            'slime/backends/megatron_utils/loss.py', 'slime/rollout/sglang_rollout.py']
    patch = subprocess.check_output(['git', 'diff', '--', *core], cwd=REPO, text=True)
    if not patch or 'arm_rl_megatron' not in patch or 'arm_rl_runtime' not in patch:
        raise ValueError('Expected ARM-only core changes are missing; inspect source changes')
    shutil.copytree(source, output, symlinks=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    command = ['patch', '--batch', '--forward', '-p1', '-d', str(output)]
    subprocess.run([*command, '--dry-run'], input=patch, text=True, check=True, capture_output=True)
    applied = subprocess.run(command, input=patch, text=True, check=True, capture_output=True)
    copied = ['openwebrl/arm_rl.py', 'openwebrl/arm_rl_collection.py', 'openwebrl/arm_rl_megatron.py',
              'openwebrl/arm_rl_runtime.py', 'openwebrl/arm_inference.py', 'scripts/audit_arm_preference_pairs.py',
              'scripts/serve_arm.py']
    for name in copied:
        shutil.copyfile(REPO / name, output / name)
    # Only add the ordinary-format selector hook to the preserved browser driver.
    name = 'openwebrl/generate_browser.py'
    text = (output / name).read_text()
    start = text.index('            # 7. Run inference ---------------------------------------------',
                       text.index('async def _generate_turn_sample_impl'))
    end = text.index('            # --------------------------------------------------------------', start)
    working = (REPO / name).read_text()
    a = working.index('            # 7. Run inference. The optional selector')
    b = working.index('            # --------------------------------------------------------------', a)
    replacement = working[a:b].replace('turn_sampling_params', 'sampling_params')
    (output / name).write_text(text[:start] + replacement + text[end:])
    files = [*core, *copied, name]
    manifest = json.loads((output / 'reference_manifest.json').read_text())
    hashes = {n: hashlib.sha256((output/n).read_bytes()).hexdigest() for n in files}
    manifest['recipe_files_sha256'].update(hashes)
    manifest['arm_rl_mechanics'] = dict(parent_source=str(source), changed_files_sha256=hashes,
        core_patch_sha256=hashlib.sha256(patch.encode()).hexdigest(), patch_output=applied.stdout,
        supported_topology='TP2, DP1, CP1, PP1, microbatch1', gpu_validated=False,
        baseline_pointer_modified=False, scientific_training_requires_gradient_calibration=True)
    (output / 'arm_core.patch').write_text(patch)
    write_json(output / 'reference_manifest.json', manifest)
    validate_source(output)
    return manifest['arm_rl_mechanics']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output), indent=2))
