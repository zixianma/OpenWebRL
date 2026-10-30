#!/usr/bin/env python3
"""Prepare isolated ARM calibration and gated first-batch training; never submit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from evaluate_baseline_checkpoint import REPO, RUNTIME
from resume_baseline import validate_source, write_json

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source anchor changed or is ambiguous: ' + old[:100])
    return text.replace(old, new, 1)


def copy_plain(source, destination):
    # Avoid GPFS sendfile stalls when freezing these small Python source files.
    Path(destination).write_bytes(Path(source).read_bytes())
    shutil.copymode(source, destination)
    return destination


def prepare(source, output, multi_cycle=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_source(source)
    if output.exists() or not output.is_relative_to(RUNTIME):
        raise ValueError('Use a new isolated runtime source')
    shutil.copytree(source, output, symlinks=True,
        copy_function=copy_plain,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
    copied = ['openwebrl/arm_turn_bonus.py', 'openwebrl/arm_turn_bonus_runtime.py', 'openwebrl/arm_inference.py',
              'scripts/audit_arm_preference_pairs.py', 'scripts/serve_arm.py']
    if multi_cycle:
        copied.append('openwebrl/arm_turn_bonus_cycles.py')
    for name in copied:
        copy_plain(REPO / name, output / name)
    name = 'openwebrl/generate_browser.py'
    text = (output / name).read_text()
    start = text.index('            # 7. Run inference ---------------------------------------------',
                       text.index('async def _generate_turn_sample_impl'))
    end = text.index('            # --------------------------------------------------------------', start)
    working = (REPO / name).read_text()
    a = working.index('            # 7. Run inference. The optional selector')
    b = working.index('            # --------------------------------------------------------------', a)
    block = working[a:b].replace('turn_sampling_params', 'sampling_params')
    (output / name).write_text(text[:start] + block + text[end:])

    # Attach a report after the native trajectory-level normalization; no reimplementation.
    name = 'slime/ray/rollout.py'
    text = (output / name).read_text()
    old = ('                normalized_rewards = [normalized_rewards_by_index[id(sample)] for sample in samples]\n'
           '                return raw_rewards, normalized_rewards')
    new = old.replace('                return raw_rewards, normalized_rewards',
        '                if os.environ.get("OPENWEBRL_ARM_TURN_BONUS_CONFIG"):\n'
        '                    from openwebrl.arm_turn_bonus import post_process_rewards\n'
        '                    return post_process_rewards(self.args, samples, raw_rewards, normalized_rewards)\n'
        '                return raw_rewards, normalized_rewards')
    text = replace_once(text, old, new)
    old = '            self.data_source.get_samples(int(os.environ.get("OPENWEBRL_REPLAY_CONSUMED_GROUPS", "144")))\n'
    new = ('            if os.environ.get("OPENWEBRL_ARM_REPLAY_CURSOR"):\n'
           '                from openwebrl.arm_turn_bonus_runtime import load_replay_cursor\n'
           '                load_replay_cursor(self.data_source, rollout_id)\n'
           '            else:\n'
           '    ' + old)
    text = replace_once(text,old,new)
    old = '        data = self._convert_samples_to_train_data(data)\n'
    new = old + ('        if os.environ.get("OPENWEBRL_ARM_TURN_BONUS_CONFIG"):\n'
                 '            self.data_source.save(rollout_id)\n')
    text = replace_once(text, old, new)
    if multi_cycle:
        text = replace_once(text, '        self.rollout_id = rollout_id\n',
            '        self.rollout_id = rollout_id\n'
            '        from openwebrl.arm_turn_bonus_cycles import reset_collection\n'
            '        reset_collection(self.args, rollout_id)\n')
    (output / name).write_text(text)

    name = 'train.py'
    text = (output / name).read_text()
    old = ('            if args.offload_rollout:\n'
           '                _ray_get_with_actor_retry(rollout_manager.offload.remote(), label="rollout_manager.offload")\n')
    new = ('            if os.environ.get("OPENWEBRL_ARM_TURN_BONUS_CONFIG"):\n'
           '                from openwebrl.arm_turn_bonus_runtime import before_optimizer\n'
           '                stop_after_calibration = before_optimizer(args, rollout_id)\n'
           '                _log_train_progress(rollout_id, args.num_rollout, "arm_calibration_complete")\n'
           '                if stop_after_calibration:\n'
           '                    break\n\n' + old)
    text = replace_once(text, old, new)
    if multi_cycle:
        text = replace_once(text, '            _log_train_progress(rollout_id, args.num_rollout, "generate")\n',
            '            from openwebrl.arm_turn_bonus_cycles import before_collection, before_training, after_checkpoint\n'
            '            if before_collection(args, rollout_id):\n'
            '                break\n'
            '            _log_train_progress(rollout_id, args.num_rollout, "generate")\n')
        text = replace_once(text, '                if stop_after_calibration:\n                    break\n',
            '                if stop_after_calibration:\n                    break\n'
            '                before_training()\n')
        text = replace_once(text, '                save(rollout_id)\n',
            '                save(rollout_id)\n'
            '                if after_checkpoint(args, rollout_id):\n'
            '                    break\n')
    (output / name).write_text(text)

    name = 'slime/rollout/sglang_rollout.py'
    text = (output/name).read_text()
    old = '            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)\n'
    new = old + ('            if os.environ.get("OPENWEBRL_ARM_TURN_BONUS_CONFIG"):\n'
                 '                from openwebrl.arm_turn_bonus_runtime import record_completed_group\n'
                 '                await asyncio.to_thread(record_completed_group, args, rollout_id, group,\n'
                 '                    dynamic_filter_output, len(data) if len(data) < target_data_size else None)\n')
    (output/name).write_text(replace_once(text,old,new))
    changed = [*copied, 'openwebrl/generate_browser.py', 'slime/ray/rollout.py', 'train.py',name]
    hashes = {n: hashlib.sha256((output/n).read_bytes()).hexdigest() for n in changed}
    manifest = json.loads((output / 'reference_manifest.json').read_text())
    manifest['recipe_files_sha256'].update(hashes)
    manifest['arm_turn_bonus_calibration'] = dict(parent_source=str(source),
        changed_files_sha256=hashes, gpu_validated=False, optimizer_updates=0,
        actor_and_loss_backends_unchanged=True, baseline_pointer_modified=False,
        supports_gated_first_batch_training=True, incremental_outcome_groups=True,
        supports_multiple_collections=multi_cycle)
    write_json(output / 'reference_manifest.json', manifest)
    for path in ('slime/backends/megatron_utils/actor.py', 'slime/backends/megatron_utils/model.py',
                 'slime/backends/megatron_utils/loss.py'):
        if (output/path).read_bytes() != (source/path).read_bytes():
            raise ValueError('Unexpected change to native backend/collector')
    validate_source(output)
    return manifest['arm_turn_bonus_calibration']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--multi-cycle', action='store_true')
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output, a.multi_cycle), indent=2))
