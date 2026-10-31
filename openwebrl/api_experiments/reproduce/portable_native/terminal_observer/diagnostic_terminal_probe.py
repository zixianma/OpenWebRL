"""Synthetic, ungraded explicit-close observer probe; invoke inside approved job.

This module never allocates resources and never invokes model/judge APIs. The
existing diagnostic's metered physical-browser reservation remains mandatory.
The controller must verify the private receipt after its confined child exits.
"""
import hashlib
import json
from pathlib import Path
import shlex
import struct
import sys
import time
from trusted_terminal_capture import request_capture, request_remember_session


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _execute(environment, command):
    result = environment.execute({'command': command, 'bash_command': command})
    observation = result.get('observation', {})
    if not observation.get('success'):
        raise ValueError('Synthetic terminal probe command failed: ' + str(observation.get('command_output', ''))[-2000:])
    return observation


def exercise(environment, output, cfg):
    """Confined native-owner child: real blank step, default close, immediate RPC."""
    output = Path(output)
    code = output / 'synthetic-terminal-probe.py'
    code.write_text("await page.set_content('<html><body><h1>SYNTHETIC UNGRADED TERMINAL OBSERVER DIAGNOSTIC</h1></body></html>')\n")
    step_command = shlex.join([sys.executable, '-m', 'cuawright.webwright.tools.browser_session',
                              'step', '--workspace-dir', str(output), '--action',
                              'Synthetic ungraded blank-page terminal observer check', '--code-file', str(code),
                              '--screenshot', 'none', '--settle-ms', '0'])
    _execute(environment, step_command)
    steps = [json.loads(line) for line in (output / 'browser-steps.jsonl').read_text().split('\n') if line.strip()]
    if len(steps) != 1 or steps[0].get('screenshot_path'):
        raise ValueError('Probe must exercise one actual step without a saved end-of-step frame')
    original = json.loads((output / '.browser-session.json').read_text())
    remembered = request_remember_session(cfg)
    (output / 'trajectory.json').write_text(json.dumps({'synthetic_fixture_only': True,
        'scientific_result_eligible': False, 'messages': [],
        'info': {'submission': 'Synthetic diagnostic only; no benchmark grading.'}}) + '\n')
    close_command = shlex.join([sys.executable, '-m', 'cuawright.webwright.tools.browser_session',
                               'close', '--workspace-dir', str(output)])
    _execute(environment, close_command)
    requested = time.time()
    capture = request_capture(cfg, requested)
    if (output / '.browser-session.json').exists():
        raise ValueError('Default CLI close should remove its session manifest')
    events = [json.loads(line) for line in (output / 'browser-sessions.jsonl').read_text().split('\n') if line.strip()]
    if not events or events[-1]['event'] != 'closed' or events[-1]['session_epoch'] != original['session_epoch'] or events[-1]['session_id'] != original['session_id']:
        raise ValueError('Synthetic close did not preserve exact final lifecycle identity')
    record = {'synthetic_fixture_only': True, 'scientific_result_eligible': False,
              'actor_calls': 0, 'judge_calls': 0, 'new_browser_starts': 1,
              'selected_browser_steps': 1, 'step_had_no_screenshot': True,
              'default_close_deleted_session_manifest': True,
              'original_session_epoch': original['session_epoch'], 'original_session_id': original['session_id'],
              'remembered': remembered, 'terminal_capture': capture,
              'immediate_final_request_epoch': requested,
              'closed_lifecycle_sha256': sha(output / 'browser-sessions.jsonl')}
    (output / 'terminal-probe.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def verify(output):
    """Unconfined controller, after worker.invoke has independently reaped child."""
    output = Path(output).resolve()
    record = json.loads((output / 'terminal-probe.json').read_text())
    if record.get('synthetic_fixture_only') is not True or record.get('scientific_result_eligible') is not False or record.get('actor_calls') != 0 or record.get('judge_calls') != 0:
        raise ValueError('Probe cannot become a scientific task or invoke paid APIs')
    parent = json.loads((output / 'parent-owned-teardown.json').read_text())
    startup = json.loads((output / 'startup-diagnostic.json').read_text())
    if parent.get('verified') is not True or startup.get('actor_release_permitted') is not True:
        raise ValueError('Probe owned teardown/startup gate not independently verified')
    proof = record['terminal_capture']
    receipt_path = Path(proof['receipt_path']).resolve()
    broker_evidence = Path(startup['owned_teardown']['broker_verification']['evidence']).resolve()
    if receipt_path.parent != broker_evidence or receipt_path.is_symlink() or sha(receipt_path) != proof['receipt_sha256']:
        raise ValueError('Probe terminal receipt not bound to owned private broker evidence')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('verified') is not True or receipt['episode'] != str(output) or receipt['source_files_sha256'].get('.browser-session.json', 'missing') is not None:
        raise ValueError('Probe did not qualify the actual deleted terminal session')
    details = receipt['capture']
    if not all(details.get(key) is True for key in ('session_epoch_verified', 'no_later_browser_action_verified', 'no_later_owned_browser_launch_verified')):
        raise ValueError('Probe closed endpoint identity checks missing')
    provisional = json.loads(Path(details['preclose_receipt_path']).read_text())
    if sha(details['preclose_receipt_path']) != details['preclose_receipt_sha256'] or provisional['session_epoch'] != record['original_session_epoch'] or provisional['session_id'] != record['original_session_id']:
        raise ValueError('Probe terminal epoch/session proof differs')
    image = Path(receipt['screenshot_path'])
    if image.is_symlink() or image.resolve().parent != broker_evidence or sha(image) != receipt['screenshot_sha256'] or sha(image) != proof['screenshot_sha256']:
        raise ValueError('Probe raw PNG provenance differs')
    raw = image.read_bytes()
    if raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[12:16] != b'IHDR' or list(struct.unpack('>II', raw[16:24])) != receipt['screenshot_dimensions']:
        raise ValueError('Probe endpoint PNG geometry differs')
    for name, expected in receipt['source_files_sha256'].items():
        path = output / name
        if expected is None:
            if path.exists():
                raise ValueError('Deleted probe manifest reappeared')
        elif sha(path) != expected:
            raise ValueError('Probe selected actions/lifecycle changed')
    if receipt['source_files_sha256']['browser-sessions.jsonl'] != record['closed_lifecycle_sha256']:
        raise ValueError('Probe closed lifecycle hash differs')
    return {'verified': True, 'synthetic_fixture_only': True, 'scientific_result_eligible': False,
            'actor_calls': 0, 'judge_calls': 0, 'receipt_sha256': proof['receipt_sha256'],
            'screenshot_sha256': proof['screenshot_sha256'], 'owned_teardown_verified': True}
