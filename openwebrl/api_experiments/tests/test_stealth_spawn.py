"""Import the actual fresh stealth worker under spawn, stopping before SDKs."""
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]


def guarded_worker(arm, plan, control_root, queue):
    from unittest.mock import patch
    import socket
    from openai import OpenAI
    from openwebrl.api_experiments.reproduce import stealth_study as study, common, cloud_browser
    def forbidden(*args, **kwargs):
        raise AssertionError('No provider, SDK, scheduler or browser launch is allowed')
    def inspect(args, actual, tasks, root, keys, actor_key, cloud, worker, environments, setup):
        runtime = Path(plan['runtime'])
        assert Path(worker.__file__).resolve() == runtime / 'worker.py'
        assert Path(environments.__file__).resolve() == runtime / 'environments.py'
        assert len(tasks) == 150 and [row['task_id'] for row in tasks] == plan['task_ids']
        assert actual['arm'] == arm and actual['candidate_count'] == (1 if arm == study.ARMS[0] else 5)
        assert actual['max_output_tokens'] == 12288 and actual['max_decisions'] == 30
        assert cloud['proxy_country'] == 'us' and setup is not None
        assert worker.live_page.__module__ == '_openwebrl_browser_lifecycle'
        assert environments.live_page.__module__ == 'portable_browser_lifecycle'
        support = Path(json.loads((runtime / 'runtime-config.json').read_text())['frozen_support_source']) / 'openwebrl'
        for name, module in tuple(sys.modules.items()):
            candidate = support.joinpath(*name.split('.')[1:]).with_suffix('.py')
            if name.startswith('openwebrl.') and candidate.is_file():
                assert Path(module.__file__).resolve() == candidate.resolve(), name
        assert not (root / 'budget').exists()
        queue.put(dict(arm=arm, actual_frozen_worker_imported=True, tasks=150, provider_calls=0))
        return 0
    try:
        with patch.object(common, '_execute_worker', side_effect=inspect), \
                patch.object(OpenAI, '__init__', side_effect=forbidden), \
                patch.object(cloud_browser.V4HTTP, '__init__', side_effect=forbidden), \
                patch.object(socket.socket, 'connect', side_effect=forbidden), \
                patch.object(socket.socket, 'connect_ex', side_effect=forbidden), \
                patch.object(subprocess, 'Popen', side_effect=forbidden), \
                patch.object(subprocess, 'run', side_effect=forbidden), \
                patch.object(subprocess, 'check_output', side_effect=forbidden):
            # Actual public target, including its fresh-process owner-class binding.
            study._child_entry(arm, plan, None, control_root)
    except SystemExit as error:
        if error.code != 0:
            raise
    except BaseException:
        queue.put(dict(error=traceback.format_exc()))
        raise


@unittest.skipUnless(os.environ.get('OPENWEBRL_API_DEPENDENCIES'), 'Set OPENWEBRL_API_DEPENDENCIES for the actual frozen stealth worker check')
class ActualFreshStealthWorker(unittest.TestCase):
    def test_both_actual_arms_spawn_import_without_main_reentry_or_provider_clients(self):
        from openwebrl.api_experiments.reproduce import cloud_browsercode, prepare_browsercode, stealth_study as study, tasks
        dependencies = json.loads(Path(os.environ['OPENWEBRL_API_DEPENDENCIES']).read_text())
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='stealth-fresh-spawn-') as temporary:
            root = Path(temporary)
            task_file = root / 'tasks.json'
            task_file.write_text(json.dumps(tasks.load_tasks('online-mind2web')))
            runtime = root / 'runtime'
            prepare_browsercode.build('browsercode_luna_single_high', runtime,
                **{name: dependencies[name] for name in ('cuawright_source', 'browsercode_source', 'bun', 'chromium', 'node_modules')}, tasks_json=task_file)
            cloud_browsercode.configure(runtime)
            plan = study.prepare(runtime, root / 'study', 0)
            shard = Path(plan['output'])
            study.exclusive(shard / 'study-control-config.json', dict(global_control=str(root / 'global'), shard=0))
            global_control = study.GlobalControl(root / 'global')
            global_control.initialize(time.time(), plan['global_limits'])
            control = study.Control(shard / 'shared-control')
            control.initialize(time.time() + 57600)
            context = multiprocessing.get_context('spawn')
            queue = context.Queue()
            workers = [context.Process(target=guarded_worker, args=(arm, plan, str(control.root), queue)) for arm in study.ARMS]
            from unittest.mock import patch
            with patch.dict(os.environ, OPENAI_API_KEY='offline-unused', BROWSER_USE_API_KEY='offline-unused', PYTHONDONTWRITEBYTECODE='1'):
                try:
                    for worker in workers:
                        worker.start()
                    for worker in workers:
                        worker.join(90)
                        self.assertEqual(worker.exitcode, 0)
                    rows = [queue.get(timeout=5) for _ in workers]
                    self.assertEqual({row['arm'] for row in rows}, set(study.ARMS))
                    self.assertTrue(all(row['actual_frozen_worker_imported'] and row['provider_calls'] == 0 for row in rows))
                finally:
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                        worker.join()
                    queue.close()
                    queue.join_thread()


if __name__ == '__main__':
    unittest.main()
