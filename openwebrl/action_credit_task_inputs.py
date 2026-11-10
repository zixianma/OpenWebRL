"""Exact dataset and native resource inputs for fixed action-credit tasks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from .action_credit import digest
from .action_credit_collect import generator_protocol


def _verified_read_path(reference):
    path = Path(reference['path'])
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != reference['sha256']:
        raise ValueError('Pinned input file changed: '+str(path))
    return path


def direct_engine_args(native_args, actor, generate_url):
    """Use the assigned engine; no routing or fallback endpoint is permitted."""
    parsed = urlsplit(generate_url)
    if (parsed.scheme != 'http' or not parsed.hostname or parsed.port is None
            or parsed.path != '/generate' or parsed.query or parsed.fragment
            or parsed.username or parsed.password or digest(generate_url) != actor.endpoint_sha256):
        raise ValueError('Direct engine URL must match the frozen actor')
    return dict(native_args, sglang_router_ip=parsed.hostname, sglang_router_port=parsed.port)


class NativeTaskInputs:
    """Resolve Sample metadata and the actual native loader's six input pins.

    Task identifiers are checked in original parquet order, then against every
    fixed update group. Resources are loaded with the same native function used
    by the child browser worker. This does not open a browser or call an API.
    """
    def __init__(self, *, dataset_reference, task_order_reference, browser_config,
                 browser_manifest, native_args):
        import pyarrow.parquet as pq
        dataset = _verified_read_path(dataset_reference)
        order = json.loads(_verified_read_path(task_order_reference).read_text())
        if order['source_sha256'] != dataset_reference['sha256']:
            raise ValueError('Task order belongs to a different dataset')
        rows = pq.read_table(dataset).to_pylist()
        ids = [row['metadata']['task_id'] for row in rows]
        if len(set(ids)) != len(ids) or ids != order['original_manifest_task_order']:
            raise ValueError('Dataset membership/order differs from pinned training manifest')
        for update in order['iterations']:
            for group in update['groups']:
                if ids[group['source_index']] != group['task_id']:
                    raise ValueError('Fixed group does not match original source index')
        self.rows = {task: row for task, row in zip(ids, rows)}
        self.browser_config = deepcopy(browser_config)
        self.browser_manifest = deepcopy(browser_manifest)
        self.native_args = deepcopy(native_args)
        self.generator_sha = digest(generator_protocol(SimpleNamespace(**native_args)))
        self.cache = {}

    def sample(self, request):
        row = self.rows[request.task_id]
        # A unique native trajectory id is independent of completion ordering.
        return dict(prompt=deepcopy(row['prompt']), metadata=deepcopy(row['metadata']),
                    index=request.request_id)

    def request_pins(self, task_id):
        if task_id not in self.cache:
            from .generate_browser import _load_local_resources
            task, tools, policy = _load_local_resources(task_id, deepcopy(self.browser_config),
                deepcopy(self.rows[task_id]['metadata']),
                response_format_mode_name=self.native_args['browser_response_format_mode'])
            self.cache[task_id] = dict(task_data_sha256=digest(task), tools_sha256=digest(tools),
                environment_policy_sha256=digest(policy),
                browser_config_sha256=digest(self.browser_config),
                browser_manifest_sha256=digest(self.browser_manifest),
                generator_protocol_sha256=self.generator_sha)
        return dict(self.cache[task_id])
