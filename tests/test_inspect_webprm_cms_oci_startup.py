import io
import tarfile
import time

import unittest
import tempfile
import os
from pathlib import Path

from scripts import inspect_webprm_cms_oci_startup as inventory


def archive(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as tar:
        for name, raw in entries:
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            tar.addfile(item, io.BytesIO(raw))
    return output.getvalue()


def scan(raw, state=None, **caps):
    state = state or inventory.new_state()
    inventory.scan_layer(io.BytesIO(raw), 0, len(raw), state, time.time() + 10,
                         dict(inventory.CAPS, **caps))
    return state


class StartupInventoryTest(unittest.TestCase):
    def test_selected_only_without_extraction(self):
        previous = os.getcwd()
        temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, previous)
        tmp_path = Path(temporary.name)
        os.chdir(tmp_path)
        state = scan(archive([('docker-entrypoint.sh', b'exec supervisord\n'), ('var/lib/mysql/data', b'not selected')]))
        assert list(state['files']) == ['docker-entrypoint.sh']
        assert state['mutable_roots']['var/lib/mysql']['cumulative_layer_bytes'] == 12
        assert not list(tmp_path.iterdir())


    def test_whiteout_applies_lower_layer_not_same_layer_new_file(self):
        state = scan(archive([('etc/supervisor.d/mysql.ini', b'old')]))
        scan(archive([('etc/supervisor.d/mysql.ini', b'new'), ('etc/supervisor.d/.wh..wh..opq', b'')]), state)
        assert state['files']['etc/supervisor.d/mysql.ini']['raw'] == b'new'
        scan(archive([('etc/supervisor.d/.wh.mysql.ini', b'')]), state)
        assert not state['files']


    def test_traversal_fails_without_writes(self):
        with self.assertRaisesRegex(inventory.GateClosed, 'traversal'):
            scan(archive([('../docker-entrypoint.sh', b'bad')]))


    def test_caps_and_selected_binary_fail_closed(self):
        raw = archive([('docker-entrypoint.sh', b'hello')])
        with self.assertRaisesRegex(inventory.GateClosed, 'Selected text cap'):
            scan(raw, selected_file_bytes=4)
        with self.assertRaisesRegex(inventory.GateClosed, 'Decompression byte cap'):
            scan(raw, raw_bytes=1)
        with self.assertRaisesRegex(inventory.GateClosed, 'ordinary text'):
            scan(archive([('docker-entrypoint.sh', b'\0')]))


    def test_symlink_remains_unresolved_not_read(self):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w') as tar:
            item = tarfile.TarInfo('docker-entrypoint.sh')
            item.type = tarfile.SYMTYPE
            item.linkname = '/gpfs/home/zixianma/private'
            tar.addfile(item)
        state = scan(output.getvalue())
        assert not state['files']
        assert state['links']['docker-entrypoint.sh']['target'] == '/gpfs/home/zixianma/private'
