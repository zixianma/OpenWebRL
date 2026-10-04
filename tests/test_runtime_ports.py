import ast
from pathlib import Path
import socket
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from runtime_ports import lease_ports, patch_rollout_ports, OLD_BASE


class PortIsolation(unittest.TestCase):
    def test_simultaneous_controllers_do_not_share_or_leak_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, base = lease_ports('1', directory=tmp, bases=[29100, 29200], size=8)
            try:
                second, other = lease_ports('1', directory=tmp, bases=[29100, 29200], size=8)
                try:
                    self.assertNotEqual(base, other)
                    with self.assertRaises(RuntimeError):
                        lease_ports('1', directory=tmp, bases=[29100, 29200], size=8)
                finally:
                    second.close()
            finally:
                first.close()
            again, recovered = lease_ports('1', directory=tmp, bases=[base], size=8)
            again.close()
            self.assertEqual(recovered, base)

    def test_existing_listener_disqualifies_entire_block(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as sock:
            sock.bind(('0.0.0.0', 29303)); sock.listen()
            with self.assertRaises(RuntimeError):
                lease_ports('1', directory=tmp, bases=[29300], size=8)

    def test_source_patch_uses_environment_and_preserves_cursor(self):
        code=compile(ast.parse(patch_rollout_ports(OLD_BASE)), '<ports>', 'exec')
        import types
        for cursors, environment, expected in [({}, {}, 15000), ({}, {'OPENWEBRL_ROLLOUT_PORT_BASE':'22048'}, 22048), ({0:22114}, {'OPENWEBRL_ROLLOUT_PORT_BASE':'22048'}, 22114)]:
            scope=dict(port_cursors=cursors, os=types.SimpleNamespace(environ=environment))
            exec(code,scope)
            self.assertEqual(scope['base_port'], expected)


if __name__ == '__main__':
    unittest.main()
