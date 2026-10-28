"""CPU regression for the 395121/395122 startup failure: run the patched launcher's real CampaignRunner.switch_critic (port
mapping from a real lease base, server env construction, health polling, /server_info, relays, preflight) against fake
servers that, like sglang, warm up through the inherited proxy environment. The cluster proxy is simulated by a dead
http_proxy and the cluster's no_proxy (no 127.0.0.1). Negative control: with that env unchanged, /server_info never answers."""
import asyncio, json, os, shlex, socket, subprocess, sys, tempfile, time, unittest
from pathlib import Path
PKG = Path(os.environ['V12_PACKAGE']); sys.path.insert(0, str(PKG / 'source'))
from scripts import launch_webprm_table5_campaign as L  # noqa: E402
T = Path(__file__).resolve().parent
FAKE = T / 'fake_sglang_v12.py'
CLUSTER_PROXY = dict(http_proxy='http://127.0.0.1:9', HTTP_PROXY='http://127.0.0.1:9', https_proxy='http://127.0.0.1:9',
                     no_proxy='localhost,hyak.local,hyakm.washington.edu', NO_PROXY='localhost,hyak.local,hyakm.washington.edu')

def free_base():
    for base in range(31000, 39000, 8):
        try:
            socks = []
            for p in range(base, base + 4):
                s = socket.socket(); s.bind(('127.0.0.1', p)); socks.append(s)
            for s in socks: s.close()
            return base
        except OSError:
            for s in socks: s.close()
    raise RuntimeError('no free ports')

def fake_command(argv, log, env):
    cmd = shlex.split(argv[-1].split('exec ', 1)[1]) if argv[0] == 'bash' else argv
    if any(str(x).endswith('owrl_uds_relay.py') for x in cmd):
        return subprocess.Popen(cmd, env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT)
    opt = lambda k: cmd[cmd.index(k) + 1] if k in cmd else None
    if 'sglang.launch_server' in cmd:
        args = ['--kind', 'sglang', '--port', opt('--port'), '--model', opt('--model-path'), '--seed', opt('--random-seed') or '0']
    else:
        args = ['--kind', 'select', '--port', opt('--port'), '--model', opt('--model')]
    return subprocess.Popen([sys.executable, '-B', str(FAKE), *args], env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT)

class StartupPath(unittest.TestCase):
    def runner(self, tmp):
        r = L.CampaignRunner.__new__(L.CampaignRunner)
        base = free_base()
        r.ports = {name: base + i for i, name in enumerate(L.CRITICS)}
        r.owrl_ports = dict(actor=base + 1, select=base + 2, selector_sglang=base + 3)  # as patched start()
        r.halt_reason = None; r.servers = []; r.children = []; r.broker = None; r.proposal = dict(domains={'reddit': {}})
        r.last_storage = time.monotonic(); r.context = dict(deadline=time.time() + 900)
        r.output = tmp / 'out'; r.output.mkdir(); r.scratch = tmp / 'scr'; r.scratch.mkdir()
        r.env = dict(os.environ, **CLUSTER_PROXY)
        r.command = lambda argv, log, env=None: (lambda p: (r.children.append(p), p)[1])(fake_command(argv, log, r.env if env is None else env))
        return r

    def setUp(self):
        self.saved = (L.put, L.read)
        L.put = lambda path, value: Path(path).write_text(json.dumps(value))
        L.read = lambda path: json.loads(Path(path).read_text())

    def tearDown(self):
        L.put, L.read = self.saved

    def test_patched_switch_critic_with_cluster_proxy(self):
        tmp = Path(tempfile.mkdtemp(prefix='v12s-', dir='/tmp'))
        r = self.runner(tmp)
        try:
            started = time.time()
            asyncio.run(r.switch_critic('piotr_selectionarm'))
            self.assertTrue(r.owrl_started)
            info = json.loads((r.output / 'owrl-actor-server-info.json').read_text())
            self.assertEqual(info['random_seed'], 4200)
            env = json.loads((r.output / 'owrl-server-environment.json').read_text())
            self.assertIn('127.0.0.1', env['no_proxy'].split(','))
            pre = json.loads((r.output / 'owrl-local-preflight.json').read_text())
            self.assertEqual(pre['select']['raw'], '{"selection": 1}')
            cmd = json.loads((r.output / 'owrl-actor-command.json').read_text())['argv'][-1]
            self.assertIn(f"--port {r.owrl_ports['actor']} ", cmd)
            print(f'patched switch_critic passed under the cluster proxy env in {time.time()-started:.1f}s; actor port {r.owrl_ports["actor"]}, no_proxy {env["no_proxy"]}')
        finally:
            for p in r.children: p.terminate()

    def test_negative_control_old_env_hangs(self):
        port = free_base()
        p = subprocess.Popen([sys.executable, '-B', str(FAKE), '--kind', 'sglang', '--port', str(port), '--model', 'm', '--seed', '4200'],
                             env=dict(os.environ, **CLUSTER_PROXY), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(50):
                try:
                    L.loopback_urlopen(f'http://127.0.0.1:{port}/get_model_info', timeout=2).read(); break
                except OSError:
                    time.sleep(.2)
            with self.assertRaises(OSError):
                L.loopback_urlopen(f'http://127.0.0.1:{port}/server_info', timeout=10)   # the r0 call
            print('negative control: with the inherited proxy env the r0 /server_info call times out (reproduces 395121/395122)')
        finally:
            p.terminate()

if __name__ == '__main__':
    unittest.main(verbosity=2)
