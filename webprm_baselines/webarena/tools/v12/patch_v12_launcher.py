# v12 launcher patch (exec'd by build_v12.py after the unchanged v9/v11/v11b patches; free name: t = launcher text).
# OpenWebRL-4B-SFT actor: start the crossbench actor + Piotr SelectionARM servers once (instead of the v11 critic
# switch), relay them to the cells over UNIX sockets, run the v12 worker with the OpenWebRL venv, audit with
# native-audit-v12, and bind the v12 cohort/actor identity. Science, caps and every other control stay as v11b.

def _one(old, new, count=1):
    global t
    assert t.count(old) == count, (old[:120], t.count(old))
    t = t.replace(old, new)

# Cohort / actor / fresh-experiment identity.
_one("            or cohort['actors']!=['gpt-4o-2024-08-06']", "            or cohort['actors']!=[V12_ACTOR]")
_one("c['actor']!='gpt-4o-2024-08-06' or raw[c['task_id']]", "c['actor']!=V12_ACTOR or raw[c['task_id']]")
_one("if proposal['cell_routes']['gpt-4o-2024-08-06'][selector]!=", "if proposal['cell_routes'][V12_ACTOR][selector]!=")
_one("fresh.get('identity') != 'native_table5_v11'", "fresh.get('identity') != 'native_table5_v12_owrl'")
_one("or fresh.get('actor') != 'gpt-4o-2024-08-06' or", "or fresh.get('actor') != V12_ACTOR or")
_one("reused.get('path')!=str(REPO/'scripts/run_native_webarena_cell.py')", "reused.get('path')!=str(REPO/'scripts/run_owrl_webarena_cell.py')")
_one("              'scripts/run_native_webarena_cell.py','openwebrl/webprm_native_release.py',",
     "              'scripts/run_native_webarena_cell.py','scripts/run_owrl_webarena_cell.py','openwebrl/webprm_owrl_v12.py','openwebrl/webprm_native_release.py',")
_one("    if (digest(cohort)!=proposal['cohort_sha256'] or cohort.get('protocol')!=V9_COHORT_PROTOCOL",
     "    if (digest(cohort)!=proposal['cohort_sha256'] or cohort.get('protocol')!=V12_PROTOCOL")

# Ports: one more block member per server (actor, /select, selector sglang).
_one("self.port_lease,base=lease_ports(self.context['job_id'],size=2,directory=ports_dir)",
     "self.port_lease,base=lease_ports(self.context['job_id'],size=4,directory=ports_dir)")
_one("        self.ports={name:base+i for i,name in enumerate(CRITICS)}\n",
     "        self.ports={name:base+i for i,name in enumerate(CRITICS)}\n"
     "        self.owrl_ports=dict(actor=base+1,select=base+2,selector_sglang=base+3)  # v12\n")

# Per-cell plan: the job-scratch UNIX sockets.
_one("    plan.update(authorization_status='approved', job_id=context['job_id'], deadline=context['deadline'],",
     "    plan['owrl']=dict(template['owrl'],actor_socket=str(Path(context['scratch'])/'owrl'/'actor.sock'),  # v12\n"
     "        select_socket=str(Path(context['scratch'])/'owrl'/'select.sock'))\n"
     "    plan.update(authorization_status='approved', job_id=context['job_id'], deadline=context['deadline'],")

# Worker: the v12 cell with the OpenWebRL venv.
_one("argv=prefix+[BROWSER,'-B',str(REPO/'scripts/run_native_webarena_cell.py'),",
     "argv=prefix+[V12_PYTHON,'-B',str(REPO/'scripts/run_owrl_webarena_cell.py'),")

# Servers: replace the v11 critic switch.
_start = t.index('    async def switch_critic(self,name):')
_end = t.index('    async def start(self, cells):')
t = t[:_start] + '''    async def switch_critic(self,name):
        """v12: crossbench serving on GPU 0 (selector sglang .30, actor sglang .45, serve_arm_sglang /select), then the
        UNIX-socket relays and two zero-paid local preflights. Started once; the v11 critic switch is not used."""
        if getattr(self,'owrl_started',False):return
        import shlex
        o=self.owrl_ports
        env={k:v for k,v in self.env.items() if k not in ('PYTHONPATH','NLTK_DATA')}
        env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH='')
        def sourced(argv):
            return ['bash','-c','set -euo pipefail\\nsource '+shlex.quote(str(V12_SNAPSHOT/'scripts/h200_env.sh'))+'\\nexec '+shlex.join(argv)]
        stages=[('selector-sglang',sourced([V12_PYTHON,'-m','sglang.launch_server','--model-path',V12_PIOTR_SGLANG,'--host','127.0.0.1',
                    '--port',str(o['selector_sglang']),'--dtype','bfloat16','--mem-fraction-static','.30','--context-length','32768',
                    '--grammar-backend','xgrammar','--cuda-graph-max-bs','48','--random-seed','0']),
                 f"http://127.0.0.1:{o['selector_sglang']}/get_model_info",('model_path',V12_PIOTR_SGLANG)),
                ('actor',sourced([V12_PYTHON,'-m','sglang.launch_server','--model-path',V12_ACTOR_PATH,'--host','127.0.0.1',
                    '--port',str(o['actor']),'--dtype','bfloat16','--tp','1','--mem-fraction-static','.45','--context-length','32768',
                    '--max-running-requests','40','--chunked-prefill-size','4096','--cuda-graph-max-bs','48','--random-seed','4200']),
                 f"http://127.0.0.1:{o['actor']}/get_model_info",('model_path',V12_ACTOR_PATH)),
                ('selector',sourced([V12_PYTHON,str(V12_SNAPSHOT/'scripts/serve_arm_sglang.py'),'serve','--sglang',
                    f"http://127.0.0.1:{o['selector_sglang']}",'--port',str(o['select']),'--model',V12_PIOTR_SELECTION]),
                 f"http://127.0.0.1:{o['select']}/health",('mode','selection'))]
        for label,argv,health,(key,expected) in stages:
            put(self.output/f'owrl-{label}-command.json',dict(argv=argv,gpu_index=0,scientific_identity_unchanged=True))
            self.servers.append(self.command(argv,self.output/f'owrl-{label}.log',env))
            until=min(self.context['deadline'],time.time()+1200)
            while True:
                self.check()
                if time.time()>=until:raise TimeoutError('v12 '+label+' startup deadline reached')
                try:
                    with loopback_urlopen(health,timeout=3) as response:card=json.load(response)
                except OSError:
                    await asyncio.sleep(3);continue
                if str(card.get(key))!=expected:raise ValueError(f'v12 {label} identity differs: {key}={card.get(key)!r}')
                put(self.output/f'owrl-{label}-identity.json',card);break
        with loopback_urlopen(f"http://127.0.0.1:{o['actor']}/server_info",timeout=10) as response:info=json.load(response)
        if info.get('random_seed')!=4200:raise ValueError('v12 actor server seed differs from the crossbench protocol')
        put(self.output/'owrl-actor-server-info.json',{k:info.get(k) for k in ('random_seed','sampling_backend','mem_fraction_static','cuda_graph_max_bs','context_length')})
        card=read(self.output/'owrl-selector-identity.json')
        if card.get('model')!=V12_PIOTR_SELECTION or card.get('adapter') is not None:raise ValueError('v12 selector service identity differs')
        sockets=self.scratch/'owrl';sockets.mkdir(mode=0o700)
        for label,port in (('actor',o['actor']),('select',o['select'])):
            ready=self.output/f'owrl-relay-{label}.json'
            self.servers.append(self.command([V12_PYTHON,'-B',str(V12_RELAY),'--listen',str(sockets/(label+'.sock')),'--port',str(port),
                '--ready',str(ready)],self.output/f'owrl-relay-{label}.log',env))
            until=time.time()+60
            while not ready.exists():
                self.check()
                if time.time()>=until:raise TimeoutError('v12 relay '+label+' did not start')
                await asyncio.sleep(.5)
        preflight=await asyncio.to_thread(v12_preflight,sockets)
        put(self.output/'owrl-local-preflight.json',dict(preflight,primary_task=False,paid_calls=0))
        self.owrl_started=True
        self.current_critic=name

''' + t[_end:]

# Audit / constants.
t = t.replace("\nif __name__=='__main__':main()", '''

# ---- v12 (2026-10-28, user: "ah yes try C"): OpenWebRL-4B-SFT actor in its own action space; see webprm_owrl_v12. ----
V12_PROTOCOL = 'native_table5_v12_owrl'
V12_ACTOR = 'openwebrl-sft-4b'
V12_PYTHON = '/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python'
V12_TOOLS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/v12-tools')
V12_SNAPSHOT = V12_TOOLS/'owrl-snapshot'
V12_RELAY = V12_TOOLS/'owrl_uds_relay.py'
V12_ACTOR_PATH = '/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT'
V12_PIOTR_SELECTION = '/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89'
V12_PIOTR_SGLANG = '/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection-sglang-81b452d'
V12_PREFLIGHT = REPO/'openwebrl/webprm_owrl_v12_preflight.json'


def v12_uds_json(socket_path, path, payload, timeout=180):
    """POST JSON over a relay UNIX socket (host side; stdlib)."""
    import http.client, socket as _socket
    class Connection(http.client.HTTPConnection):
        def connect(self):
            self.sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect(str(socket_path))
    connection = Connection('localhost', timeout=timeout)
    try:
        body = json.dumps(payload).encode()
        connection.request('POST', path, body=body, headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        data = response.read()
        if response.status != 200:
            raise ValueError(f'v12 preflight {path} HTTP {response.status}')
        return json.loads(data)
    finally:
        connection.close()


def v12_preflight(sockets):
    """Zero-paid local checks through the relays: one actor /generate and one selector-shaped /select."""
    frozen = json.loads(V12_PREFLIGHT.read_text())
    actor = v12_uds_json(sockets/'actor.sock', '/generate', frozen['actor'])
    meta = actor.get('meta_info', {})
    if not isinstance(actor.get('text'), str) or 'output_token_logprobs' not in meta:
        raise ValueError('v12 actor preflight lacks text/logprobs')
    selection = v12_uds_json(sockets/'select.sock', '/select', frozen['select'])
    match = re.findall(r'\\{\\s*"selection"\\s*:\\s*(\\d+)\\s*\\}', selection.get('raw') or '')
    if not match or not 1 <= int(match[-1]) <= len(frozen['select']['candidates']):
        raise ValueError('v12 selector preflight did not return a valid selection')
    return dict(actor=dict(text=actor['text'], finish_reason=meta.get('finish_reason'), prompt_tokens=meta.get('prompt_tokens')),
                select=dict(raw=selection['raw'], input_tokens=selection.get('input_tokens'), seconds=selection.get('seconds')),
                frozen_payload_sha256=hashlib.sha256(V12_PREFLIGHT.read_bytes()).hexdigest())

if __name__=='__main__':main()''')
assert t.count("if __name__=='__main__':main()") == 1
