#!/usr/bin/env python3
"""Zero-paid-call end-to-end validation of the OM2W Table 5-style controller (CPU job).

A local OpenAI-compatible fixture server stands in for every provider and critic
route; all other components are the frozen package code: real controller Runner,
claim store, concurrent budget broker and journal, per-cell worker subprocesses with
external watchdogs, real Chromium/BrowserGym on the first two live OM2W start URLs,
native observation/action code, all five selector paths and the judge plumbing.
Phase 1 runs task 1 (10 cells); phase 2 is a new broker epoch that resumes the claim
store, never reruns a recorded or claimed-without-outcome cell, and runs the rest of
task 2. Responses are scripted; nothing here measures task success.
"""
from __future__ import annotations

import argparse
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRIPTED = dict(judge=0, policy=0, state_diff=0, terminal=0, pair=0, checklist=0, pointwise=0, piotr=0)


def fixture_server(arbiter_source):
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness
    from openwebrl.webprm_om2w_table5 import JUDGE
    from openwebrl.webprm_webarena_piotr import MODEL as PIOTR
    from openwebrl.webprm_webarena_worker import load_viewer_reward
    harness = load_arbiter_harness(arbiter_source)
    reward_system = load_viewer_reward(arbiter_source).build(intent='x', observation='o', current_url='u', start_url='u',
        trajectory='', state_diff='', domain_knowledge=None, candidate_a_thought='a', candidate_a_action='a',
        candidate_b_thought='b', candidate_b_action='b')[0]
    lock = threading.Lock()

    def respond(payload):
        model, messages = payload['model'], payload['messages']
        system = messages[0]['content'] if messages and isinstance(messages[0].get('content'), str) else ''
        user = json.dumps(messages[-1].get('content'))
        logprobs = None
        if model == JUDGE:
            kind = 'judge'; text = 'Thoughts: scripted fixture.\nStatus: ' + ('failure' if 'scroll [up]' in user else 'success')
        elif model == PIOTR:
            kind = 'piotr'; text = '{"selection":4}'
        elif 'Shepherd' in model:
            if payload.get('logprobs'):
                kind = 'pointwise'; text = '<answer>Yes</answer>'
                logprobs = {'content': [dict(token='<answer>', logprob=0., top_logprobs=[]), dict(token='Yes', logprob=-.1,
                    top_logprobs=[dict(token='Yes', logprob=-.1), dict(token='No', logprob=-3.), dict(token='In', logprob=-4.)])]}
            else:
                kind = 'checklist'; text = '<answer>Reach the page described in the task.</answer>'
        elif 'Arbiter' in model or system == reward_system:
            kind = 'pair'; text = '<Answer>Response 1</Answer>'
        elif payload.get('response_format'):
            kind = 'policy'
            actions = ['scroll [down]', 'scroll [up]', 'scroll [down]', 'scroll [up]', 'scroll [down]']
            count = 1 if 'Generate exactly ONE action' in system else 5
            text = json.dumps({'BLOCKS': [dict(thought=f'Fixture candidate {i}', action=actions[i], confidence=.5) for i in range(count)]})
        elif system == harness.state_diff_system_prompt:
            kind = 'state_diff'; text = 'Scripted fixture: the viewport moved.'
        else:
            kind = 'terminal'
            done = 'Action 1:' in user
            text = json.dumps(dict(thought='Scripted stop' if done else 'Continue', confidence=.9,
                                   action='send_msg_to_user [Fixture answer]' if done else None))
        with lock:
            SCRIPTED[kind] += 1
        choice = dict(index=0, message=dict(role='assistant', content=text), finish_reason='stop')
        if logprobs:
            choice['logprobs'] = logprobs
        return dict(id='fixture', object='chat.completion', model=model, choices=[choice],
                    usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps(dict(data=[])).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            body = json.dumps(respond(payload)).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def build(work):
    from scripts.prepare_webprm_om2w_table5 import build as build_package
    server_port = int(os.environ['OM2W_FIXTURE_PORT'])
    work = Path(work)
    package = work/'package'
    env_file = work/'fixture.env'  # the user's real credential file is never read by this fixture
    env_file.write_text('OPENAI_API_KEY=fixture-not-a-provider-key\n'); env_file.chmod(0o600)
    proposal, pending = build_package('pilot10', 50.0, package, campaign_root=package/'campaign',
        fixture_overrides=dict(api_base=f'http://127.0.0.1:{server_port}/v1', env_file=str(env_file),
                               local_endpoint=f'http://127.0.0.1:{server_port}/v1'))
    return package


def run(package, report):
    import launch_webprm_om2w_table5 as launch
    from openwebrl.webprm_campaign_queue import ClaimStore
    from scripts.launch_webprm_reddit_pilot import SCRATCH
    proposal = launch.read(Path(package)/'proposal.json')
    cohort, template = launch.validate_proposal(proposal)
    preview = launch.preview(proposal)
    cells = launch.stage_cells(proposal, cohort)[:20]  # first two tasks: every actor x selector twice
    job = os.environ['SLURM_JOB_ID']
    root = Path(proposal['campaign_root']); root.mkdir(mode=0o700)
    port = int(os.environ['OM2W_FIXTURE_PORT'])

    class FixtureRunner(launch.Runner):
        gate = None

        async def start_servers(self):
            return {name: port for name in launch.CRITICS}

        def can_start(self, cell, active):
            if self.gate is not None and not self.gate(cell):
                return self.decide(cell, active, False, 'fixture_phase_gate')
            return super().can_start(cell, active)

    phases = []
    for phase, gate in ((1, lambda c: c['task_index'] == 0), (2, None), (3, None)):
        output = root/f'attempt-{job}-phase{phase}'; output.mkdir(mode=0o700)
        scratch = SCRATCH/('f'+secrets.token_hex(3)); scratch.mkdir(mode=0o700, parents=True)
        # Each phase stands in for a separate allocation: a distinct financial epoch id.
        context = dict(job_id=job, epoch_id=f'job-{job}-phase{phase}', deadline=time.time()+5400)
        before = dict(SCRIPTED)
        with ClaimStore(root/'queue', cells) as store:
            if phase == 2:
                # A claim without an outcome (e.g. an interrupted primary) must never be rerun.
                store.claim(cells[10])
            runner = FixtureRunner(proposal, template, context, output, scratch)
            runner.gate = gate
            summary = asyncio.run(runner.run(cells, store))
            states = {c['cell_id']: store.state(c) for c in cells}
        phases.append(dict(phase=phase, summary=summary, states_count={s: list(states.values()).count(s) for s in set(states.values())},
                           scripted_calls={k: SCRIPTED[k]-before[k] for k in SCRIPTED},
                           broker_closure=launch.read(output/'broker-closure.json') if (output/'broker-closure.json').exists() else None))
    results = []
    for path in sorted(root.glob('attempt-*/cells/*/episodes/*/*/result.json')):
        result = json.loads(path.read_text())
        folder = path.parent
        results.append(dict(cell=result['task_id']+'/'+result['arm'], valid=result['valid'], success=result['success'],
            turns=result.get('turns'), error_type=result.get('error_type'), error=str(result.get('error'))[:300],
            judge=(folder/'om2w-judge.json').exists(), terminal_png=(folder/'native-evaluation/terminal.png').exists(),
            watchdog_closed=json.loads((folder/'watchdog.json').read_text()).get('remaining_owned_processes') == {}))
    events = [json.loads(line) for line in (root/'shared-budget.json.events.jsonl').read_text().splitlines()]
    kinds = {}
    for event in events:
        kinds[event['event']] = kinds.get(event['event'], 0)+1
    value = dict(schema_version=1, preview=preview, phases=phases, results=results,
        budget_events=kinds, charged_usd=sum(float(e['actual']) for e in events if e['event'] == 'settle'),
        provider_calls=0, paid_provider='local fixture server only (api_base override in a /tmp fixture package)')
    Path(report).write_text(json.dumps(value, indent=1, sort_keys=True, default=str)+'\n')
    print(json.dumps(dict(phases=[dict(p, broker_closure=None) for p in phases],
        results=len(results), valid=sum(r['valid'] for r in results), budget_events=kinds), indent=1, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    # Never load the user's real credential file in this fixture; the scripted server ignores the key.
    os.environ['OPENAI_API_KEY'] = 'fixture-not-a-provider-key'
    server = fixture_server('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
    os.environ['OM2W_FIXTURE_PORT'] = str(server.server_address[1])
    package = build(args.work)
    source = package/'source'
    os.chdir(source); sys.path[:0] = [str(source), str(source/'scripts')]
    for name in [m for m in sys.modules if m == 'scripts' or m.startswith(('scripts.', 'openwebrl'))]:
        del sys.modules[name]
    run(package, args.report)
    server.shutdown()


if __name__ == '__main__':
    main()
