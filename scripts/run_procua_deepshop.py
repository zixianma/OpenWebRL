#!/usr/bin/env python3
"""PRO-CUA (arXiv 2605.29119) on our DeepShop-150: their released runner, our tasks and our native judge.

  prepare   freeze plan.json + their-format task file (from the pinned crossbench DeepShop copy)
  agent     run PRO-CUA's own WebVoyager runner (run_qwen3_og_parallel_playwright_data_collection_train.py,
            imported unmodified from the pinned clone) against a local sglang OpenAI endpoint; resumable
  judge     score every saved trajectory with openwebrl.eval.reward_deepshop (gpt-4o, 30 images, cost-capped)
  summarize overall (unanswered/failed = 0) and valid-only success rates
  execute   inside the job: sglang up -> agent -> sglang down -> judge -> summarize

Only three runner settings are patched, all outside their agent logic: the OpenAI base_url (their main() hardcodes
localhost:8000), the thread-pool width (hardcoded 32; set to our 8 CPUs) and top_p (our T1.0/p0.9 cohort protocol).
Their prompt, 1000x1000 window, one-screenshot context, action parser/executors and step loop are untouched.
"""
import argparse
import asyncio
import base64
import concurrent.futures
import functools
import hashlib
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
PYTHON = RUNTIME / 'venv/bin/python'
PROCUA = Path('/gpfs/scrubbed/zixianma/external/PRO-CUA')
PROCUA_COMMIT = '83f60cffb7d7924fb464e9ed28a5eddf8b60e772'
RUNNER = 'run_qwen3_og_parallel_playwright_data_collection_train'
CKPTS = Path('/gpfs/scrubbed/zixianma/checkpoints/web')
MODELS = {
    '4b': dict(path=CKPTS / 'PRO-CUA-4B-3b028de74a50', repo='PRO-CUA/PRO-CUA-4B',
               revision='3b028de74a50781f117b98475d109356a3a7a4b7'),
    '8b': dict(path=CKPTS / 'PRO-CUA-8B-72eeab6212c0', repo='PRO-CUA/PRO-CUA-8B',
               revision='72eeab6212c017b24a2a1546f9d0c8753bee0eaa'),
}
TASKS = RUNTIME / 'evaluations/sol61-inference-scaling-20261009/preparation/tasks/deepshop.jsonl'
TASKS_SHA256 = '4694e61ec2b8c826334a2d0c4e6e82f0452a00d29b10a42ab95da5f75e2193b2'
PROTOCOL = dict(
    runner=f'PRO-CUA@{PROCUA_COMMIT}:WebVoyager/{RUNNER}.py', max_iter=30, max_attached_imgs=1, window=[1000, 1000],
    temperature=1.0, top_p=0.9, workers=8, context_length=12800,   # their launch_vllm_servers.sh VLLM_MAX_MODEL_LEN
    judge=dict(module='openwebrl.eval.reward_deepshop', model='gpt-4o', images=30, max_calls=600, max_usd=18.0),
)


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(p)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(a):
    root = Path(a.root)
    if (root / 'plan.json').exists():
        raise SystemExit(f'{root}/plan.json exists; plans are frozen')
    model = MODELS[a.model]
    assert sha256(TASKS) == TASKS_SHA256, 'DeepShop task source changed'
    commit = subprocess.check_output(['git', '-C', str(PROCUA), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == PROCUA_COMMIT, f'PRO-CUA clone at {commit}'
    assert read_json(model['path'] / 'PROVENANCE.json')['revision'] == model['revision']
    tasks = [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]
    assert len(tasks) == 150 and len({t['task_id'] for t in tasks}) == 150
    # their loader wants id/web/ques; web_name is used only by their exclude filter (allrecipe/github: none here)
    theirs = [dict(id=t['task_id'], web=t['website'], ques=t['task_name'], web_name='deepshop') for t in tasks]
    root.mkdir(parents=True, exist_ok=True)
    (root / 'tasks-procua.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in theirs))
    write_json(root / 'plan.json', dict(
        created=time.strftime('%Y-%m-%dT%H:%M:%S'), model=a.model, actor=str(model['path']), hf_repo=model['repo'],
        hf_revision=model['revision'], served_name=f"PRO-CUA-{a.model.upper()}", port=a.port,
        task_source=str(TASKS), task_source_sha256=TASKS_SHA256, n_tasks=150, protocol=PROTOCOL))
    print(root / 'plan.json')


# ---------------------------------------------------------------- agent (their runner)

def agent(a):
    root = Path(a.root)
    plan = read_json(root / 'plan.json')
    proto = plan['protocol']
    webvoyager = PROCUA / 'WebVoyager'
    sys.path.insert(0, str(webvoyager))
    os.chdir(webvoyager)              # their relative imports/fonts; outputs go to absolute paths below
    runner = importlib.import_module(RUNNER)

    base_url = f"http://127.0.0.1:{plan['port']}/v1"
    real_openai = runner.OpenAI

    def client_factory(**unused):
        import httpx    # the cluster http_proxy would otherwise intercept 127.0.0.1
        client = real_openai(base_url=base_url, api_key='not-needed', timeout=300,
                             http_client=httpx.Client(trust_env=False, timeout=300))
        create = client.chat.completions.create
        client.chat.completions.create = functools.partial(create, top_p=proto['top_p'])
        return client
    runner.OpenAI = client_factory
    runner.concurrent = SimpleNamespace(futures=SimpleNamespace(
        ThreadPoolExecutor=functools.partial(concurrent.futures.ThreadPoolExecutor, max_workers=proto['workers']),
        as_completed=concurrent.futures.as_completed))

    # resume: their main() rmtree's output_dir/round_<k>, so each attempt writes a fresh round with only unfinished tasks
    done = {t for t in finished_tasks(root)}
    remaining = [json.loads(line) for line in (root / 'tasks-procua.jsonl').read_text().splitlines()
                 if json.loads(line)['id'] not in done]
    if not remaining:
        print('agent: all tasks have trajectories')
        return
    attempt = 1 + max([int(p.name.split('_')[1]) for p in root.glob('round_*')] or [0])
    task_file = root / f'tasks-attempt{attempt}.jsonl'
    task_file.write_text(''.join(json.dumps(t) + '\n' for t in remaining))
    downloads = root / 'downloads'
    downloads.mkdir(exist_ok=True)
    print(f'agent: attempt {attempt}, {len(remaining)} tasks ({len(done)} already finished)', flush=True)
    sys.argv = [RUNNER, '--test_file', str(task_file), '--output_dir', str(root), '--round', str(attempt),
                '--max_iter', str(proto['max_iter']), '--max_attached_imgs', str(proto['max_attached_imgs']),
                '--temperature', str(proto['temperature']), '--window_width', str(proto['window'][0]),
                '--window_height', str(proto['window'][1]), '--fix_box_color', '--seed', '42', '--headless',
                '--model', 'qwen3', '--model_name', plan['served_name'], '--num_trials', '1',
                '--download_dir', str(downloads)]
    runner.main()


def finished_tasks(root):
    """task id -> trajectory dir (latest attempt wins); a task is finished once their runner wrote interact_messages."""
    out = {}
    for d in sorted(Path(root).glob('round_*/task*-1'), key=lambda p: int(p.parent.name.split('_')[1])):
        if (d / 'interact_messages.json').exists():
            out[d.name[len('task'):-len('-1')]] = d
    return out


# ---------------------------------------------------------------- judge (our native DeepShop judge)

def extract_answer(messages):
    """Final answer from their trajectory: the `answer` tool_call text; for `terminate`, the whole last message (as their
    auto_eval_gpt.py does). No answer/terminate (step limit, crash) -> '' -> our judge scores 0."""
    last = next((m for m in reversed(messages) if m['role'] == 'assistant'), None)
    if last is None:
        return '', 'no_assistant_turn'
    text = last['content'] if isinstance(last['content'], str) else ''
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    try:
        call = json.loads(m.group(1).strip()) if m else None
    except json.JSONDecodeError:
        call = None
    action = (call or {}).get('arguments', {}).get('action')
    if action == 'answer':
        return str(call['arguments'].get('text') or '').strip(), 'answer'
    if action == 'terminate':
        return text.strip(), 'terminate'
    return '', f'last_action={action}'


async def judge_all(a):
    sys.path.insert(0, str(REPO))
    from openwebrl import crossbench_selector_eval as xb
    from openwebrl.eval import reward_deepshop as rd
    from slime.utils.types import Sample
    root = Path(a.root)
    plan = read_json(root / 'plan.json')
    cfg = plan['protocol']['judge']
    if not os.getenv('OPENAI_API_KEY'):          # repo .env; never printed
        for line in (REPO / '.env').read_text().splitlines():
            if line.startswith('OPENAI_API_KEY='):
                os.environ['OPENAI_API_KEY'] = line.split('=', 1)[1].strip().strip('"\'')
    judge = xb.NativeCappedJudge(root / 'judge_budget', 'deepshop', max_calls=cfg['max_calls'], max_usd=cfg['max_usd'],
                                 attempt_context=xb.ATTEMPT, client=xb.LiveRuntime().judge_client())
    rd._get_openai_client = lambda **unused: judge
    rd._extract_final_answer = lambda parser, response: response      # answer already extracted from their format
    args = SimpleNamespace(judge_max_attached_imgs=cfg['images'], judge_api_model=cfg['model'], judge_timeout_secs=180.0,
                           judge_api_mode='token', log_judge_output=False)
    tasks = {json.loads(l)['id']: json.loads(l) for l in (root / 'tasks-procua.jsonl').read_text().splitlines()}
    traj = finished_tasks(root)
    out_dir = root / 'verdicts'
    sem = asyncio.Semaphore(8)

    async def one(task_id):
        path = out_dir / f'{task_id}.json'
        if path.exists() and read_json(path).get('score') is not None:
            return
        d = traj.get(task_id)
        rec = dict(task_id=task_id, trajectory=str(d) if d else None, judge_model=cfg['model'], judge_protocol='deepshop')
        if d is None:
            rec.update(score=0.0, answer='', answer_kind='no_trajectory', judge_text='runner produced no trajectory',
                       n_steps=0, n_screenshots=0)
        else:
            messages = read_json(d / 'interact_messages.json')
            answer, kind = extract_answer(messages)
            shots = sorted(d.glob('screenshot*.png'), key=lambda p: int(re.search(r'(\d+)', p.name).group(1)))
            sample = Sample(response=answer, status=Sample.Status.COMPLETED,
                            metadata=dict(intent=tasks[task_id]['ques'], task_id=task_id,
                                          full_image_list=[base64.b64encode(p.read_bytes()).decode() for p in shots]))
            async with sem:
                score, text, timeout = await rd._judge(args, sample, None)
            rec.update(score=score, answer=answer, answer_kind=kind, judge_text=text, judge_timeout=timeout,
                       n_steps=sum(m['role'] == 'assistant' for m in messages), n_screenshots=len(shots),
                       judged_screenshots=min(len(shots), cfg['images']))
        write_json(path, rec)

    await asyncio.gather(*(one(t) for t in tasks))
    print(f'judge: ledger {json.dumps(judge.budget.state().get("charged_or_reserved_usd"))}')


def summarize(a):
    root = Path(a.root)
    plan = read_json(root / 'plan.json')
    recs = [read_json(p) for p in sorted((root / 'verdicts').glob('*.json'))]
    n = plan['n_tasks']
    succ = sum(r['score'] == 1.0 for r in recs)
    valid = [r for r in recs if r['score'] is not None]
    kinds = {}
    for r in recs:
        kinds[r['answer_kind']] = kinds.get(r['answer_kind'], 0) + 1
    by_site = {}
    for r in valid:
        s = by_site.setdefault(r['task_id'].split('--')[0], [0, 0])
        s[0] += r['score'] == 1.0
        s[1] += 1
    summary = dict(model=plan['hf_repo'], revision=plan['hf_revision'], n_tasks=n, verdicts=len(recs),
                   successes=succ, overall_rate=succ / n, valid=len(valid),
                   valid_only_rate=(succ / len(valid)) if valid else None, judge_invalid=len(recs) - len(valid),
                   answer_kinds=kinds, by_site={k: dict(successes=v[0], n=v[1]) for k, v in sorted(by_site.items())},
                   mean_steps=sum(r['n_steps'] for r in recs) / max(1, len(recs)))
    write_json(root / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- execute (inside the job)

def execute(a):
    root = Path(a.root)
    plan = read_json(root / 'plan.json')
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    port = plan['port']
    argv = [str(PYTHON), '-m', 'sglang.launch_server', '--model-path', plan['actor'], '--served-model-name',
            plan['served_name'], '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16', '--tp', '1',
            '--mem-fraction-static', '.80', '--context-length', str(plan['protocol']['context_length']),
            '--random-seed', '4200',
            # sglang otherwise infers a template from the path (picked points-v15-chat for PRO-CUA-*): use the HF one
            '--chat-template', str(Path(plan['actor']) / 'chat_template.jinja')]
    env = dict(os.environ, NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    server = subprocess.Popen(argv, stdout=open(logs / 'sglang.log', 'a'), stderr=subprocess.STDOUT,
                              start_new_session=True, env=env)
    try:
        import urllib.request
        deadline = time.time() + 1200
        while True:
            if server.poll() is not None:
                raise SystemExit(f'sglang exited {server.returncode}; see {logs}/sglang.log')
            try:
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                    f'http://127.0.0.1:{port}/get_model_info', timeout=5)
                break
            except Exception:
                if time.time() > deadline:
                    raise SystemExit('sglang did not become healthy in 20 min')
                time.sleep(10)
        print('sglang healthy', flush=True)
        rc = subprocess.call([str(PYTHON), __file__, 'agent', '--root', str(root)],
                             stdout=open(logs / 'agent.log', 'a'), stderr=subprocess.STDOUT)
        print(f'agent exit {rc}', flush=True)
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
    asyncio.run(judge_all(a))
    summarize(a)


SBATCH = """#!/usr/bin/env bash
#SBATCH --job-name=procua-{model}-ds
#SBATCH --account=vlm
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output={root}/slurm-%j.log
set -euo pipefail
source {repo}/scripts/h200_env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec {python} {script} execute --root {root}
"""


def launch_script(a):
    root = Path(a.root)
    plan = read_json(root / 'plan.json')
    path = root / 'launch.sbatch'
    path.write_text(SBATCH.format(model=plan['model'], root=root, repo=REPO, python=PYTHON, script=Path(__file__).resolve()))
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('prepare', 'agent', 'judge', 'summarize', 'execute', 'launch-script'):
        s = sub.add_parser(name)
        s.add_argument('--root', required=True)
        if name == 'prepare':
            s.add_argument('--model', choices=sorted(MODELS), required=True)
            s.add_argument('--port', type=int, default=30711)
    a = p.parse_args()
    dict(prepare=prepare, agent=agent, judge=lambda a: asyncio.run(judge_all(a)), summarize=summarize,
         execute=execute, **{'launch-script': launch_script})[a.cmd](a)


if __name__ == '__main__':
    main()
