#!/usr/bin/env python3
"""WebPRMBench for more selectors: Recipe 1/2 heads (actor-feature scoring) and 4B LoRAs (generative).

Same protocol as the earlier Piotr and K=1 runs: released data, unchanged upstream evaluator (prompts, parser,
metrics), T 0, 1,024 tokens, stop </Answer>. Adds an order-swapped copy of the benchmark (responses 1 and 2
exchanged, labels flipped) to measure position bias of generative judges.

  swap  --data D --out S                         CPU: write the swapped benchmark and check every pair parses
  gen   --name N --model M [--tokenizer T --chat-template C --config C] --data D... --out O
                                                 GPU: one SGLang server, one evaluator pass per data file
  head  --name N --head H --model M --data D --out O
                                                 GPU: score both responses with a frozen head on actor features
Every pass writes O/<name>/<data-stem>/{pairs.jsonl,metrics.json,raw.jsonl,done.json}; a finished pass is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.webprm_static_bench import run_official  # noqa: E402

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
EVALUATOR = RUNTIME / 'webprm-baselines-20261009/sources/arbiter/eval/WebPRMBench/script/evaluate_webprm_vllm.py'
EVALUATOR_SHA = '96426a7ea96406b100869bd1a1ed138e086bb5ef81ac21f5640712c204981666'
PARAMS = dict(temperature=0., top_p=1., max_tokens=1024, stop=['</Answer>'])
B1, E1, B2, E2 = ('[The Begin of Response 1]', '[The End of Response 1]',
                  '[The Begin of Response 2]', '[The End of Response 2]')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def swap_prompt(prompt):
    if any(prompt.count(m) != 1 for m in (B1, E1, B2, E2)):
        raise ValueError('ambiguous response delimiters')
    a0, a1 = prompt.index(B1) + len(B1), prompt.index(E1)
    b0, b1 = prompt.index(B2) + len(B2), prompt.index(E2)
    if not a0 < a1 < b0 < b1:
        raise ValueError('response delimiters out of order')
    return prompt[:a0] + prompt[b0:b1] + prompt[a1:b0] + prompt[a0:a1] + prompt[b1:]


def cmd_swap(args):
    rows = [json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
    keys = {(r['source_name'], r['state_idx'], r['pair_idx']) for r in rows}
    assert len(rows) == 4600 and len(keys) == 4600, 'expected the full released benchmark'
    out = []
    for r in rows:
        s = swap_prompt(r['prompt'])
        assert swap_prompt(s) == r['prompt'] and r['expected'] in (1, 2)
        out.append({**r, 'prompt': s, 'expected': 3 - r['expected']})
    Path(args.out).write_text(''.join(json.dumps(r) + '\n' for r in out))
    print(json.dumps(dict(rows=len(out), source=args.data, source_sha256=sha(args.data),
                          swapped=args.out, swapped_sha256=sha(args.out),
                          expected_1=sum(r['expected'] == 1 for r in out))))


def recount(data, folder):
    """Independent recount from saved pairs (pooled and per-source; official state score = mean of sources)."""
    rows = [json.loads(x) for x in Path(data).read_text().splitlines() if x.strip()]
    saved = [json.loads(x) for x in (folder / 'pairs.jsonl').read_text().splitlines() if x.strip()]
    assert len(saved) == len(rows) == 4600
    states, sources = {}, {}
    for r, s in zip(rows, saved):
        assert all(s[k] == r[k] for k in ('source_name', 'state_idx', 'pair_idx', 'expected'))
        ok = s['pred'] == r['expected']
        assert bool(s['correct']) == ok
        states.setdefault((r['source_name'], r['state_idx']), []).append(ok)
        sources.setdefault(r['source_name'], []).append(ok)
    per = {}
    for src, oks in sources.items():
        panels = [p for (s, _), p in states.items() if s == src]
        per[src] = dict(pairs_total=len(oks), pairs_ok=sum(oks), pair_acc=sum(oks) / len(oks),
                        states_total=len(panels), states_ok=sum(all(p) for p in panels),
                        state_acc=sum(all(p) for p in panels) / len(panels))
    official = json.loads((folder / 'metrics.json').read_text())
    for src, m in per.items():
        assert m['pairs_ok'] == official['per_source'][src]['pairs_ok']
        assert m['states_ok'] == official['per_source'][src]['states_ok']
    preds = [s['pred'] for s in saved]
    return dict(per_source=per, pairs_ok=sum(m['pairs_ok'] for m in per.values()),
                states_ok=sum(m['states_ok'] for m in per.values()),
                predicted_1=preds.count(1), predicted_2=preds.count(2), unparseable=preds.count(0),
                official_overall=official['overall'])


def pass_dir(out, name, data):
    return Path(out) / name / Path(data).stem


def finish(folder, data, extra):
    audit = recount(data, folder)
    (folder / 'done.json').write_text(json.dumps(dict(audit, data=str(data), data_sha256=sha(data),
                                                      evaluator_sha256=EVALUATOR_SHA, **extra), indent=1) + '\n')
    print(json.dumps(dict(folder=str(folder), pairs_ok=audit['pairs_ok'], states_ok=audit['states_ok'],
                          predicted_1=audit['predicted_1'], unparseable=audit['unparseable'])), flush=True)


def cmd_gen(args):
    assert sha(EVALUATOR) == EVALUATOR_SHA
    todo = [d for d in args.data if not (pass_dir(args.out, args.name, d) / 'done.json').exists()]
    if not todo:
        return
    ipc = Path(os.environ.get('TMPDIR', '/tmp'))
    env = {k: v for k, v in os.environ.items() if not any(s in k for s in ('API_KEY', 'AUTH_TOKEN', 'ACCESS_TOKEN'))}
    for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'):
        env.pop(k, None)
    env.update(NO_PROXY='localhost,127.0.0.1,::1', no_proxy='localhost,127.0.0.1,::1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1',
               CUDA_HOME=str(RUNTIME / 'cuda'), PATH=str(RUNTIME / 'cuda/bin') + ':' + env.get('PATH', ''),
               LD_LIBRARY_PATH=str(RUNTIME / 'cuda/lib64') + ':' + env.get('LD_LIBRARY_PATH', ''))
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    # Server flags as in the earlier Piotr WebPRMBench run (job 367841), except concurrency (--batch-size; T 0, so
    # outputs do not depend on it): judges that write to the 1,024-token cap made batch 8 ~0.6 pairs/s.
    # No --max-total-tokens cap: a 32K KV pool would hold only a few ~5K-token prompts at once.
    cmd = [str(RUNTIME / 'venv/bin/python'), '-m', 'sglang.launch_server', '--model-path', args.model,
           '--host', '127.0.0.1', '--port', str(port), '--tp', '1', '--dtype', 'bfloat16',
           '--mem-fraction-static', '.70', '--context-length', '32768',
           '--max-running-requests', str(args.batch_size), '--random-seed', '42', '--disable-cuda-graph',
           '--disable-custom-all-reduce', '--skip-server-warmup', '--served-model-name', args.name]
    if args.tokenizer:
        cmd += ['--tokenizer-path', args.tokenizer]
    if args.chat_template:
        cmd += ['--chat-template', args.chat_template]
    if args.config:
        cmd += ['--decrypted-config-file', args.config]
    log_dir = Path(args.out) / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'server-command.json').write_text(json.dumps(cmd, indent=1) + '\n')
    log = (log_dir / f'server-{os.environ.get("SLURM_JOB_ID", "local")}.log').open('a')
    server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    http, url = build_opener(ProxyHandler({})), f'http://127.0.0.1:{port}'
    try:
        start = time.time()
        while True:
            if server.poll() is not None:
                raise RuntimeError('SGLang server exited during startup')
            try:
                with http.open(url + '/health', timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.time() - start > 900:
                raise TimeoutError('server not healthy after 900 s')
            time.sleep(3)
        print(json.dumps(dict(model=args.name, server_ready_seconds=round(time.time() - start))), flush=True)
        for data in todo:
            folder = pass_dir(args.out, args.name, data)
            if folder.exists():  # an unfinished earlier pass: keep it, start a fresh numbered folder
                folder.rename(folder.with_name(folder.name + f'.partial-{int(time.time())}'))
            folder.mkdir(parents=True)
            raw = (folder / 'raw.jsonl').open('x')
            t0 = time.time()

            def generate(texts, params):
                if params != PARAMS:
                    raise ValueError(f'sampling differs: {params}')
                if server.poll() is not None:
                    raise RuntimeError('SGLang server exited')
                payload = dict(text=texts, sampling_params=dict(temperature=0., top_p=1., max_new_tokens=1024,
                                                                stop=['</Answer>']))
                req = Request(url + '/generate', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                with http.open(req, timeout=600) as r:
                    resp = json.load(r)
                if not isinstance(resp, list) or len(resp) != len(texts):
                    raise ValueError('incomplete batch from server')
                for t, x in zip(texts, resp):
                    raw.write(json.dumps(dict(prompt_sha256=hashlib.sha256(t.encode()).hexdigest(),
                                              text=x['text'])) + '\n')
                raw.flush()
                return [x['text'] for x in resp]

            run_official(EVALUATOR, model=args.model, data=data, output=folder, generate=generate,
                         batch_size=args.batch_size)
            raw.close()
            finish(folder, data, dict(kind='generative', model=args.model, seconds=round(time.time() - t0),
                                      job_id=os.environ.get('SLURM_JOB_ID'), sampling=PARAMS, batch_size=args.batch_size))
    finally:
        server.terminate()
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()


def cmd_head(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from openwebrl.selection_head import CandidateSetHead, HeadConfig
    from openwebrl.selection_head_features import candidate_tokens, extract
    from openwebrl.selection_head_webprmbench import actor_prefix, choose, parse_pair, prompt_key
    assert sha(EVALUATOR) == EVALUATOR_SHA
    folder = pass_dir(args.out, args.name, args.data)
    if (folder / 'done.json').exists():
        return
    if folder.exists():
        folder.rename(folder.with_name(folder.name + f'.partial-{int(time.time())}'))
    folder.mkdir(parents=True)
    t0 = time.time()
    rows = [json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
    groups = {}
    for r in rows:
        context, responses = parse_pair(r['prompt'])
        g = groups.setdefault(context, [])
        g.extend(x for x in responses if x not in g)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='cuda:0',
                                                        attn_implementation='sdpa', local_files_only=True).eval()
    model.requires_grad_(False)
    ckpt = torch.load(args.head, map_location='cpu', weights_only=True)
    config = HeadConfig.from_json(ckpt['config'])
    head = CandidateSetHead(config)
    head.load_state_dict(ckpt['state_dict'], strict=True)
    head.eval().cuda()
    pad = processor.tokenizer.pad_token_id or 0
    encoded, out_rows, parity = {}, [], None
    preds_f = (folder / 'head-predictions.jsonl').open('x')
    position = [0]

    def generate(texts, params):
        nonlocal parity
        answers = []
        for text in texts:
            row = rows[position[0]]; position[0] += 1
            if row['prompt'] != text:
                raise ValueError('evaluator pair order changed')
            context, responses = parse_pair(text)
            if context not in encoded:
                prefix = processor(text=[actor_prefix(processor.tokenizer, context)], return_tensors='pt').to('cuda:0')
                cands = [candidate_tokens(processor.tokenizer, x) for x in groups[context]]
                state, feats = extract(model, prefix, cands, config.layers, pools=config.pools, batch_size=2, pad_id=pad)
                if parity is None:  # cached-prefix features must match a full forward pass (same 2% gate as K=1)
                    _, full = extract(model, prefix, cands[:2], config.layers, pools=config.pools, reuse_prefix=False,
                                      pad_id=pad)
                    parity = ((feats[:2].float() - full.float()).norm() / full.float().norm()).item()
                    if parity > .02:
                        raise ValueError(f'cached/full feature parity failed: {parity}')
                encoded[context] = (state.float().cpu().flatten(),
                                    {x: f.float().cpu().flatten() for x, f in zip(groups[context], feats)})
            state, feats = encoded[context]
            batch = torch.stack([feats[x] for x in responses])[None].cuda()
            with torch.no_grad():
                logits = head(batch, state[None].cuda())[0][0].float().cpu().tolist()
                swapped = head(batch.flip(1), state[None].cuda())[0][0].flip(0).float().cpu().tolist()
            pick = choose(logits)
            rec = dict(source_name=row['source_name'], state_idx=row['state_idx'], pair_idx=row['pair_idx'],
                       prompt_sha256=prompt_key(text), scores=logits, swapped_scores=swapped, prediction=pick,
                       swapped_prediction=choose(swapped))
            out_rows.append(rec)
            preds_f.write(json.dumps(rec) + '\n')
            answers.append(f'<Answer>Response {pick}</Answer>' if pick else 'TIE')
        preds_f.flush()
        return answers

    run_official(EVALUATOR, model=args.head, data=args.data, output=folder, generate=generate, batch_size=8)
    preds_f.close()
    order_error = max(max(abs(a - b) for a, b in zip(r['scores'], r['swapped_scores'])) for r in out_rows)
    flips = sum(r['prediction'] != r['swapped_prediction'] for r in out_rows)
    finish(folder, args.data, dict(kind='head', head=args.head, head_sha256=sha(args.head), actor=args.model,
                                   head_meta={k: v for k, v in ckpt.items() if k not in ('state_dict',)},
                                   feature_parity_relative_error=parity, max_order_score_difference=order_error,
                                   predictions_changed_by_order=flips, seconds=round(time.time() - t0),
                                   job_id=os.environ.get('SLURM_JOB_ID')))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('swap'); s.add_argument('--data', required=True); s.add_argument('--out', required=True)
    g = sub.add_parser('gen')
    for k in ('name', 'model', 'out'):
        g.add_argument('--' + k, required=True)
    for k in ('tokenizer', 'chat-template', 'config'):
        g.add_argument('--' + k)
    g.add_argument('--data', nargs='+', required=True)
    g.add_argument('--batch-size', type=int, default=8)
    h = sub.add_parser('head')
    for k in ('name', 'head', 'model', 'data', 'out'):
        h.add_argument('--' + k, required=True)
    args = ap.parse_args()
    dict(swap=cmd_swap, gen=cmd_gen, head=cmd_head)[args.cmd](args)


if __name__ == '__main__':
    main()
