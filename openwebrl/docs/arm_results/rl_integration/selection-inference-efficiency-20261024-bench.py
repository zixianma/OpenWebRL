#!/usr/bin/env python3
"""Replay saved DeepShop steps against a local sglang actor server and time five-candidate decisions.

Each episode's step-t prompt is rebuilt from the saved final prompt (turns up to t, with the step-t screenshot block
restored; earlier observations are text-only, as in the live loop) and its saved step-t screenshot. EPISODES run
concurrently (as the live eval's 8 browsers per GPU); within an episode, steps run back to back (no browser time).
Request format matches the live eval: /generate with text, image_data (data URL), return_logprob, T 1.0 / top-p 0.9,
1,024 new tokens.

Modes (how one decision's five candidates are requested):
  sep      five concurrent requests, one sampling_seed each (current eval)
  prefill  one max_new_tokens=0 request first (caches the prompt), then the five seeded requests
  n5       one request with sampling_params n=5 (one seed for the request)

  bench_replay.py --port P --mode sep --label A --out OUT --samples S1.json [S2.json ...] [--steps 12]
"""
import argparse
import asyncio
import json
import re
import statistics as st
import time
from pathlib import Path

import aiohttp

SHOT = '\nscreenshot:\n<|vision_start|><|image_pad|><|vision_end|>\n</observation>'


def steps_of(path, n):
    x = json.loads(Path(path).read_text())
    t, images = x['llm_input_texts'], x['images']
    starts = [m.start() for m in re.finditer(r'<\|im_start\|>assistant\n', t)]
    out = []
    for i, a in enumerate(starts[:n]):
        head = t[:a]
        if SHOT not in head[-len(SHOT) - 400:]:
            k = head.rfind('\n\n</observation>')
            head = head[:k] + SHOT + head[k + len('\n\n</observation>'):]
        out.append(dict(text=head + '<|im_start|>assistant\n<think>\n', image=images[i][0]))
    return out


async def post(session, url, payload):
    async with session.post(url, json=payload) as r:
        r.raise_for_status()
        return await r.json()


async def episode(session, base, eid, steps, mode, rows):
    params = dict(temperature=1.0, top_p=0.9, max_new_tokens=1024)
    for s, step in enumerate(steps):
        body = dict(text=step['text'], image_data=[step['image']], return_logprob=True)
        t0 = time.monotonic()
        pre = None
        if mode == 'prefill':
            r = await post(session, base + '/generate', dict(body, sampling_params=dict(params, max_new_tokens=0)))
            pre = dict(prompt_tokens=r['meta_info']['prompt_tokens'], cached_tokens=r['meta_info'].get('cached_tokens', 0),
                       seconds=time.monotonic() - t0)
        if mode == 'n5':
            outs = await post(session, base + '/generate', dict(body, sampling_params=dict(params, n=5, sampling_seed=1000 * eid + s)))
        else:
            outs = await asyncio.gather(*(post(session, base + '/generate',
                                               dict(body, sampling_params=dict(params, sampling_seed=100000 * eid + 10 * s + j)))
                                          for j in range(5)))
        wall = time.monotonic() - t0
        metas = [o['meta_info'] for o in outs]
        texts = [o['text'] for o in outs]
        rows.append(dict(episode=eid, step=s, wall=wall, prefill=pre, distinct_outputs=len(set(texts)),
                         requests=[{k: m.get(k) for k in ('prompt_tokens', 'cached_tokens', 'completion_tokens', 'e2e_latency')}
                                   for m in metas]))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, required=True); ap.add_argument('--mode', choices=('sep', 'prefill', 'n5'), required=True)
    ap.add_argument('--label', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=12); ap.add_argument('--samples', nargs='+', required=True)
    args = ap.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    episodes = [steps_of(p, args.steps) for p in args.samples]
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        async with session.post(base + '/flush_cache') as r:
            await r.read()
        rows = []
        t0 = time.monotonic()
        await asyncio.gather(*(episode(session, base, e, steps, args.mode, rows) for e, steps in enumerate(episodes)))
        total = time.monotonic() - t0
    reqs = [q for r in rows for q in r['requests']]
    comp = sum(q['completion_tokens'] for q in reqs)
    prompt = sum(q['prompt_tokens'] for q in reqs) + sum(r['prefill']['prompt_tokens'] for r in rows if r['prefill'])
    cached = sum(q['cached_tokens'] or 0 for q in reqs) + sum(r['prefill']['cached_tokens'] for r in rows if r['prefill'])
    walls = sorted(r['wall'] for r in rows)
    summary = dict(label=args.label, mode=args.mode, episodes=len(episodes), decisions=len(rows), total_seconds=round(total, 1),
                   decision_mean=round(st.mean(walls), 2), decision_p50=round(walls[len(walls) // 2], 2),
                   decision_p90=round(walls[int(.9 * len(walls))], 2), completion_tokens=comp,
                   completion_tokens_per_decision=round(comp / len(rows), 1), tokens_per_second=round(comp / total, 1),
                   seconds_per_100_tokens_per_decision=round(100 * st.mean(r['wall'] / max(1, max(q['completion_tokens'] for q in r['requests'])) for r in rows), 3),
                   uncached_prompt_tokens=prompt - cached, cached_share=round(cached / prompt, 4),
                   prefill_seconds_mean=round(st.mean(r['prefill']['seconds'] for r in rows), 3) if args.mode == 'prefill' else None,
                   distinct_outputs_mean=round(st.mean(r['distinct_outputs'] for r in rows), 2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'{args.label}.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
