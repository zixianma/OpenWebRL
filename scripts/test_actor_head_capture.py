#!/usr/bin/env python3
"""Parity and overhead test for the KV-cache-reusing selection head (one GPU).

Phase A: start the actor's sglang server exactly as the Online-Mind2Web controller does, plus the capture hooks
(`openwebrl.actor_head_capture`). For saved branching states, generate five candidates per state as the actor does,
then send each candidate's follow-up request; build head features from the captured hidden states.
Reference: the same candidate token ids teacher-forced through the Hugging Face actor (`extract`, the offline and
sidecar path). Reports feature agreement, head-score agreement, pick agreement, follow-up alignment, extra tokens.
Phase B: the same server without hooks on the same workload, for latency and throughput overhead.
"""
import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
R = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008")
RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
MODEL = "/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT"
SAMPLING = dict(temperature=0.7, top_p=0.9, max_new_tokens=1024)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


GRAPHS = False      # --cuda-graph: decode through captured CUDA graphs (capture via graph buffers)
NGRAM = False       # --ngram: n-gram speculative decoding (TARGET_VERIFY forwards; capture keeps accepted rows)


def start_server(port, hooks, capture_dir, log):
    from openwebrl.actor_head_capture import hook_specs
    cmd = [str(RUNTIME / "venv/bin/python"), "-m", "sglang.launch_server", "--model-path", MODEL, "--host", "127.0.0.1",
           "--port", str(port), "--dtype", "bfloat16", "--tp", "1", "--mem-fraction-static", ".45",
           "--context-length", "32768", "--max-running-requests", "40", "--chunked-prefill-size", "4096",
           "--random-seed", "4500"] + (["--cuda-graph-max-bs", "48"] if GRAPHS else ["--disable-cuda-graph"]) \
          + (["--speculative-algorithm", "NGRAM"] if NGRAM else [])
    if hooks:
        cmd += ["--forward-hooks", json.dumps(hook_specs())]
    env = dict(os.environ, PYTHONPATH=str(REPO), ACTORHEAD_DIR=str(capture_dir))
    proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    import httpx
    for _ in range(600):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise SystemExit(f"server exited; see {log}")
        time.sleep(2)
    raise SystemExit("server did not become healthy")


def stop(proc):
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)


def capture_root():
    return Path(f"/dev/shm/actorhead-{os.getenv('SLURM_JOB_ID', 'local')}")


async def workload(port, states, capture, tokenizer, concurrency):
    capture_dir = capture_root()
    """Generate five candidates per state (and, with capture, their follow-ups); returns per-state records."""
    import httpx
    url = f"http://127.0.0.1:{port}/generate"
    sem = asyncio.Semaphore(concurrency)

    async def post(client, payload):
        t = time.monotonic()
        r = await client.post(url, json=payload, timeout=600)
        r.raise_for_status()
        return r.json(), time.monotonic() - t

    async def one(client, st, k):
        async with sem:
            data = "data:image/png;base64," + base64.b64encode(Path(st["image"]).read_bytes()).decode()
            gens = []
            started = time.monotonic()
            reqs = []
            for i in range(5):
                p = dict(text=st["prompt_text"], image_data=[data], return_logprob=True,
                         sampling_params=dict(SAMPLING, sampling_seed=1000 * k + i))
                if capture:
                    p["rid"] = f"actorhead-{st['state_id']}-{i}"
                reqs.append(post(client, p))
            gens = await asyncio.gather(*reqs)
            generation_seconds = time.monotonic() - started
            out = dict(state=st["state_id"], generation_seconds=generation_seconds,
                       request_seconds=[s for _, s in gens], prompt_tokens=[g["meta_info"]["prompt_tokens"] for g, _ in gens],
                       cached_tokens=[g["meta_info"].get("cached_tokens") for g, _ in gens],
                       output_ids=[[t[1] for t in g["meta_info"]["output_token_logprobs"]] for g, _ in gens])
            if capture:
                from openwebrl.actor_head_scoring import load_export, tail_text
                t0 = time.monotonic()
                exports = [await asyncio.to_thread(load_export, capture_dir, f"actorhead-{st['state_id']}-{i}") for i in range(5)]
                need = [i for i, g in enumerate(exports) if not g.get("final_fed")]   # end token not fed (length stop)
                tails = await asyncio.gather(*(post(client, dict(
                    text=st["prompt_text"] + tail_text(tokenizer, out["output_ids"][i]), image_data=[data],
                    rid=f"actorhead-{st['state_id']}-{i}-tail", sampling_params=dict(temperature=0.0, max_new_tokens=1)))
                    for i in need))
                out.update(tail_seconds=time.monotonic() - t0, exports=exports, follow_ups=need,
                           tail_prompt_tokens=[g["meta_info"]["prompt_tokens"] for g, _ in tails],
                           tail_cached_tokens=[g["meta_info"].get("cached_tokens") for g, _ in tails])
            return out

    async with httpx.AsyncClient(trust_env=False) as client:
        started = time.monotonic()
        rows = await asyncio.gather(*(one(client, st, k) for k, st in enumerate(states)))
        return rows, time.monotonic() - started


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", type=int, default=40)
    ap.add_argument("--skip-plain", action="store_true", help="skip the no-hook latency phase")
    ap.add_argument("--cuda-graph", action="store_true", help="serve with CUDA graphs (both phases)")
    ap.add_argument("--concurrency", type=int, default=8, help="states in flight (5 requests each)")
    ap.add_argument("--ngram", action="store_true", help="serve with n-gram speculative decoding (both phases)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    global GRAPHS, NGRAM
    GRAPHS, NGRAM = args.cuda_graph, args.ngram
    SAMPLING["temperature"] = args.temperature
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from openwebrl.actor_head_scoring import candidate_features, load_export, score
    from openwebrl.selection_head import CandidateSetHead, HeadConfig
    from openwebrl.selection_head_features import candidate_tokens_from_ids, extract

    prompts = {json.loads(l)["state_id"]: json.loads(l)["prompt_text"] for l in open(R / "branch124/states.jsonl")}
    states = [dict(json.loads(l), prompt_text=None) for l in open(R / "branch124/panels.jsonl")][:args.states]
    for st in states:
        st["prompt_text"] = prompts[st["state_id"]]
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    tok = processor.tokenizer
    saved = torch.load(R / "heads/pointwise/head.pt", map_location="cpu")
    config = HeadConfig.from_json(saved["config"])
    head = CandidateSetHead(config); head.load_state_dict(saved["state_dict"]); head.eval()
    layers = config.layers

    capture_dir = capture_root()
    capture_dir.mkdir(parents=True, exist_ok=True)
    report = dict(states=len(states), concurrency=args.concurrency, layers=list(layers), pools=list(config.pools),
                  cuda_graph=GRAPHS, ngram=NGRAM, sampling=dict(SAMPLING))

    # Phase A: capture server
    port = free_port()
    server = start_server(port, True, capture_dir, args.out / "server-hooks.log")
    try:
        asyncio.run(workload(port, states[:2], False, tok, args.concurrency))          # warm-up (not captured)
        # hooks active but no capture requests or follow-ups: isolates the hooks' own cost from follow-up load
        rows_g, wall_g = asyncio.run(workload(port, states, False, tok, args.concurrency))
        report["hooks_generation_only"] = dict(wall_seconds=wall_g, mean_generation_seconds=sum(r["generation_seconds"] for r in rows_g) / len(rows_g),
                                               generated_tokens=sum(len(x) for r in rows_g for x in r["output_ids"]))
        rows, wall = asyncio.run(workload(port, states, True, tok, args.concurrency))
        report["hooks"] = dict(wall_seconds=wall, requests=5 * len(rows),
                               mean_generation_seconds=sum(r["generation_seconds"] for r in rows) / len(rows),
                               mean_tail_seconds=sum(r["tail_seconds"] for r in rows) / len(rows),
                               generated_tokens=sum(len(x) for r in rows for x in r["output_ids"]),
                               tail_uncached_tokens=sum(p - c for r in rows for p, c in zip(r["tail_prompt_tokens"], r["tail_cached_tokens"])),
                               actor_uncached_prompt_tokens=sum(p - c for r in rows for p, c in zip(r["prompt_tokens"], r["cached_tokens"])))
        report["follow_ups"] = sum(len(r["follow_ups"]) for r in rows)
        captured, failures = {}, []
        for r in rows:
            per = []
            try:
                for i in range(5):
                    gen = r["exports"][i]
                    tail = load_export(capture_dir, f"actorhead-{r['state']}-{i}-tail") if i in r["follow_ups"] else None
                    assert gen["output_ids"] == r["output_ids"][i], "export ids differ from the HTTP response"
                    per.append(candidate_features(gen, tail, tok, layers, config.pools))
                captured[r["state"]] = per
            except Exception as exc:
                failures.append(dict(state=r["state"], error=f"{type(exc).__name__}: {exc}"))
        report["capture_failures"] = failures
        rows = [{k: v for k, v in r.items() if k != "exports"} for r in rows if r["state"] in captured]
    finally:
        stop(server)

    # Reference: teacher-forced Hugging Face features for the same generated ids
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                        attn_implementation="sdpa", local_files_only=True).eval()
    head_gpu = CandidateSetHead(config); head_gpu.load_state_dict(saved["state_dict"]); head_gpu.to("cuda:0").eval()
    by_state = {st["state_id"]: st for st in states}
    per_state, cos = [], []
    for r in rows:
        st = by_state[r["state"]]
        image = Image.open(st["image"]).convert("RGB")
        prefix = processor(text=[st["prompt_text"]], images=[image], return_tensors="pt").to("cuda:0")
        cands = [candidate_tokens_from_ids(tok, ids) for ids in r["output_ids"]]
        with torch.no_grad():
            s, feats = extract(model, prefix, cands, layers, pools=config.pools, batch_size=5, pad_id=tok.pad_token_id or 0)
            f16 = feats.to(torch.float16).to(torch.float32); s16 = s.to(torch.float16).to(torch.float32)
            ref = [float(x) for x in head_gpu(f16.reshape(1, 5, -1), s16.reshape(1, -1))[0][0].cpu()]
        cap = captured[r["state"]]
        if len(per_state) < 6:   # diagnostic: captured prompt-state per layer vs every HF hidden-state index
            with torch.no_grad():
                hs = model(**prefix, output_hidden_states=True).hidden_states
            diag = {int(l): {j: float(torch.nn.functional.cosine_similarity(cap[0][0][k].float().cuda(), hs[j][0, -1].float(), dim=0))
                             for j in range(max(0, l - 2), min(len(hs), l + 3))} for k, l in enumerate(layers)}
            report.setdefault("layer_diagnostic", []).append(dict(state=r["state"], hidden_states=len(hs), cosine=diag))
        live = score(head, config, [(a, b) for a, b, _ in cap])
        cf = torch.stack([b for _, b, _ in cap]).float()
        c = torch.nn.functional.cosine_similarity(cf.reshape(-1, cf.shape[-1]), feats.float().cpu().reshape(-1, cf.shape[-1]), dim=-1)
        sc = torch.nn.functional.cosine_similarity(cap[0][0].float(), s.float().cpu(), dim=-1)
        cos += c.tolist()
        per_state.append(dict(state=r["state"], hf_prompt_tokens=int(prefix["input_ids"].shape[1]),
                              server_prompt_tokens=r["prompt_tokens"][0], live_scores=live, reference_scores=ref,
                              live_pick=max(range(5), key=lambda i: live[i]), reference_pick=max(range(5), key=lambda i: ref[i]),
                              max_score_abs_diff=max(abs(a - b) for a, b in zip(live, ref)),
                              min_feature_cosine=float(c.min()), min_state_cosine=float(sc.min()),
                              tails_aligned=sum(x["aligned"] for _, _, x in cap)))
    del model
    torch.cuda.empty_cache()
    if not per_state:
        (args.out / "report.json").write_text(json.dumps(report, indent=1))
        raise SystemExit("no state captured: " + json.dumps(report.get("capture_failures", [])[:3]))
    report["parity"] = dict(
        picks_agree=sum(p["live_pick"] == p["reference_pick"] for p in per_state), states=len(per_state),
        prompt_token_counts_match=sum(p["hf_prompt_tokens"] == p["server_prompt_tokens"] for p in per_state),
        tails_aligned=sum(p["tails_aligned"] for p in per_state), candidates=5 * len(per_state),
        max_score_abs_diff=max(p["max_score_abs_diff"] for p in per_state),
        mean_score_abs_diff=sum(p["max_score_abs_diff"] for p in per_state) / len(per_state),
        min_feature_cosine=min(cos), mean_feature_cosine=sum(cos) / len(cos))
    report["per_state"] = per_state

    if args.skip_plain:
        (args.out / "report.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: v for k, v in report.items() if k != "per_state"}, indent=1))
        return
    # Phase B: same workload without hooks
    server = start_server(port, False, capture_dir, args.out / "server-plain.log")
    try:
        asyncio.run(workload(port, states[:2], False, tok, args.concurrency))
        rows_b, wall_b = asyncio.run(workload(port, states, False, tok, args.concurrency))
    finally:
        stop(server)
    report["plain"] = dict(wall_seconds=wall_b, mean_generation_seconds=sum(r["generation_seconds"] for r in rows_b) / len(rows_b),
                           generated_tokens=sum(len(x) for r in rows_b for x in r["output_ids"]))
    h, p = report["hooks"], report["plain"]
    report["overhead"] = dict(
        generation_seconds_per_token_hooks=h["mean_generation_seconds"] * len(rows) / h["generated_tokens"],
        generation_seconds_per_token_plain=p["mean_generation_seconds"] * len(rows_b) / p["generated_tokens"],
        tail_seconds_per_state=h["mean_tail_seconds"],
        generation_seconds_per_token_hooks_no_capture=report["hooks_generation_only"]["mean_generation_seconds"] * len(rows)
        / report["hooks_generation_only"]["generated_tokens"])
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_state"}, indent=1))


if __name__ == "__main__":
    main()
