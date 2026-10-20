#!/usr/bin/env python3
"""Frozen selection head served for a different actor: encode the actor's prompt and its candidates with the model the
head was trained on (here OpenWebRL-4B-SFT) and score them with the unchanged head.

The encoding is the offline feature extractor's (`selection_head_features.extract`: prompt + current screenshot, then
each candidate teacher-forced after the prompt), so the head sees exactly the feature definition it was trained on.
The actor and the encoder must share the tokenizer and chat template (checked at start-up).

  serve_frozen_head_scorer.py --model ENCODER --head HEAD.pt --actor ACTOR --port P
  GET  /health -> {model, head_sha256, layers}
  POST /score  {prompt_text, image, candidate_ids: [[int]]} -> {scores, input_tokens, candidate_tokens, seconds, flops}
"""
import argparse
import asyncio
import base64
import hashlib
import io
import threading
import time
from pathlib import Path


def same_tokenizer(a, b):
    files = ("vocab.json", "merges.txt", "tokenizer.json", "chat_template.jinja")
    digest = lambda d, f: hashlib.sha256((Path(d) / f).read_bytes()).hexdigest()
    return all(digest(a, f) == digest(b, f) for f in files)


def decode_image(value):
    from PIL import Image
    if isinstance(value, list):
        value = value[-1]                                # the current screenshot is the last image
    if isinstance(value, str) and value.startswith("data:"):
        value = value.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--head", required=True); ap.add_argument("--actor", required=True)
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--params", type=int, default=4022468096 + 415347712, help="encoder parameters for FLOP accounting")
    ap.add_argument("--batch-size", type=int, default=5)
    args = ap.parse_args()
    if not same_tokenizer(args.model, args.actor):
        raise SystemExit("actor and encoder tokenizers/chat templates differ; candidate ids would not transfer")

    import torch
    import uvicorn
    from fastapi import FastAPI
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from openwebrl.actor_head_scoring import bundle, score
    from openwebrl.selection_head_features import candidate_tokens_from_ids, extract

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device,
                                                        attn_implementation="sdpa", local_files_only=True).eval()
    model.requires_grad_(False)
    b = bundle(args.head, args.model)
    config, lock = b["config"], threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health():
        return dict(mode="frozen_head_scorer", model=str(Path(args.model).resolve()), head_sha256=b["head_sha256"],
                    layers=list(config.layers), pools=list(config.pools))

    def run(req):
        started = time.monotonic()
        with lock:
            prefix = processor(text=[req["prompt_text"]], images=[decode_image(req["image"])], return_tensors="pt").to(args.device)
            cands = [candidate_tokens_from_ids(tokenizer, ids) for ids in req["candidate_ids"]]
            state, feats = extract(model, prefix, cands, tuple(config.layers), tuple(config.pools),
                                   batch_size=args.batch_size, pad_id=tokenizer.pad_token_id or 0)
        scores = score(b["head"], config, [(state.cpu(), f.cpu()) for f in feats], score_head=b["score_head"])
        n_prefix, n_cand = int(prefix["input_ids"].shape[1]), sum(len(ids) for ids, _ in cands)
        return dict(scores=scores, input_tokens=n_prefix, candidate_tokens=n_cand, seconds=time.monotonic() - started,
                    flops=2 * args.params * (n_prefix + n_cand) + 2 * b["head_params"] * len(cands),
                    head_sha256=b["head_sha256"])

    @app.post("/score")
    async def score_endpoint(req: dict):
        return await asyncio.to_thread(run, req)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
