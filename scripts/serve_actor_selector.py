#!/usr/bin/env python3
"""Sidecar selector over the actor's own weights, for live inference (one GPU).

  --mode head  frozen OpenWebRL-4B-SFT hidden-state features + saved pointwise head
  --mode kev   Kev-style LoRA pass over the actor (saved adapter, pointer and <decide> embedding)

POST /select with the actor's exact prompt text, the current screenshot (base64) and each candidate's generated
token ids; returns per-candidate scores and telemetry: tokens processed, seconds and an estimated FLOP count
(2 x parameters x tokens for the language model; 2 x vision parameters x image patches). Scoring follows the
offline evaluation code paths. At startup the server re-scores saved branching states and refuses to serve if
any pick differs from the saved offline picks.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time

R = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008")


def build(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from openwebrl.selection_head import CANDIDATE_POOLS, CandidateSetHead, HeadConfig
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                        attn_implementation="sdpa", local_files_only=True).eval()
    model.requires_grad_(False)
    vision = sum(p.numel() for n, p in model.named_parameters() if ".visual." in n or n.startswith("visual."))
    language = sum(p.numel() for p in model.parameters()) - vision
    state = dict(processor=processor, tokenizer=tokenizer, model=model, vision_params=vision, language_params=language)
    if args.mode == "head":
        saved = torch.load(args.head, map_location="cpu")
        objective = saved.get("objective", "soft_ce")
        if objective not in ("soft_ce", "pairwise_teacher"):
            raise ValueError(f"Unsupported saved head objective: {objective}")
        config = HeadConfig.from_json(saved["config"])
        head = CandidateSetHead(config); head.load_state_dict(saved["state_dict"]); head.to("cuda:0").eval()
        state.update(head=head, config=config, score_head="value" if objective == "pairwise_teacher" else "select",
                     identity=dict(head=str(args.head), objective=objective,
                                   head_sha256=hashlib.sha256(Path(args.head).read_bytes()).hexdigest()))
    else:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        from openwebrl.selection_adapter import add_scoring_adapter
        from openwebrl.selection_kev import KevStyleScorer
        model = add_scoring_adapter(model, top_layers=36, rank=16, alpha=32)
        adapter = Path(args.kev) / "adapter" / "adapter_model.safetensors"
        set_peft_model_state_dict(model, load_file(str(adapter)))
        decide_id = tokenizer.convert_tokens_to_ids("<|fim_pad|>")
        scorer = KevStyleScorer(model, decide_id).to("cuda:0")
        missing = scorer.load_state_dict(torch.load(Path(args.kev) / "head.pt", map_location="cpu"), strict=False).missing_keys
        if any(not k.startswith("model.") for k in missing):
            raise SystemExit("Kev-style pointer/decide checkpoint incomplete")
        model.eval(); scorer.eval()
        state.update(model=model, scorer=scorer, identity=dict(kev=str(args.kev), adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                                                              head_sha256=hashlib.sha256((Path(args.kev) / "head.pt").read_bytes()).hexdigest()))
    return state


def score(state, args, prompt_text, image, candidate_ids):
    """image: one PIL image or the actor's full list of images, in prompt order."""
    """Scores for one decision plus processed-token and FLOP telemetry."""
    import torch
    from openwebrl.selection_head_features import candidate_tokens_from_ids, extract
    proc, tok = state["processor"], state["tokenizer"]
    images = image if isinstance(image, list) else [image]
    prefix = proc(text=[prompt_text], images=images, return_tensors="pt").to("cuda:0")
    cands = [candidate_tokens_from_ids(tok, ids) for ids in candidate_ids]
    torch.cuda.synchronize(); started = time.monotonic()
    with torch.no_grad():
        if args.mode == "head":
            cfg = state["config"]
            s, feats = extract(state["model"], prefix, cands, cfg.layers, pools=cfg.pools, batch_size=5,
                               pad_id=tok.pad_token_id or 0)
            # offline features were stored as float16 before scoring; round identically
            feats, s = feats.to(torch.float16).to(torch.float32), s.to(torch.float16).to(torch.float32)
            outputs = state["head"](feats.reshape(1, len(cands), -1), s.reshape(1, -1))
            logits = outputs[1 if state["score_head"] == "value" else 0][0]
        else:
            logits = state["scorer"](prefix, [cands])[0]
    torch.cuda.synchronize(); seconds = time.monotonic() - started
    prompt_tokens = int(prefix["input_ids"].shape[1]); cand_tokens = sum(len(ids) for ids, _ in cands)
    patches = int(prefix["image_grid_thw"].prod(-1).sum()) if "image_grid_thw" in prefix else 0
    flops = 2 * state["language_params"] * (prompt_tokens + cand_tokens) + 2 * state["vision_params"] * patches
    return [float(x) for x in logits.float().cpu()], dict(input_tokens=prompt_tokens + cand_tokens, output_tokens=0,
                                                          seconds=seconds, prompt_tokens=prompt_tokens, candidate_tokens=cand_tokens,
                                                          image_patches=patches, flops=flops)


def self_test(state, args, n=6):
    """Re-score saved branching states; every pick must equal the saved offline pick."""
    from PIL import Image
    panels = [json.loads(l) for l in open(R / "branch124/panels.jsonl")][:n]
    prompts = {json.loads(l)["state_id"]: json.loads(l)["prompt_text"] for l in open(R / "branch124/states.jsonl")}
    saved = json.loads(Path(args.reference_picks).read_text())["branch_head_picks"]
    results = []
    for row in panels:
        scores, _ = score(state, args, prompts[row["state_id"]], Image.open(row["image"]).convert("RGB"), row["candidate_token_ids"])
        pick = max(range(len(scores)), key=lambda i: scores[i])
        results.append(dict(state=row["state_id"], pick=pick, saved=saved[row["state_id"]], scores=scores))
    if any(r["pick"] != r["saved"] for r in results):
        raise SystemExit("self-test failed: live picks differ from saved offline picks " + json.dumps(results))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["head", "kev"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default=str(R / "heads/pointwise/head.pt"))
    ap.add_argument("--kev", default=str(R / "kev2-353914"))
    ap.add_argument("--reference-picks", help="metrics.json with the saved offline branch picks for the self-test")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    if args.reference_picks is None:
        args.reference_picks = str(R / ("heads/pointwise" if args.mode == "head" else "kev2-353914") / "metrics.json")
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from PIL import Image
    state = build(args)
    checks = self_test(state, args)
    import threading
    gpu_lock = threading.Lock()   # one scoring pass at a time: the model and its KV cache are not thread-safe
    app = FastAPI()
    card = dict(mode="actor_scalar", selector=args.mode, model=args.model, language_params=state["language_params"],
                vision_params=state["vision_params"], self_test=dict(states=len(checks), all_picks_match=True), **state["identity"])

    @app.get("/health")
    def health():
        return card

    @app.post("/select")
    def select(payload: dict):
        try:
            # the actor's exact image list (base64 data URLs, prompt order); the screenshot alone only for one-image prompts
            raw = payload.get("images") or [payload["screenshot"]]
            images = [Image.open(io.BytesIO(base64.b64decode(x.split(",", 1)[1] if x.startswith("data:") else x))).convert("RGB") for x in raw]
            ids = payload["candidate_token_ids"]
            if len(ids) != 5 or not all(isinstance(x, list) and x for x in ids):
                raise ValueError("five non-empty candidate token-id lists required")
            queued = time.monotonic()
            with gpu_lock:
                waited = time.monotonic() - queued
                scores, telemetry = score(state, args, payload["prompt_text"], images, ids)
            telemetry["queue_seconds"] = waited   # time waiting for other episodes' scoring; excluded from "seconds"
        except Exception as exc:
            import logging, traceback
            logging.getLogger("serve_actor_selector").error("select failed: %s", traceback.format_exc())
            raise HTTPException(422, f"{type(exc).__name__}: {exc}")
        return dict(scores=scores, **telemetry)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
