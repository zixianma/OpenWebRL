#!/usr/bin/env python3
"""Piotr SelectionARM served by sglang (CUDA graphs, batching) behind the `serve_arm.py --mode selection` contract.

`assemble` builds an sglang-loadable model directory: the released selection weights (linked), the actor's config,
tokenizer and chat template (`serve_arm.py` loads the weights with the same base config after checking the
architecture, vocabulary and template match), and a preprocessor config with the server's image limits
(min_pixels 65,536, max_pixels 262,144).

`serve` exposes POST /select with the same request and response fields as `serve_arm.py`: the same canonical
prompt builder, greedy decoding, the same JSON schema (enforced by sglang's xgrammar backend), and 128 new tokens.
It forwards to an sglang server running the assembled directory.
"""
import argparse
import base64
import json
from pathlib import Path
import shutil
import sys
import time

BASE = Path("/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")
SELECTION = Path("/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89")
SOURCE_ROOT = "/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source"


def assemble(out, selection=SELECTION):
    out.mkdir(parents=True, exist_ok=True)
    for f in selection.glob("model-*.safetensors"):
        (out / f.name).unlink(missing_ok=True); (out / f.name).symlink_to(f)
    shutil.copy(selection / "model.safetensors.index.json", out)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "chat_template.json", "merges.txt", "vocab.json", "processor_config.json", "video_preprocessor_config.json"):
        if (BASE / name).exists():
            shutil.copy(BASE / name, out)
    pre = json.loads((BASE / "preprocessor_config.json").read_text())
    pre["size"] = {"shortest_edge": 65536, "longest_edge": 262144}      # serve_arm.py: min_pixels / max_pixels
    pre["min_pixels"], pre["max_pixels"] = 65536, 262144
    (out / "preprocessor_config.json").write_text(json.dumps(pre, indent=2))
    print(json.dumps(dict(assembled=str(out), weights=str(selection), config=str(BASE))))


def serve(args):
    import httpx
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from transformers import AutoProcessor
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openwebrl.arm_inference import load_selection_builder, selection_messages, selection_schema
    processor = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    builder = load_selection_builder(SOURCE_ROOT)
    app = FastAPI()
    client = httpx.AsyncClient(trust_env=False, timeout=args.timeout)

    @app.get("/health")
    def health():
        return {"mode": "selection", "model": str(args.model), "base": str(BASE), "adapter": None, "max_pixels": 262144,
                "serving": "sglang", "selection_decoding": "canonical_no_cot_json_schema/xgrammar(sglang)"}

    @app.post("/select")
    async def select(request: dict):
        if request.get("mode") != "selection" or not 1 <= len(request["candidates"]) <= 5:
            raise HTTPException(400, "Wrong mode or unsupported candidate count")
        started = time.monotonic()
        image_bytes = base64.b64decode(request["screenshot"], validate=True)
        messages = selection_messages(builder, request["task"], request["url"], request["history"], request["candidates"],
                                      image_bytes, request.get("candidate_representation", "full"))
        messages[1]["content"][1] = {"type": "image"}         # as serve_arm.py: keep the builder's text blocks
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        payload = dict(text=text, image_data=["data:image/png;base64," + request["screenshot"]],
                       sampling_params=dict(temperature=0.0, max_new_tokens=args.max_new_tokens,
                                            json_schema=json.dumps(selection_schema(len(request["candidates"])))))
        r = await client.post(args.sglang + "/generate", json=payload)
        r.raise_for_status()
        out = r.json()
        meta = out["meta_info"]
        return {"raw": out["text"], "candidate_representation": request.get("candidate_representation", "full"),
                "input_tokens": meta["prompt_tokens"], "output_tokens": meta["completion_tokens"],
                "cached_tokens": meta.get("cached_tokens"), "seconds": time.monotonic() - started}

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assemble"); a.add_argument("--out", type=Path, required=True)
    a.add_argument("--selection", type=Path, default=SELECTION, help="merged SelectionARM weights (default: Piotr's release)")
    s = sub.add_parser("serve"); s.add_argument("--sglang", required=True); s.add_argument("--port", type=int, required=True)
    s.add_argument("--model", type=Path, default=SELECTION, help="weights the sglang server runs (reported by /health)")
    s.add_argument("--max-new-tokens", type=int, default=128); s.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()
    assemble(args.out, args.selection) if args.command == "assemble" else serve(args)


if __name__ == "__main__":
    main()
