#!/usr/bin/env python3
"""Branching-state panels + soft supervision targets -> LLaMA-Factory SFT dataset for Piotr's SelectionARM recipe (CPU).

Port of Piotr's ``data_generation/openwebrl_actor/build_selection_sft.py`` to our branching states. Each sample is the
ShareGPT triple his builder writes (system prompt, user text with ``<image>`` where the screenshot sits, assistant
``{"selection": N}``), but the prompt comes from the exact path our deployed ``piotr5`` selector uses at inference:
the crossbench selector payload (task, active-tab url, ``split_response`` of every past response and candidate) goes
through ``openwebrl.arm_inference.selection_messages`` with the pinned released ``selection_prompt.py`` and canonical
flags, i.e. what ``scripts/serve_arm_sglang.py`` renders. LLaMA-Factory's ``qwen3_vl`` template renders these
messages to the same prompt string the selector server sends to sglang (tests/test_build_selection_sft_ours.py).

Conventions, following Piotr's builder (and his teacher-batch builder):
  * candidate order is the actor's sample order, unpermuted (his builder never shuffles; inference uses sample order);
  * one target index, 1-based, no chain of thought (his no-CoT variant);
  * states with fewer than two distinct action strings are skipped ("nothing to select");
  * additionally, states whose soft target is all-equal carry no signal and are skipped.
Ties: Piotr's teacher returns exactly one pick per candidate set, so his data has no ties. Our targets are vote shares
or tie-split outcome indicators. ``--ties all`` (default) writes one sample per tied argmax index (the SFT analogue of
cross-entropy on the uniform distribution over the argmax set); ``skip`` drops tied states (one unambiguous sample per
state, Piotr's one-sample-per-set shape); ``first`` keeps the lowest tied index (position-biased; diagnostics only).
Response format (``--format``). The selector server renders the prompt with the base model's chat template, whose
generation prompt is ``<|im_start|>assistant\n<think>\n``; sglang then decodes under the selection JSON schema.
  * ``piotr`` (default, Piotr's recipe unchanged): template ``qwen3_vl`` (LLaMA-Factory ReasoningTemplate). The
    prompt ends at ``<|im_start|>assistant\n`` and the trained response is ``<think>\n\n</think>\n\n{"selection": N}``
    (empty CoT added by the template, with loss). The serving prompt is therefore the training prompt + ``<think>\n``
    -- exactly the relation between Piotr's released SelectionARM and our ``piotr5`` serving path.
  * ``serving-exact``: template ``qwen3_vl_nothink`` and response ``<think>\n{"selection": N}``, so the training token
    sequence equals the serving prompt followed by the selection tokens (only deviation from arm_lora.yaml: template).
Split: training uses folds != ``--test-fold`` (default 0, the frozen-head students' shared test fold); ``--all-folds``
trains on every fold. Held-out fold samples are written to ``<name>_test.json`` (registered, not used for training).

Outputs in --out-dir: <name>_train.json, <name>_test.json, images/<state_id>.png (copies, sha256-checked against the
panel), dataset_info.json, manifest.jsonl (one row per sample: state, fold, split, target, label, token counts),
summary.json. Samples longer than LLaMA-Factory's cutoff_len (8192) are skipped: truncation would change the prompt.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.arm_inference import load_selection_builder, selection_messages, split_response  # noqa: E402

SOURCE_ROOT = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source")
BASE = Path("/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")
IMAGE_PLACEHOLDER = "<image>"
CUTOFF_LEN = 8192                       # arm_lora.yaml cutoff_len
TRAIN_IMAGE_MAX_PIXELS = 262144         # arm_lora.yaml image_max_pixels
SERVE_MIN_PIXELS, SERVE_MAX_PIXELS = 65536, 262144     # serve_arm(.py|_sglang.py) processor limits
TIE_TOL = 1e-9


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def selector_payload(panel, screenshot):
    """The /select request CrossbenchSelector sends for this state (openwebrl/crossbench_selector_eval.py)."""
    return {"mode": "selection", "task": panel["task"], "url": panel["url"],
            "history": [split_response(r) for r in panel["history"]],
            "candidates": [split_response(c) for c in panel["candidates"]],
            "screenshot": base64.b64encode(screenshot).decode()}


def prompt_messages(builder, payload):
    """System + user messages, LLaMA-Factory style: the screenshot block becomes one ``<image>`` placeholder."""
    messages = selection_messages(builder, payload["task"], payload["url"], payload["history"], payload["candidates"],
                                  base64.b64decode(payload["screenshot"]), payload.get("candidate_representation", "full"))
    system, user = messages
    if (system["role"], user["role"]) != ("system", "user") or not isinstance(system["content"], str):
        raise ValueError("Unexpected selection message layout")
    parts = user["content"]
    if [p.get("type") for p in parts] != ["text", "image_url", "text"]:
        raise ValueError("Selection prompt must be text, screenshot, text")
    texts = [system["content"], parts[0]["text"], parts[2]["text"]]
    if any(tag in t for t in texts for tag in (IMAGE_PLACEHOLDER, "<video>", "<audio>")):
        raise ValueError("Prompt text contains a LLaMA-Factory media placeholder")
    return [{"role": "system", "content": system["content"]},
            {"role": "user", "content": parts[0]["text"] + IMAGE_PLACEHOLDER + parts[2]["text"]}]


def target_labels(target, ties="all"):
    """0-based label indices for one soft target, or (None, reason) when the state is skipped."""
    best = max(target)
    tied = [i for i, v in enumerate(target) if math.isclose(v, best, abs_tol=TIE_TOL)]
    if len(tied) == len(target):
        return None, "all_equal_target"
    if len(tied) > 1:
        if ties == "skip":
            return None, "tied_target"
        if ties == "first":
            return tied[:1], None
    return tied, None


FORMATS = {"piotr": dict(template="qwen3_vl", response_prefix="", suffix=""),
           "serving-exact": dict(template="qwen3_vl_nothink", response_prefix="<think>\n", suffix="_exact")}


def selection_text(index, fmt="piotr"):
    return FORMATS[fmt]["response_prefix"] + f'{{"selection": {index + 1}}}'


def smart_resize(height, width, factor=32, min_pixels=SERVE_MIN_PIXELS, max_pixels=16777216):
    """transformers' qwen2_vl smart_resize (Qwen3-VL: patch 16 x merge 2 = factor 32)."""
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar, w_bar = max(factor, math.floor(height / beta / factor) * factor), max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_tokens(width, height):
    """(training, inference) visual token counts: LLaMA-Factory pre-resize to image_max_pixels then the base processor
    (shortest_edge 65,536 / longest_edge 16,777,216); the selector server's processor at min 65,536 / max 262,144."""
    w, h = width, height
    if w * h > TRAIN_IMAGE_MAX_PIXELS:
        f = math.sqrt(TRAIN_IMAGE_MAX_PIXELS / (w * h))
        w, h = int(w * f), int(h * f)
    th, tw = smart_resize(h, w)
    ih, iw = smart_resize(height, width, max_pixels=SERVE_MAX_PIXELS)
    return (th // 32) * (tw // 32), (ih // 32) * (iw // 32)


def render_qwen3_vl(messages, response=None, fmt="piotr"):
    """LLaMA-Factory qwen3_vl(/_nothink) slots with one unexpanded image pad (prompt only unless a response is given)."""
    system, user = messages[0]["content"], messages[1]["content"]
    text = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
            + user.replace(IMAGE_PLACEHOLDER, "<|vision_start|><|image_pad|><|vision_end|>")
            + "<|im_end|>\n<|im_start|>assistant\n")
    if response is not None:
        empty_cot = "<think>\n\n</think>\n\n" if fmt == "piotr" else ""      # ReasoningTemplate empty CoT (loss on)
        text += empty_cot + response + "<|im_end|>\n"
    return text


def build(args):
    from PIL import Image
    builder = load_selection_builder(args.source_root)
    panels = read_jsonl(args.panels)
    targets = json.loads(Path(args.targets).read_text())
    states = targets["states"]
    if args.source not in targets.get("sources", [args.source]):
        raise ValueError(f"Source {args.source!r} not in targets file sources")
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    fmt = getattr(args, "format", "piotr")
    name = args.name or f"owrl_selection_ours_{args.source}{FORMATS[fmt]['suffix']}"
    splits = {"train": [], "test": []}
    manifest, counts, lengths = [], Counter(), {"train": [], "test": []}
    image_sizes, image_bytes, token_mismatch = Counter(), [], []
    seen = set()
    for panel in panels:
        sid = panel["state_id"]
        counts["panels"] += 1
        if sid in seen:
            raise ValueError(f"Duplicate panel state {sid}")
        seen.add(sid)
        entry = states.get(sid)
        if entry is None:
            counts["skipped_no_targets_entry"] += 1
            continue
        target = entry["targets"].get(args.source)
        if target is None:
            counts["skipped_no_source_target"] += 1
            continue
        if len(target) != len(panel["candidates"]) or not 2 <= len(panel["candidates"]) <= getattr(args, "max_candidates", 5):
            raise ValueError(f"Bad panel/target shape for {sid}")
        if entry["task"] != panel["task_id"]:
            raise ValueError(f"Task mismatch for {sid}")
        split = "train" if args.all_folds or entry["fold"] != args.test_fold else "test"
        if len({split_response(c)["action"] for c in panel["candidates"]}) < 2:
            counts[f"{split}/skipped_single_distinct_action"] += 1
            continue
        labels, reason = target_labels(target, args.ties)
        if labels is None:
            counts[f"{split}/skipped_{reason}"] += 1
            continue
        screenshot = Path(panel["image"]).read_bytes()
        shot_sha = hashlib.sha256(screenshot).hexdigest()
        if panel.get("screenshot_sha256") and shot_sha != panel["screenshot_sha256"]:
            raise ValueError(f"Screenshot changed for {sid}")
        with Image.open(panel["image"]) as im:
            width, height = im.size
        messages = prompt_messages(builder, selector_payload(panel, screenshot))
        prompt_sha = hashlib.sha256(render_qwen3_vl(messages).encode()).hexdigest()
        train_img, serve_img = image_tokens(width, height)
        n_text = len(tokenizer(render_qwen3_vl(messages, selection_text(labels[0], fmt), fmt),
                               add_special_tokens=False)["input_ids"])
        total = n_text - 1 + train_img          # one placeholder pad in the rendered text -> the visual tokens
        # serving prompt = training prompt + the chat template's "<think>\n" generation prefix (2 tokens)
        serve_prompt = len(tokenizer(render_qwen3_vl(messages), add_special_tokens=False)["input_ids"]) - 1 + serve_img + 2
        if total > args.cutoff_len:
            # LLaMA-Factory would truncate the prompt (and could cut visual tokens); Piotr's builder has no such
            # filter, but a truncated prompt is not the served prompt, so it is skipped and counted
            counts[f"{split}/skipped_over_cutoff"] += 1
            continue
        if train_img != serve_img:
            token_mismatch.append(sid)
        image_rel = f"images/{sid}.png"
        dst = out / image_rel
        if not dst.exists() or dst.read_bytes() != screenshot:
            shutil.copyfile(panel["image"], dst)
        image_sizes[f"{width}x{height}"] += 1
        image_bytes.append(len(screenshot))
        lengths[split].append(total)
        counts[f"{split}/states"] += 1
        counts[f"{split}/tied_states"] += len(labels) > 1
        for label in labels:
            sample = {"messages": messages + [{"role": "assistant", "content": selection_text(label, fmt)}],
                      "images": [image_rel]}
            splits[split].append(sample)
            manifest.append({"state_id": sid, "task_id": panel["task_id"], "fold": entry["fold"], "split": split,
                             "source": args.source, "target": target, "label": label, "tie_size": len(labels),
                             "prompt_sha256": prompt_sha, "screenshot_sha256": shot_sha, "train_tokens": total,
                             "serve_prompt_tokens": serve_prompt})
            counts[f"{split}/examples"] += 1
            counts[f"{split}/label_{label + 1}"] += 1
    for split, rows in splits.items():
        (out / f"{name}_{split}.json").write_text(json.dumps(rows, ensure_ascii=False))
    columns = {"messages": "messages", "images": "images"}
    tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
            "system_tag": "system"}
    (out / "dataset_info.json").write_text(json.dumps({
        name: {"file_name": f"{name}_train.json", "formatting": "sharegpt", "columns": columns, "tags": tags},
        f"{name}_test": {"file_name": f"{name}_test.json", "formatting": "sharegpt", "columns": columns, "tags": tags},
    }, indent=1))
    with open(out / "manifest.jsonl", "w") as stream:
        for row in manifest:
            stream.write(json.dumps(row) + "\n")

    def stats(values):
        if not values:
            return None
        s = sorted(values)
        return {"n": len(s), "min": s[0], "median": s[len(s) // 2], "p95": s[int(0.95 * (len(s) - 1))], "max": s[-1],
                "over_cutoff": sum(v > args.cutoff_len for v in s)}
    summary = {"name": name, "source": args.source, "format": fmt, "llamafactory_template": FORMATS[fmt]["template"],
               "ties": args.ties, "test_fold": None if args.all_folds else args.test_fold,
               "panels": str(Path(args.panels).resolve()), "targets": str(Path(args.targets).resolve()),
               "panels_sha256": hashlib.sha256(Path(args.panels).read_bytes()).hexdigest(),
               "targets_sha256": hashlib.sha256(Path(args.targets).read_bytes()).hexdigest(),
               "selection_prompt_source": str(Path(args.source_root) / "inference/selection_prompt.py"),
               "counts": dict(sorted(counts.items())),
               "train_tokens_per_example": stats(lengths["train"]),
               "tokens_note": "text + training visual tokens + response (incl. empty CoT for piotr format), per state",
               "test_tokens_per_example": stats(lengths["test"]), "cutoff_len": args.cutoff_len,
               "image_sizes": dict(image_sizes),
               "image_bytes": stats(image_bytes) and {k: v for k, v in stats(image_bytes).items() if k != "over_cutoff"},
               "train_vs_serve_image_token_mismatch": len(token_mismatch)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", type=Path, required=True, help="panels.jsonl (state_id, task_id, task, url, history, candidates, image)")
    ap.add_argument("--targets", type=Path, required=True, help="targets.json (states: {task, fold, targets{source: 5-way}})")
    ap.add_argument("--source", required=True, help="target source, e.g. teacher_k1 or outcome_draws_0_2")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", help="dataset name (default owrl_selection_ours_<source>)")
    ap.add_argument("--test-fold", type=int, default=0)
    ap.add_argument("--all-folds", action="store_true", help="train on every fold (no held-out test split)")
    ap.add_argument("--ties", choices=("all", "skip", "first"), default="all")
    ap.add_argument("--format", choices=tuple(FORMATS), default="piotr", help="response format / LLaMA-Factory template")
    ap.add_argument("--source-root", type=Path, default=SOURCE_ROOT, help="pinned released ARM source (selection_prompt.py)")
    ap.add_argument("--tokenizer", default=str(BASE), help="tokenizer for lengths (the OpenWebRL-4B-SFT base)")
    ap.add_argument("--cutoff-len", type=int, default=CUTOFF_LEN, help="skip samples longer than LLaMA-Factory cutoff_len")
    args = ap.parse_args()
    print(json.dumps(build(args), indent=1))


if __name__ == "__main__":
    main()
