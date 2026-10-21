#!/usr/bin/env python3
"""Luna judgments with multi-step lookahead: for each candidate, the immediate result plus the fixed SFT actor's
next K actions from execution draw 0, each with the screenshot it produced (and the actor's final answer if the
episode ended within K steps). No judge verdicts or rewards are shown. Held-out draws 3-7 are never read.

The request is the Luna after-execution request (openwebrl.arm_branch_teacher.request, draw-0 evidence, same
3 repeats + 1 shuffled layout) with the lookahead appended per candidate and one sentence added to the
instruction, so it compares directly with Luna after (draw-0 evidence).

  prepare | dry-run | register --job ID | execute --job ID

With LOOKAHEAD_COLLECTION=<campaign shards dir> at prepare, states come from that collection's released states
with draws 3-7 for every candidate, and LOOKAHEAD_CONDITIONS (default "lookahead") may add the matched "before"
(no evidence) and "after" (draw-0 immediate evidence) conditions on the same states, 4 calls each.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_branch_sol_labels import HIST, LUNA, REPO, RUNTIME, Ledger, read, scoreable, write  # noqa: E402

ROOT = RUNTIME / "selection-head-20261008/luna-lookahead-labels-20261011"
MODEL, EFFORT, DRAW = "gpt-6-luna", "high", 0
K = int(os.environ.get("LOOKAHEAD_K", "3"))                          # lookahead depth; frozen in the plan
HISTORY_IMAGES = int(os.environ.get("LOOKAHEAD_HISTORY_IMAGES", "0"))  # screenshots before the last N past actions
# history format "steps": the last HISTORY_STEPS past steps as abbreviated actions, each with the screenshot before it,
# replacing the default text history (full past outputs with reasoning); frozen in the plan
HISTORY_FORMAT = os.environ.get("LOOKAHEAD_HISTORY_FORMAT", "text")
HISTORY_STEPS = int(os.environ.get("LOOKAHEAD_HISTORY_STEPS", "3"))
ACTION_CHARS = 1500                     # cap on each abbreviated action (history and lookahead steps)
# lookahead steps include the environment's reply (tool response, observation text) unless LOOKAHEAD_FEEDBACK=0,
# which keeps only the actor's own message (the tool call), as in the history steps; frozen in the plan
FEEDBACK = os.environ.get("LOOKAHEAD_FEEDBACK", "1") != "0"
# LOOKAHEAD_IMAGES=current keeps only the original state screenshot; past steps, immediate results and lookahead
# steps stay as text (actions, receipts, tool feedback). Default "all". Frozen in the plan.
IMAGES = os.environ.get("LOOKAHEAD_IMAGES", "all")          # "all" | "current" | "none" (no screenshots at all)
# LOOKAHEAD_STATE_TEXT=none drops page text (visible text, interactive elements) from the current and immediate states
# (keeping url, title, tabs) and the observation text from lookahead replies; actions and tool results stay. Frozen.
STATE_TEXT = os.environ.get("LOOKAHEAD_STATE_TEXT", "full")
# LOOKAHEAD_ORDERS: "fixed" (default: 3 calls in canonical order + 1 seeded shuffle) or "random:N" (N calls per state,
# each in its own seeded random candidate order, so no call favours the actor's first sample by position). Frozen.
ORDERS = os.environ.get("LOOKAHEAD_ORDERS", "fixed")


def call_orders(state, canonical, permuted):
    if ORDERS == "fixed":
        return [(t, canonical) for t in range(3)] + [("permuted", permuted)]
    from openwebrl.arm_branch_protocol import seed
    n = int(ORDERS.split(":", 1)[1])
    out = []
    for r in range(n):
        order = canonical.copy()
        random.Random(seed(state, f"random-order-{r}")).shuffle(order)
        out.append((f"random{r}", order))
    return out


def calls_per_state():
    return 4 if ORDERS == "fixed" else int(ORDERS.split(":", 1)[1])


def abbreviate(text):
    """An actor output reduced to its action: drop everything up to the last </think>, special tokens, and cap it."""
    text = text[0] if isinstance(text, list) else text
    text = text.rsplit("</think>", 1)[-1] if "</think>" in text else re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("<|im_end|>", "").replace("<think>", "").strip()[:ACTION_CHARS]
HIDE_ANSWER = os.environ.get("LOOKAHEAD_HIDE_ANSWER") == "1"   # recorded in the plan; the plan decides later
CONDITIONS = os.environ.get("LOOKAHEAD_CONDITIONS", "lookahead").split(",")   # likewise frozen in the plan
TRANSIENT = {429, 500, 502, 503, 504}
PRICE = {"input_cache_write": 0.125, "output": 0.5}
LOOKAHEAD_NOTE = (" For each candidate you also see the next steps the same fixed actor took after executing it in one "
                  "execution, each with the resulting screenshot, and its final answer if it stopped. These are one sample of "
                  "the actor's behaviour, not a success judgment.")
DATA_URL = re.compile(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+")


def collection_states(shards):
    """Released states of a branch collection with draws 3-7 recorded for all five candidates."""
    import sqlite3
    from collections import defaultdict
    out = []
    informative_only = os.environ.get("LOOKAHEAD_REQUIRE_INFORMATIVE") == "1"   # keep states whose candidates differ
    roots = sorted(Path(shards).glob("shard*")) or sorted(Path(shards).glob("jobs/*/seg*"))   # parallel10 or fast1k layout
    for root in roots:
        cell = defaultdict(set); wins = defaultdict(lambda: defaultdict(int))
        db = sqlite3.connect(f"file:{root}/state.sqlite3?mode=ro", uri=True)
        for (raw,) in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
            r = json.loads(raw)
            if r.get("mode") == "branch" and r.get("released"):
                cell[r["state"]].add((r["candidate"], r["repeat"]))
                wins[r["state"]][r["candidate"]] += bool(r["valid"]) and (r.get("reward") or 0) >= 1
        accepted = (read(root / "accepted-states.json")["state_ids"] if (root / "accepted-states.json").exists()
                    else sorted(s for s in cell if (root / "states" / s / "anchor-reference.json").exists()))
        for sid in accepted:
            if not ((root / "states" / sid / "release.json").exists() and all((i, k) in cell[sid] for i in range(5) for k in range(8))):
                continue
            if informative_only and len({wins[sid][i] for i in range(5)}) == 1:
                continue                               # every candidate succeeds equally often: no choice to evaluate
            out.append(dict(state=sid, collection=Path(shards).name if (Path(shards) / "jobs").exists() else Path(shards).parent.name,
                            root=str(root)))
    return out


def states():
    if os.environ.get("LOOKAHEAD_STATES_FROM"):     # reuse the exact state list of earlier plans (labelled states only)
        out = []
        for plan_path in os.environ["LOOKAHEAD_STATES_FROM"].split(","):
            plan = read(plan_path); done = Path(plan_path).parent / "judgments"
            out += [e for e in plan["states"] if (done / f"{e['state']}.json").exists() and read(done / f"{e['state']}.json")]
        return out
    if os.environ.get("LOOKAHEAD_STATE_LIST"):      # explicit [{state, collection, root}] across campaign ledgers
        return read(os.environ["LOOKAHEAD_STATE_LIST"])
    if os.environ.get("LOOKAHEAD_COLLECTION"):
        return collection_states(os.environ["LOOKAHEAD_COLLECTION"])
    keep = scoreable()
    out = []
    for item in read(LUNA / "plan.json")["schedule"]:
        out.append(dict(state=item["state"], collection=item["collection"], root=str(LUNA)))
    for sid in read(HIST / "accepted-states.json")["state_ids"]:
        out.append(dict(state=sid, collection="historical", root=str(HIST)))
    return [e for e in out if e["state"] in keep]


def draw0(entry):
    """Draw-0 attempt directories by candidate (from the evidence release, which lists draws 0-2 only)."""
    rel = read(Path(entry["root"]) / "states" / entry["state"] / "release.json")
    out = {}
    for d in rel["attempt_directories"]:
        f = Path(d) / "immediate.json"
        if f.exists():
            x = read(f)
            if x["repeat"] == DRAW:
                out[x["candidate"]] = (Path(d), x)
    return out


def lookahead(attempt, depth, candidate_text):
    """Next K actor turns after the candidate, with the screenshot after each; verifies the turn alignment."""
    files = list((attempt / "rollouts").rglob("*.json"))
    if len(files) != 1:
        return None
    r = read(files[0])
    segments = r["llm_input_texts"].split("<|im_start|>assistant\n")[1:]
    # the input ends with the final turn's generation prefix (e.g. "<think>\n"), which llm_response continues
    turns = segments[:-1] + [segments[-1] + r["llm_response"]] if segments else [r["llm_response"]]
    images = [m.group(0) if (m := DATA_URL.search(str(x))) else None for x in r["images"]]
    if len(turns) <= depth or candidate_text.strip()[:200] not in turns[depth]:
        return None          # alignment check: turn `depth` must be the executed candidate
    steps = []
    for j in range(1, K + 1):
        t = depth + j
        if t >= len(turns):
            break
        text = turns[t] if FEEDBACK else turns[t].split("<|im_end|>", 1)[0]
        if STATE_TEXT == "none":                     # keep the tool result line, drop the page observation text
            text = re.sub(r"<observation>.*?(?:</observation>|$)", "", text, flags=re.S)
        action = abbreviate(text)
        shot = images[t + 1] if t + 1 < len(images) else None
        steps.append(dict(action=action, screenshot=shot, final=(t == len(turns) - 1)))
    return dict(steps=steps, ended=(depth + K >= len(turns) - 1), status=r.get("terminate_reason"))


def build(src, entry):
    sys.path.insert(0, str(src))
    from openwebrl.arm_branch_protocol import seed
    from openwebrl.arm_branch_teacher import request
    ref = read(Path(entry["root"]) / "states" / entry["state"] / "anchor-reference.json")
    anchor = read(ref["path"])
    ev = draw0(entry)
    if sorted(ev) != list(range(len(anchor["candidates"]))):
        return []
    shown = entry.get("candidates")
    if shown:
        # a per-state subset of the executed candidates, optionally each shown with another text of the same action
        # (`texts`); execution evidence and lookahead stay the executed candidate's (alignment is checked on its text)
        looks = [lookahead(ev[c][0], anchor["depth"], anchor["candidates"][c]["output"][0]) for c in shown]
        texts = entry.get("texts") or [anchor["candidates"][c]["output"][0] for c in shown]
        anchor = dict(anchor, candidates=[dict(anchor["candidates"][c], output=[t] + list(anchor["candidates"][c]["output"][1:]))
                                          for c, t in zip(shown, texts)])
        ev = {j: ev[c] for j, c in enumerate(shown)}
    canonical = list(range(len(anchor["candidates"])))
    if not shown:
        looks = [lookahead(ev[c][0], anchor["depth"], anchor["candidates"][c]["output"][0]) for c in canonical]
    if any(x is None for x in looks):
        return []
    rows = [ev[c][1] for c in canonical]
    permuted = canonical.copy()
    random.Random(seed(entry["state"], "permutation")).shuffle(permuted)
    if permuted == canonical:
        permuted = canonical[1:] + canonical[:1]
    out = []
    for condition in CONDITIONS:
        for teacher_repeat, order in call_orders(entry["state"], canonical, permuted):
            p = request(anchor, None if condition == "before" else rows, "index", order)
            if condition == "lookahead":
                add_lookahead(p, order, looks)
            if HISTORY_FORMAT == "steps":
                add_history_steps(p, anchor, HISTORY_STEPS)
            elif HISTORY_IMAGES:
                add_history(p, anchor, HISTORY_IMAGES)
            if IMAGES in ("current", "none"):
                keep_current_screenshot(p, keep_current=IMAGES == "current")
            if STATE_TEXT == "none":
                drop_state_text(p)
            assert p["model"] == MODEL and p["reasoning"] == {"effort": EFFORT}
            out.append((condition, teacher_repeat, order, p))
    return out


def drop_state_text(p):
    """Keep only url / title / tabs of the current and immediate page observations; states are judged from screenshots."""
    content = p["input"][0]["content"]
    view = json.loads(content[0]["text"])
    slim = lambda o: {k: o[k] for k in ("url", "title", "tabs") if isinstance(o, dict) and k in o}
    view["before_observation"] = slim(view.get("before_observation"))
    for row in view.get("immediate_results", []):
        row["observation"] = slim(row.get("observation"))
    content[0]["text"] = json.dumps(view, ensure_ascii=False)
    p["instructions"] = p["instructions"].replace("Return only", "Page text is not shown; judge each state from its screenshot. Return only", 1)


def keep_current_screenshot(p, keep_current=True):
    """Drop every image except the original state screenshot (or every image), with the caption naming each dropped one."""
    content = p["input"][0]["content"]
    kept = []
    for k, c in enumerate(content):
        nxt = content[k + 1] if k + 1 < len(content) else None
        if c.get("type") == "input_image":
            if keep_current and kept and kept[-1].get("text") == "Original state screenshot":
                kept.append(c)
        elif nxt is not None and nxt.get("type") == "input_image" and (c.get("text") != "Original state screenshot" or not keep_current):
            continue                                 # caption of an image that is dropped
        else:
            kept.append(c)
    content[:] = kept
    ins = p["instructions"]
    for a, b in ((", each with the resulting screenshot,", ","), (", each as the action taken and the screenshot before it", ", each as the action taken"),
                 ("use only what the supplied screenshots and receipts establish", "use only what the supplied receipts establish")):
        ins = ins.replace(a, b)
    note = ("Only the original state screenshot is shown; everything else is text. " if keep_current
            else "No screenshots are shown; every state is given as text. ")
    p["instructions"] = ins.replace("Return only", note + "Return only", 1)


def add_lookahead(p, order, looks):
    """Append each candidate's next K actor steps (and screenshots) to an after-execution request."""
    content = p["input"][0]["content"]
    for j, i in enumerate(order):
        content.append(dict(type="input_text", text=f"Next steps of the same fixed actor after candidate {j + 1}"
                            + (" (the episode ended within these steps)" if looks[i]["ended"] else "")))
        for k, step in enumerate(looks[i]["steps"], 1):
            shown = ("[the episode ended here; any final answer is not shown]"
                     if HIDE_ANSWER and step["final"] and looks[i]["ended"] else step["action"])
            content.append(dict(type="input_text", text=f"Candidate {j + 1}, actor step {k}: {shown}"))
            if step["screenshot"] and not step["final"]:
                content.append(dict(type="input_text", text=f"Candidate {j + 1}, screenshot after actor step {k}"))
                content.append(dict(type="input_image", image_url=step["screenshot"]))
    note = LOOKAHEAD_NOTE.replace(", and its final answer if it stopped", ", and whether it stopped (its final answer is not shown)") if HIDE_ANSWER else LOOKAHEAD_NOTE
    p["instructions"] = p["instructions"].replace("Return only", note.strip() + " Return only", 1)


def add_history_steps(p, anchor, n):
    """Replace the text history (full past outputs) with the last n past steps: abbreviated action + the screenshot
    before it. Candidates keep their full text, so reasoning appears only for the actions being judged."""
    import base64
    from openwebrl.arm_branch_worker import load_image
    content = p["input"][0]["content"]
    view = json.loads(content[0]["text"])
    view["history"] = []
    content[0]["text"] = json.dumps(view, ensure_ascii=False)
    past = anchor.get("prefix", [])[-n:] if n > 0 else []          # [-0:] would be the whole prefix
    at = next(i for i, c in enumerate(content) if c.get("type") == "input_text" and c.get("text") == "Original state screenshot")
    extra = []
    for k, step in enumerate(past):
        ago = len(past) - k
        extra += [dict(type="input_text", text=f"Past step {ago} back: screenshot before the action"),
                  dict(type="input_image", image_url="data:image/png;base64," + base64.b64encode(load_image(step["before_observation"])).decode()),
                  dict(type="input_text", text=f"Past step {ago} back: action: {abbreviate(step['output'])}")]
    content[at:at] = extra
    note = (f"You see the last {len(past)} past steps, each as the action taken and the screenshot before it. " if past
            else "Earlier steps of the task are not shown. " if n == 0 else "This is the first step of the task. ")
    p["instructions"] = p["instructions"].replace("Return only", note + "Return only", 1)


def add_history(p, anchor, n):
    """Screenshots before the last n past actions, inserted ahead of the current screenshot."""
    import base64
    from openwebrl.arm_branch_worker import load_image
    past = anchor.get("prefix", [])[-n:]
    if not past:
        return
    content = p["input"][0]["content"]
    at = next(i for i, c in enumerate(content) if c.get("type") == "input_text" and c.get("text") == "Original state screenshot")
    extra = []
    for k, step in enumerate(past):
        ago = len(past) - k
        extra += [dict(type="input_text", text=f"Screenshot before the past action {ago} step{'s' if ago > 1 else ''} ago"),
                  dict(type="input_image", image_url="data:image/png;base64," + base64.b64encode(load_image(step["before_observation"])).decode())]
    content[at:at] = extra
    p["instructions"] = p["instructions"].replace("Return only", f"You also see the screenshots before the last {len(past)} past actions. Return only", 1)


def bound_usd(p):
    from openwebrl.controlled_sft_state import request_bound
    text, outcap = request_bound(p)
    ip, op = PRICE["input_cache_write"], PRICE["output"]
    if text > 272000:
        ip, op = ip * 2, op * 1.5
    return (text * ip + outcap * op) / 1e6


def parse(raw, n):
    if raw.get("model", "") != MODEL or raw.get("status") != "completed":
        raise ValueError("wrong model or incomplete response")
    texts = [c["text"] for item in raw.get("output", []) if item.get("type") == "message"
             for c in item.get("content", []) if c.get("type") == "output_text"]
    value = json.loads(texts[0]) if len(texts) == 1 else None
    if not value or set(value) != {"selection"} or not 1 <= value["selection"] <= n:
        raise ValueError("invalid selection")
    return value


def cost(usage):
    it, ot = usage.get("input_tokens"), usage.get("output_tokens")
    ip, op = PRICE["input_cache_write"], PRICE["output"]
    if it > 272000:
        ip, op = ip * 2, op * 1.5
    return (it * ip + ot * op) / 1e6


async def label_state(entry, src, ledger, client, sem, key, outdir):
    out = outdir / f"{entry['state']}.json"
    if out.exists():
        return
    built = build(src, entry)
    judgments = []
    for condition, teacher_repeat, order, p in built:
        label = dict(condition={"lookahead": "after_lookahead"}.get(condition, condition), execution_repeat=DRAW, lookahead_steps=K, history_images=HISTORY_IMAGES, history_format=HISTORY_FORMAT, teacher_repeat=teacher_repeat,
                     order=order, state=entry["state"], form="index", model=MODEL, effort=EFFORT)
        reserved = bound_usd(p)
        async with sem:
            for attempt in range(4):        # transient provider errors: up to 3 retries, each separately reserved
                ledger.reserve(reserved)
                try:
                    r = await client.post("https://api.openai.com/v1/responses", json=p, headers={"Authorization": "Bearer " + key})
                    raw = r.json() if r.status_code not in TRANSIENT else {}
                    transient = r.status_code in TRANSIENT
                except (Exception,) as exc:
                    transient, raw, r = True, {}, None
                    reason = type(exc).__name__
                if transient and not (raw.get("usage") or {}):
                    keep_reserved(ledger, reserved)          # charged at the reservation; no halt
                    if attempt == 3:
                        ledger.settle(0.0, None)
                        raise RuntimeError(f"transient provider failure persisted ({r.status_code if r is not None else reason})")
                    await asyncio.sleep(5 * 2 ** attempt)
                    continue
                break
            usage = raw.get("usage") or {}
            # settle with the long-context-aware cost: pass usage through a shim the ledger understands
            if "input_tokens" in usage:
                it, ot = usage["input_tokens"], usage["output_tokens"]
                long = it > 272000          # long-context pricing: 2x input, 1.5x output
                billed = {"input_tokens": it * (2 if long else 1), "output_tokens": int(round(ot * (1.5 if long else 1)))}
            else:
                billed = None
            charged = ledger.settle(reserved, billed)
            if charged is None:
                raise RuntimeError("missing usage; halted")
            (outdir / "raw").mkdir(exist_ok=True)
            with open(outdir / "raw" / f"{entry['state']}.jsonl", "a") as f:
                f.write(json.dumps(dict(label=label, status=r.status_code, response=raw)) + "\n")
            try:
                r.raise_for_status(); value = parse(raw, len(order))
                position = order[value["selection"] - 1]       # index into the shown candidates
                shown = entry.get("candidates")
                judgments.append(dict(label, valid=True, candidate=shown[position] if shown else position,
                                      shown_position=position, output=value, usage=usage))
            except Exception as exc:
                judgments.append(dict(label, valid=False, reason=type(exc).__name__))
    write(out, judgments)


def keep_reserved(ledger, reserved):
    def fn(s):
        s["reserved_usd"] -= reserved; s["charged_usd"] += reserved; s["transient_retries"] = s.get("transient_retries", 0) + 1
    ledger._update(fn)


def main():
    import run_branch_sol_labels as sol
    sol.PRICE = PRICE      # the shared ledger settles at Luna's conservative rates
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "dry-run", "register", "execute"])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--job")
    args = ap.parse_args()
    root = args.root.resolve()
    global HIDE_ANSWER, CONDITIONS, K, HISTORY_IMAGES, HISTORY_FORMAT, HISTORY_STEPS, FEEDBACK, IMAGES, ORDERS, STATE_TEXT
    if args.command != "prepare":   # the frozen plan decides answers shown, conditions, depth and history images
        frozen = read(root / "plan.json")
        HIDE_ANSWER = bool(frozen.get("hide_final_answer", False))
        CONDITIONS = frozen.get("conditions", ["lookahead"])
        K = frozen.get("lookahead_steps", 3); HISTORY_IMAGES = frozen.get("history_images", 0)
        HISTORY_FORMAT = frozen.get("history_format", "text"); HISTORY_STEPS = frozen.get("history_steps", 3)
        FEEDBACK = frozen.get("lookahead_feedback", True); IMAGES = frozen.get("images", "all")
        ORDERS = frozen.get("orders", "fixed"); STATE_TEXT = frozen.get("state_text", "full")
    if args.command == "prepare":
        if (root / "plan.json").exists():
            raise SystemExit("already prepared; plan is frozen")
        src = root / "source"; root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--exclude", "docs", "--exclude", "__pycache__",
                        str(REPO / "openwebrl"), str(REPO / "scripts"), str(src) + "/"], check=True)
        sts = states()
        write(root / "plan.json", dict(root=str(root), model=MODEL, effort=EFFORT, lookahead_steps=K, evidence_draw=DRAW,
              price_usd_per_million=PRICE, calls_per_state=calls_per_state() * len(CONDITIONS), orders=ORDERS, states=sts, source=str(src),
              hide_final_answer=HIDE_ANSWER, conditions=CONDITIONS, collection=os.environ.get("LOOKAHEAD_COLLECTION"),
              history_images=HISTORY_IMAGES, states_from=os.environ.get("LOOKAHEAD_STATES_FROM"), state_list=os.environ.get("LOOKAHEAD_STATE_LIST"),
              history_format=HISTORY_FORMAT, history_steps=HISTORY_STEPS, action_chars=ACTION_CHARS, lookahead_feedback=FEEDBACK, images=IMAGES, state_text=STATE_TEXT,
              note="Luna after-execution request with K-step actor lookahead from draw 0; scoring draws 3-7 never read"))
        print(json.dumps(dict(states=len(sts))))
    elif args.command == "dry-run":
        plan = read(root / "plan.json"); src = Path(plan["source"]); sys.path.insert(0, str(src))
        total, n, imgs, empty, tokens_est = 0.0, 0, 0, 0, []
        for e in plan["states"]:
            b = build(src, e)
            empty += not b
            for _, _, _, p in b:
                total += bound_usd(p); n += 1
                imgs = max(imgs, json.dumps(p).count("data:image/"))
        rep = dict(states=len(plan["states"]), states_without_evidence_or_alignment=empty, calls=n,
                   max_images_per_call=imgs, worst_case_usd=round(total, 2))
        write(root / "dry-run.json", rep); print(json.dumps(rep))
    elif args.command == "register":
        a = read(root / "approval.json"); a.setdefault("attempts", []).append({"job_id": str(args.job), "registered_unix": time.time()})
        write(root / "approval.json", a)
    else:
        from hashlib import sha256
        plan, approval = read(root / "plan.json"), read(root / "approval.json")
        job = str(args.job)
        if job != os.getenv("SLURM_JOB_ID") or not any(a["job_id"] == job for a in approval.get("attempts", [])):
            raise SystemExit("must run inside a registered, approved Slurm job")
        if approval.get("approved") is not True or approval.get("plan_sha256") != sha256((root / "plan.json").read_bytes()).hexdigest():
            raise SystemExit("approval does not match the frozen plan")
        from dotenv import dotenv_values
        key = os.environ.get("OPENAI_API_KEY") or dotenv_values(REPO / ".env").get("OPENAI_API_KEY")
        sys.path.insert(0, plan["source"])
        ledger = Ledger(root, approval["caps"])
        outdir = root / "judgments"; outdir.mkdir(exist_ok=True)

        async def run():
            import httpx
            sem = asyncio.Semaphore(int(os.environ.get("LOOKAHEAD_CONCURRENCY", "16")))   # lower it to fit worst-case reservations under a cap
            async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
                pending = [e for e in plan["states"] if not (outdir / f"{e['state']}.json").exists()]
                for i in range(0, len(pending), 16):
                    res = await asyncio.gather(*(label_state(e, Path(plan["source"]), ledger, client, sem, key, outdir)
                                                 for e in pending[i:i + 16]), return_exceptions=True)
                    errors = [x for x in res if isinstance(x, Exception)]
                    write(root / "current.json", dict(job_id=job, done=len(list(outdir.glob("*.json"))), states=len(plan["states"]),
                                                      ledger=read(root / "ledger.json"), errors=[repr(x) for x in errors], updated_unix=time.time()))
                    if errors:
                        raise SystemExit("stopping on error: " + repr(errors[0]))
        asyncio.run(run())
        write(root / "closeout.json", dict(completed=True, ledger=read(root / "ledger.json")))


if __name__ == "__main__":
    main()
