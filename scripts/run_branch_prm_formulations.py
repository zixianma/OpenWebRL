#!/usr/bin/env python3
"""Formulation-faithful WebShepherd, WebArbiter and PRO-CUA judges on branch states, with GPT-6 Luna as the model.

Each method keeps its own prompt and scoring rule; only the model is Luna (the same teacher as the lookahead labels):
- WebShepherd: upstream checklist prompt, then the upstream progress-judging prompt per candidate (text observation,
  last 10 steps, checklist); score = mean over checklist items of Yes 1 / In Progress 0.5 / No 0 (upstream weights
  the same labels by token probabilities, which Luna does not return); pick the best score.
- WebArbiter: the official WebPRMBench pairwise template (rendered by `load_arbiter_causal_renderer`), the native
  verdict parser and the viewer's knockout tournament over the five candidates.
- WebArbiter + lookahead K = 1 (`arbiter_k1`): the same pairwise template, tournament and parser, with each response's
  execution evidence appended (the recommended lookahead labeller's K = 1 view: the screenshot right after the action
  on draw 0, the same fixed actor's next action and the screenshot after it; any final answer hidden).
- PRO-CUA: the paper's PRM grading prompt (appendix, verbatim) with the current screenshot annotated at the action's
  target; one call per candidate, binary correct/incorrect; pick among the correct ones.
Adaptations forced by the states: the page text is the saved accessibility tree rendered as `[id] tag "text"`
lines; coordinate actions are mapped to BrowserGym-style actions on the element under the click point for
WebShepherd and WebArbiter (PRO-CUA keeps the native coordinate action). Ties are kept and scored as an even split.

Usage: run_branch_prm_formulations.py prepare --root R --states-from PLAN[,PLAN] --methods shepherd,arbiter,procua
       run_branch_prm_formulations.py dry-run --root R      (also writes sample requests, no calls)
       run_branch_prm_formulations.py execute --root R
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_branch_sol_labels import REPO, Ledger, read, write  # noqa: E402

MODEL, EFFORT, MAX_OUT = "gpt-6-luna", "high", 8192
PRICE = {"input_cache_write": 0.125, "output": 0.5}
TRANSIENT = {429, 500, 502, 503, 504}
WEBPRM = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009")
PROCUA_PROMPT = Path(__file__).resolve().parent / "prompts" / "procua_prm_grading.txt"
TOOL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


# ---- state rendering ---------------------------------------------------------------------------------------------
def load_image(obs):
    from openwebrl.arm_branch_worker import load_image as li
    return li(obs)


def axtree(obs):
    """The saved accessibility tree (or interactive elements) as BrowserGym-like `[id] tag "text"` lines."""
    rows = obs.get("a11ytree") or (obs.get("selection_page") or {}).get("interactive_elements") or []
    lines = []
    for r in rows:
        text = (r.get("innerText") or r.get("textContent") or "").strip().replace("\n", " ")[:120]
        lines.append(f"[{r.get('id')}] {r.get('tag') or 'element'}" + (f' "{text}"' if text else ""))
    return "\n".join(lines) or "(no elements)"


def split_output(output):
    """(thought, list of tool-call dicts) from an actor output."""
    text = output[0] if isinstance(output, list) else output
    thought = text.split("</think>")[0].replace("<think>", "").strip() if "</think>" in text else TOOL.split(text)[0].strip()
    calls = []
    for m in TOOL.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except ValueError:
            pass
    return thought, calls


def native_action(output):
    calls = [m.group(0) for m in TOOL.finditer(output[0] if isinstance(output, list) else output)]
    return "\n".join(c.strip() for c in calls) or "(no action)"


def element_at(obs, point):
    """Element whose box contains a 0-1000 normalized point (smallest box wins)."""
    x, y = point[0] / 1000, point[1] / 1000
    rows = obs.get("a11ytree") or (obs.get("selection_page") or {}).get("interactive_elements") or []
    hits = [r for r in rows if r.get("bbox") and r["bbox"][0] <= x <= r["bbox"][2] and r["bbox"][1] <= y <= r["bbox"][3]]
    return min(hits, key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])) if hits else None


def browsergym_action(calls, obs, screen=(1280, 720)):
    """Coordinate tool calls as BrowserGym-style actions (element id when a click point falls on an element)."""
    out = []
    for c in calls:
        name, a = c.get("name"), c.get("arguments") or {}
        if name in ("click", "double_click", "right_click") and a.get("point_2d"):
            el = element_at(obs, a["point_2d"])
            fn = {"click": "click", "double_click": "dblclick", "right_click": "click"}[name]
            out.append(f"{fn}('{el['id']}')" if el else f"mouse_click({a['point_2d'][0] * screen[0] / 1000:.0f}, {a['point_2d'][1] * screen[1] / 1000:.0f})")
        elif name in ("write", "type"):
            out.append(f"keyboard_type({json.dumps(a.get('message') or a.get('text') or '')})")
        elif name == "press_keys":
            out.append(f"keyboard_press({json.dumps('+'.join(a.get('keys') or []))})")
        elif name == "scroll":
            dy = (a.get("amount") or 0.5) * screen[1] * (1 if a.get("direction", "down") == "down" else -1)
            out.append(f"scroll(0, {dy:.0f})")
        elif name == "goto_url":
            out.append(f"goto({json.dumps(a.get('url', ''))})")
        elif name == "go_back":
            out.append("go_back()")
        elif name == "new_tab":
            out.append("new_tab()")
        elif name == "switch_tab":
            out.append(f"tab_focus({a.get('tab_index', 0)})")
        elif name == "done":
            out.append(f"send_msg_to_user({json.dumps(a.get('response') or '')})")
        else:
            out.append(json.dumps(c))
    return "\n".join(out) or "noop()"


def annotate(png, calls):
    """PRO-CUA input: the current screenshot with the action's target marked (red ring and point, index label)."""
    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(png)).convert("RGB"); d = ImageDraw.Draw(im)
    W, H = im.size
    for k, c in enumerate(calls, 1):
        p = (c.get("arguments") or {}).get("point_2d")
        if p:
            x, y = p[0] * W / 1000, p[1] * H / 1000
            d.ellipse([x - 14, y - 14, x + 14, y + 14], outline=(255, 0, 0), width=4)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 0, 0))
            d.text((x + 16, y - 18), str(k), fill=(255, 0, 0))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def state_inputs(entry):
    anchor = read(read(Path(entry["root"]) / "states" / entry["state"] / "anchor-reference.json")["path"])
    obs = anchor["before_observation"]
    prefix = anchor.get("prefix", [])
    history = [split_output(step["output"]) for step in prefix]
    return dict(
        anchor=anchor, obs=obs, intent=anchor["task"].strip() if isinstance(anchor["task"], str) else json.dumps(anchor["task"]),
        start_url=(prefix[0].get("before_observation") or {}).get("active_tab_url") if prefix else obs.get("active_tab_url", ""),
        current_url=obs.get("active_tab_url", ""), text_observation=axtree(obs),
        trajectory_bg="\n".join(f"Thought {i + 1}: {t}\nAction {i + 1}: {browsergym_action(c, (prefix[i].get('before_observation') or {}))}"
                                for i, (t, c) in enumerate(history[-10:])),
        history_native="\n".join(f"Step {i + 1}: {native_action(step['output'])}" for i, step in enumerate(prefix)) or "(none)",
        candidates=[dict(index=i, thought=split_output(c["output"])[0], calls=split_output(c["output"])[1],
                         native=native_action(c["output"])) for i, c in enumerate(anchor["candidates"])])


# ---- method requests ---------------------------------------------------------------------------------------------
def to_responses(messages, images=()):
    system = "\n".join(m["content"] for m in messages if m["role"] == "system" and isinstance(m["content"], str))
    content = []
    for m in messages:
        if m["role"] != "user":
            continue
        parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for c in parts:
            if c.get("type") == "text":
                content.append({"type": "input_text", "text": c["text"]})
            elif c.get("type") == "image_url":
                content.append({"type": "input_image", "image_url": c["image_url"]["url"], "detail": "high"})
    content += [{"type": "input_image", "image_url": u, "detail": "high"} for u in images]
    return dict(model=MODEL, reasoning={"effort": EFFORT}, max_output_tokens=MAX_OUT, store=False,
                instructions=system or None, input=[{"role": "user", "content": content}])


class Methods:
    def __init__(self):
        from openwebrl.webprm_source import load_arbiter, load_arbiter_causal_renderer, load_shepherd
        self.shepherd = load_shepherd(WEBPRM / "sources" / "shepherd")
        self.arbiter = load_arbiter(WEBPRM / "sources" / "arbiter")
        self.renderer = load_arbiter_causal_renderer(WEBPRM / "WebPRMBench.jsonl")
        self.procua = PROCUA_PROMPT.read_text()

    def shepherd_checklist(self, s):
        msg = self.shepherd.get_messages(dict(intent=s["intent"], start_url=s["start_url"], text_observation=s["text_observation"],
                                              current_url=s["current_url"], trajectory=s["trajectory_bg"]), "checklist_generation", "web_shepherd")
        return to_responses(msg)

    def shepherd_judge(self, s, checklist, cand):
        ctx = dict(intent=s["intent"], start_url=s["start_url"], text_observation=s["text_observation"], current_url=s["current_url"],
                   trajectory=s["trajectory_bg"], checklist=checklist, thought=cand["thought"], action=browsergym_action(cand["calls"], s["obs"]))
        return to_responses(self.shepherd.get_messages(ctx, "judge_progress", "web_shepherd", text_obs=True, image_obs=False))

    def arbiter_compare(self, s, left, right):
        prompt = self.renderer.render(intent=s["intent"], text_observation=s["text_observation"], trajectory=s["trajectory_bg"],
                                      start_url=s["start_url"], current_url=s["current_url"],
                                      thought1=left["thought"], action1=browsergym_action(left["calls"], s["obs"]),
                                      thought2=right["thought"], action2=browsergym_action(right["calls"], s["obs"]))
        return to_responses([dict(role="user", content=prompt)])

    def arbiter_compare_k1(self, s, left, right, ev):
        p = self.arbiter_compare(s, left, right)
        content = p["input"][0]["content"]
        content.append({"type": "input_text", "text": "Execution evidence: the same fixed browser agent executed each response "
                        "once. For each response: the screenshot right after its action, the agent's next action, and the "
                        "screenshot after that next action."})
        for j, cand in ((1, left), (2, right)):
            e = ev[cand["index"]]
            content += [{"type": "input_text", "text": f"Response {j}: screenshot right after executing its action"},
                        {"type": "input_image", "image_url": e["after"], "detail": "high"}]
            for step in e["steps"]:
                shown = "[the episode ended here; any final answer is not shown]" if step["final"] and e["ended"] else step["action"]
                content.append({"type": "input_text", "text": f"Response {j}: the agent's next action: {shown}"})
                if step["screenshot"] and not step["final"]:
                    content += [{"type": "input_text", "text": f"Response {j}: screenshot after that next action"},
                                {"type": "input_image", "image_url": step["screenshot"], "detail": "high"}]
        content.append({"type": "input_text", "text": "Use this execution evidence when comparing the two responses, and answer "
                        "in the required format."})
        return p

    def procua_grade(self, s, cand, step_index):
        text = (self.procua.replace("{instruction}", s["intent"]).replace("{history_actions}", s["history_native"])
                .replace("{step_index}", str(step_index)).replace("{action_code}", f"Thought: {cand['thought']}\nAction: {cand['native']}"))
        return to_responses([dict(role="user", content=text)], images=[annotate(load_image(s["obs"]), cand["calls"])])


def lookahead_evidence(entry, anchor):
    """Per candidate: draw-0 screenshot right after the action and the next K = 1 actor step (lookahead labeller code)."""
    import base64
    for k, v in (("LOOKAHEAD_K", "1"), ("LOOKAHEAD_STATE_TEXT", "none"), ("LOOKAHEAD_HIDE_ANSWER", "1")):
        os.environ.setdefault(k, v)
    import run_branch_lookahead_labels as la
    assert la.K == 1, "arbiter_k1 needs LOOKAHEAD_K=1"
    ev = la.draw0(entry)
    if sorted(ev) != list(range(len(anchor["candidates"]))):
        return None
    out = {}
    for c, (attempt, row) in ev.items():
        look = la.lookahead(attempt, anchor["depth"], anchor["candidates"][c]["output"][0])
        if look is None:
            return None
        out[c] = dict(after="data:image/png;base64," + base64.b64encode(load_image(row["observation"])).decode(),
                      steps=look["steps"], ended=look["ended"])
    return out


# ---- parsing -----------------------------------------------------------------------------------------------------
def output_text(raw):
    texts = [c["text"] for o in raw.get("output", []) if o.get("type") == "message" for c in o.get("content", []) if c.get("type") == "output_text"]
    return texts[0] if len(texts) == 1 else ""


def parse_checklist(text):
    return text.split("<answer>")[-1].split("</answer>")[0].strip() if "<answer>" in text else text.split("[CHECKLISTS]")[-1].strip()


def parse_shepherd_score(text):
    """Mean label value over checklist items (Yes 1, In Progress 0.5, No 0): one label per checklist line, read from
    the evaluation section (after <answer> or CHECKLIST EVALUATION when present)."""
    body = text.split("<answer>")[-1] if "<answer>" in text else text.split("CHECKLIST EVALUATION")[-1]
    item = re.compile(r"(?:—|–|:|-)\s*\**\s*(Yes|In Progress|No)\b", re.I)
    labels = []
    for line in body.splitlines():
        if re.match(r"\s*(?:[-*•]|\d+[.)]|Checklist\b|\*\*)", line, re.I):
            m = item.search(line)
            if m:
                labels.append(m.group(1))
    vals = {"yes": 1.0, "in progress": 0.5, "no": 0.0}
    return sum(vals[l.lower()] for l in labels) / len(labels) if labels else None


def parse_procua(text):
    m = re.search(r'"is_correct"\s*:\s*(true|false)', text, re.I)
    return None if not m else m.group(1).lower() == "true"


# ---- execution ---------------------------------------------------------------------------------------------------
def bound_usd(p):
    chars = len(p.get("instructions") or "") + sum(len(c.get("text", "")) for c in p["input"][0]["content"])
    images = sum(c["type"] == "input_image" for c in p["input"][0]["content"])
    return ((chars / 2.5 + images * 2000) * PRICE["input_cache_write"] + p["max_output_tokens"] * PRICE["output"]) / 1e6


async def call(p, client, key, ledger, sem, log):
    reserved = bound_usd(p)
    async with sem:
        for attempt in range(4):
            ledger.reserve(reserved)
            try:
                r = await client.post("https://api.openai.com/v1/responses", json=p, headers={"Authorization": "Bearer " + key})
                raw = r.json() if r.status_code not in TRANSIENT else {}
                transient = r.status_code in TRANSIENT
            except Exception:
                transient, raw = True, {}
            if transient and not (raw.get("usage") or {}):
                ledger._update(lambda st: st.update(reserved_usd=st["reserved_usd"] - reserved, charged_usd=st["charged_usd"] + reserved))
                if attempt == 3:
                    raise RuntimeError("transient provider failure persisted")
                await asyncio.sleep(5 * 2 ** attempt)
                continue
            break
        usage = raw.get("usage") or {}
        if ledger.settle(reserved, {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]} if "input_tokens" in usage else None) is None:
            raise RuntimeError("missing usage; halted")
    text = output_text(raw)
    log.append(dict(status=raw.get("status"), text=text, usage=usage))
    return text


async def run_state(method, entry, M, client, key, ledger, sem, outdir):
    out = outdir / method / f"{entry['state']}.json"
    if out.exists():
        return
    s = state_inputs(entry); cands = s["candidates"]; log = []
    if method == "shepherd":
        checklist = parse_checklist(await call(M.shepherd_checklist(s), client, key, ledger, sem, log))
        texts = await asyncio.gather(*[call(M.shepherd_judge(s, checklist, c), client, key, ledger, sem, log) for c in cands])
        scores = [parse_shepherd_score(t) for t in texts]
        valid = [x for x in scores if x is not None]
        best = max(valid) if valid else None
        picks = [c["index"] for c, x in zip(cands, scores) if best is not None and x == best] or [c["index"] for c in cands]
        detail = dict(checklist=checklist, scores=scores)
    elif method in ("arbiter", "arbiter_k1"):
        from openwebrl.webprm_worker import native_tournament
        ev = lookahead_evidence(entry, s["anchor"]) if method == "arbiter_k1" else None
        if method == "arbiter_k1" and ev is None:
            write(out, dict(state=entry["state"], method=method, picks=None, detail=dict(skipped="no aligned draw-0 lookahead"), calls=[]))
            return
        compare = (lambda a, b: M.arbiter_compare_k1(s, a, b, ev)) if ev else (lambda a, b: M.arbiter_compare(s, a, b))
        pending = {}

        def compare_sync(left, right):
            return pending[(left["index"], right["index"])]
        # the tournament is sequential: resolve each round's pairs, then replay the native runner on the verdicts
        queue, verdicts = list(cands), {}
        while len(queue) > 1:
            pairs = [(queue[k], queue[k + 1]) for k in range(0, len(queue) - 1, 2)]
            texts = await asyncio.gather(*[call(compare(a, b), client, key, ledger, sem, log) for a, b in pairs])
            winners = []
            for (a, b), t in zip(pairs, texts):
                v = M.arbiter.parse_answer(t)
                v = v if v in (1, 2) else 1                       # missing verdict: keep the first (counted below)
                verdicts[(a["index"], b["index"])] = v
                winners.append((a, b)[v - 1])
            queue = ([queue[-1]] if len(queue) % 2 else []) + winners   # viewer order: leftover bye precedes winners
        pending.update(verdicts)
        winner, trace = native_tournament(list(cands), compare_sync)
        picks = [winner["index"]]
        detail = dict(verdicts={f"{a}-{b}": v for (a, b), v in verdicts.items()}, missing=sum(M.arbiter.parse_answer(x["text"]) not in (1, 2) for x in log))
    else:
        texts = await asyncio.gather(*[call(M.procua_grade(s, c, len(s["anchor"].get("prefix", [])) + 1), client, key, ledger, sem, log) for c in cands])
        correct = [parse_procua(t) for t in texts]
        picks = [c["index"] for c, ok in zip(cands, correct) if ok] or [c["index"] for c in cands]
        detail = dict(correct=correct)
    write(out, dict(state=entry["state"], method=method, picks=picks, detail=detail, calls=log))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "dry-run", "execute"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--states-from"); ap.add_argument("--methods", default="shepherd,arbiter,procua")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        if (root / "plan.json").exists():
            raise SystemExit("already prepared; plan is frozen")
        states = []
        for plan_path in args.states_from.split(","):
            for e in read(plan_path)["states"]:
                done = Path(plan_path).parent / "judgments" / f"{e['state']}.json"
                if done.exists() and read(done):
                    states.append(e)
        root.mkdir(parents=True, exist_ok=True)
        write(root / "plan.json", dict(model=MODEL, effort=EFFORT, methods=args.methods.split(","), states=states, price_usd_per_million=PRICE,
                                       procua_prompt_sha256=hashlib.sha256(PROCUA_PROMPT.read_bytes()).hexdigest(),
                                       note="formulation-faithful WebShepherd / WebArbiter / PRO-CUA with GPT-6 Luna"))
        print(json.dumps(dict(states=len(states), methods=args.methods.split(","))))
    elif args.command == "dry-run":
        plan = read(root / "plan.json"); M = Methods()
        per_state = {"shepherd": 6, "arbiter": 4, "arbiter_k1": 4, "procua": 5}
        calls = sum(per_state[m] for m in plan["methods"]) * len(plan["states"])
        s = state_inputs(plan["states"][0]); c = s["candidates"]
        samples = {"shepherd_checklist": M.shepherd_checklist(s), "shepherd_judge": M.shepherd_judge(s, "Checklist 1: (generated checklist here)", c[0]),
                   "arbiter_compare": M.arbiter_compare(s, c[0], c[1]),
                   **({"arbiter_k1_compare": M.arbiter_compare_k1(s, c[0], c[1], lookahead_evidence(plan["states"][0], s["anchor"]))}
                      if "arbiter_k1" in plan["methods"] else {}), "procua_grade": M.procua_grade(s, c[0], len(s["anchor"].get("prefix", [])) + 1)}
        (root / "samples").mkdir(exist_ok=True)
        bounds = {}
        for k, p in samples.items():
            parts = ["INSTRUCTIONS:\n" + (p.get("instructions") or "")] + [c_.get("text", "[IMAGE]") for c_ in p["input"][0]["content"]]
            (root / "samples" / f"{k}.txt").write_text("\n---\n".join(parts))
            bounds[k] = round(bound_usd(p), 4)
        rep = dict(states=len(plan["states"]), calls=calls, per_call_worst_case_usd=bounds,
                   manifests=dict(arbiter_prompt=M.renderer.manifest["template_sha256"], shepherd=M.shepherd.manifest.get("revision") if isinstance(M.shepherd.manifest, dict) else None))
        write(root / "dry-run.json", rep); print(json.dumps(rep, indent=1))
    else:
        plan, approval = read(root / "plan.json"), read(root / "approval.json")
        if approval.get("approved") is not True or approval.get("plan_sha256") != hashlib.sha256((root / "plan.json").read_bytes()).hexdigest():
            raise SystemExit("approval does not match the frozen plan")
        import run_branch_sol_labels as sol
        sol.PRICE = PRICE                                     # the shared ledger settles at Luna's rates
        from dotenv import dotenv_values
        key = os.environ.get("OPENAI_API_KEY") or dotenv_values(REPO / ".env").get("OPENAI_API_KEY")
        ledger, M, outdir = Ledger(root, approval["caps"]), Methods(), root / "results"
        for m in plan["methods"]:
            (outdir / m).mkdir(parents=True, exist_ok=True)

        async def run():
            import httpx
            sem = asyncio.Semaphore(args.concurrency)
            async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
                gate = asyncio.Semaphore(max(2, args.concurrency // 2))   # states in flight: each loads trees and screenshots

                async def gated(m, e):
                    async with gate:
                        await run_state(m, e, M, client, key, ledger, sem, outdir)
                jobs = [gated(m, e) for m in plan["methods"] for e in plan["states"]]
                for coro in asyncio.as_completed(jobs):
                    try:
                        await coro
                    except RuntimeError as exc:
                        print("stopping on error:", repr(exc), flush=True)
                        return
        asyncio.run(run())
        print(json.dumps({m: len(list((outdir / m).glob("*.json"))) for m in plan["methods"]}))


if __name__ == "__main__":
    main()
