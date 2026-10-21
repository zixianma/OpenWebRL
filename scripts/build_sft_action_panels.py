#!/usr/bin/env python3
"""SFT-only action panels for the selection-head ablation round (CPU only).

States: the collector's SFT self-reached, usable, >=2-distinct export (`--export`, rows with checkpoint SFT). In each
state an *action* is a distinct effective (type, target) key under the collector's diverse rule v2 that (a) the SFT
actor proposed and (b) some executed panel candidate carries, so it has continuation outcomes and teacher evidence.
RL-proposed text never enters a panel: when an action's executed candidate was proposed by the RL actor, the SFT
actor's first proposal of the same key represents it (its outcomes and execution evidence are the executed
candidate's, since continuations are always by the SFT actor). States keep >=2 actions.

Per action: the representative SFT text and token ids, every SFT proposal of the key (natural-frequency panels),
the executed panel candidates carrying the key (outcomes pooled over them; the first SFT-proposed one, else the first,
supplies the teacher's execution evidence), and outcomes by draw (0-2 may label, 3-7 are held out).

Split: task-grouped, seeded; whole tasks go to test until it holds >= `--test-frac` of the states.

Writes into --output: actions.jsonl, panels.jsonl + states.jsonl (feature extraction: representatives first, then
the other SFT proposals), label-states.json (the labeller's per-state shown candidates and texts), summary.json.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_selection_head_branch_eval import FALLBACK_VARIANT, prompt_candidates, sha  # noqa: E402

EXPORT = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/branch-selector-scaleup-20261008/"
              "fast1k-20261012/review-cohorts/self_reached_usable_ge2_states.jsonl")
COLLECTOR_SOURCE = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/rl-actor-recollect-20261016/diverse/source")


def read(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, default=EXPORT)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=20261021)
    args = ap.parse_args()
    sys.path.insert(0, str(COLLECTOR_SOURCE))
    from openwebrl.arm_branch_protocol import diverse_candidates

    def key_of(output, evidence):
        try:
            return diverse_candidates([output], 1, evidence)[0][0]["diversity_key"]
        except ValueError:
            return None

    rows = [json.loads(l) for l in open(args.export)]
    rows = [r for r in rows if r["checkpoint"] == "SFT"]
    counts, records, slots = Counter(), [], {}
    for r in rows:
        root = Path(r["ledger_root"])
        if root not in slots:
            slots[root] = defaultdict(dict)
            for (raw,) in sqlite3.connect(f"file:{root / 'state.sqlite3'}?mode=ro", uri=True).execute(
                    "SELECT result_json FROM attempts WHERE status='finished'"):
                x = json.loads(raw)
                if x.get("mode") == "branch" and x.get("released"):
                    v = bool(x["valid"])
                    slots[root][x["state"]].setdefault((x["candidate"], x["repeat"]), [v, v and (x.get("reward") or 0) >= 1])
        anchor_path = Path(read(Path(r["release_path"]).parent / "anchor-reference.json")["path"])
        anchor = read(anchor_path)
        cands, evidence, depth = anchor["candidates"], anchor.get("evidence"), anchor["depth"]
        legacy = "candidate_selection" not in anchor
        events = [read(p) for p in sorted((anchor_path.parent / "events").glob("*.json"))]

        # panel keys and SFT proposals (draw index, output)
        if legacy:
            panel_keys = [key_of(c["output"], evidence) for c in cands]
            proposer = ["sft"] * len(cands)
            sft_draws = [(c["draw"], c["output"]) for c in cands]
            if len(set(panel_keys)) != int(r["distinct_actions"]):
                counts["legacy_distinct_mismatch"] += 1
        else:
            panel_keys = [c["diversity_key"] for c in cands]
            proposer = [c["proposer"] for c in cands]
            draws = {e["record"]["index"]: e["record"] for e in events if e.get("event") == "candidate_draw"}
            if sorted(draws) != list(range(anchor["candidate_selection"]["draws"])):
                counts["skipped_journal_incomplete"] += 1; continue
            sft_draws = [(i, draws[i]["output"]) for i in sorted(draws) if draws[i].get("proposer", "sft" if i % 2 == 0 else "rl") == "sft"]
        sft_by_key = defaultdict(list)
        for i, out in sft_draws:
            if out[3] != "stop":
                counts["sft_draw_truncated"] += 1; continue
            k = panel_keys[[c["draw"] for c in cands].index(i)] if legacy else key_of(out, evidence)
            if k is not None:
                sft_by_key[k].append((i, out))

        slot = slots[root][r["state"]]
        actions = []
        for k in dict.fromkeys(panel_keys):                     # panel order, first occurrence
            if k not in sft_by_key:
                continue
            execs = [i for i, pk in enumerate(panel_keys) if pk == k]
            if any(cands[i]["output"][3] != "stop" for i in execs):
                counts["action_exec_truncated"] += 1; continue
            teacher_exec = next((i for i in execs if proposer[i] == "sft"), execs[0])
            if proposer[teacher_exec] == "sft":
                rep_draw, rep = cands[teacher_exec]["draw"], cands[teacher_exec]["output"]
            else:
                rep_draw, rep = sft_by_key[k][0]
            outcomes = {str(rep_i): [slot.get((i, rep_i)) for i in execs] for rep_i in range(8)}
            actions.append(dict(key=k, execs=execs, teacher_exec=teacher_exec, exec_proposer=proposer[teacher_exec],
                                rep_draw=rep_draw, rep_text=rep[0], rep_ids=rep[1],
                                sft_draws=[i for i, _ in sft_by_key[k]], outcomes=outcomes))
        if len(actions) < 2:
            counts["skipped_lt2_actions"] += 1; continue
        if any(None in v for a in actions for rep_i, v in a["outcomes"].items()):
            counts["skipped_missing_outcome_cell"] += 1; continue

        # verified actor prompt (same reconstruction as build_selection_head_branch_eval)
        reserved = [e["record"] for e in events if e.get("event") == "proposal_batch_reserved" and e["record"].get("turn") == depth]
        if len(reserved) != 1:
            counts["skipped_missing_prompt_hash"] += 1; continue
        target, prompt_tokens = reserved[0]["prompt_sha256"], reserved[0]["prompt_tokens"]
        prompt, fallback = None, None
        for attempt in read(r["release_path"])["attempt_directories"]:
            for rollout in Path(attempt).glob("rollouts/**/*.json"):
                variants = prompt_candidates(read(rollout).get("llm_input_texts") or "", depth)
                prompt = next((v for v in variants if sha(v) == target), None)
                if prompt:
                    break
                if fallback is None and len(variants) > FALLBACK_VARIANT:
                    fallback = variants[FALLBACK_VARIANT]
            if prompt:
                break
        verified = prompt is not None
        if not verified:
            if fallback is None:
                counts["skipped_prompt_not_reconstructed"] += 1; continue
            prompt = fallback; counts["prompt_unverified"] += 1

        # feature candidates: representatives first, then every other SFT proposal of an action's key
        texts = {(a["rep_draw"]): (a["rep_text"], a["rep_ids"]) for a in actions}
        draw_out = dict(sft_draws)
        feature_draws = [a["rep_draw"] for a in actions]
        for a in actions:
            feature_draws += [i for i in a["sft_draws"] if i not in feature_draws]
        for i in feature_draws:
            if i not in texts:
                texts[i] = (draw_out[i][0], draw_out[i][1])
        records.append(dict(
            state_id=r["state"], task_id=anchor["task_id"], cohort=r["cohort"], legacy=legacy, depth=depth,
            decision_position=r.get("decision_position"), root=str(root), collection=r["cohort"],
            image=anchor["before_observation"]["screenshot"]["path"], prompt=prompt, prompt_verified=verified,
            prompt_sha256=target, server_prompt_tokens=prompt_tokens, task=anchor["task"],
            url=anchor["before_observation"].get("active_tab_url", ""), feature_draws=feature_draws,
            feature_texts=[texts[i] for i in feature_draws], actions=actions))
        counts["accepted"] += 1

    # task-grouped split
    tasks = sorted({x["task_id"] for x in records})
    random.Random(args.split_seed).shuffle(tasks)
    per_task = Counter(x["task_id"] for x in records)
    test, n = set(), 0
    for t in tasks:
        if n >= args.test_frac * len(records):
            break
        test.add(t); n += per_task[t]
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "actions.jsonl", "w") as fa, open(args.output / "panels.jsonl", "w") as fp, \
            open(args.output / "states.jsonl", "w") as fs:
        for x in records:
            x["split"] = "test" if x["task_id"] in test else "train"
            fa.write(json.dumps({k: v for k, v in x.items() if k not in ("prompt", "feature_texts")}, ensure_ascii=False) + "\n")
            fp.write(json.dumps(dict(state_id=x["state_id"], base_state_id=x["state_id"], task_id=x["task_id"],
                                     split=x["split"], image=x["image"], server_prompt_tokens=x["server_prompt_tokens"],
                                     candidates=[t for t, _ in x["feature_texts"]],
                                     candidate_token_ids=[ids for _, ids in x["feature_texts"]],
                                     feature_draws=x["feature_draws"]), ensure_ascii=False) + "\n")
            fs.write(json.dumps(dict(state_id=x["state_id"], prompt_text=x["prompt"]), ensure_ascii=False) + "\n")
    (args.output / "label-states.json").write_text(json.dumps([
        dict(state=x["state_id"], collection=x["collection"], root=x["root"],
             candidates=[a["teacher_exec"] for a in x["actions"]], texts=[a["rep_text"] for a in x["actions"]])
        for x in records], ensure_ascii=False))
    n_actions = Counter(len(x["actions"]) for x in records)
    summary = dict(counts=dict(counts), states=len(records), tasks=len(tasks),
                   split=dict(train=sum(x["split"] == "train" for x in records), test=sum(x["split"] == "test" for x in records),
                              test_tasks=len(test)),
                   actions_per_state=dict(sorted(n_actions.items())),
                   actions_rl_exec=sum(a["exec_proposer"] != "sft" for x in records for a in x["actions"]),
                   feature_candidates=sum(len(x["feature_draws"]) for x in records),
                   legacy=sum(x["legacy"] for x in records))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
