"""Decision actor: a ``browser_action_selector`` that acts by choosing from the page's closed action set.

Per step: build options from the observation -> one decision call -> execute that option's tool call.
Text-producing options defer to the separate typing model: ``type into X`` clicks X now and writes the
message next turn (from the new, focused state); ``done`` writes the final response immediately.
The returned response has the same ``<tool_call>`` format as the SFT actor, so the rollout loop,
history, judge and every eval entry point stay unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time

from openwebrl.decision_actor.deciders import DecisionState, harness_from_config, make_decider
from openwebrl.decision_actor.options import Option, build_options, describe_action, option_summary
from openwebrl.decision_actor.typing_model import ActorTypingModel


def response_text(action: dict) -> str:
    return "</think>\n<tool_call>\n" + json.dumps(action, ensure_ascii=False) + "\n</tool_call><|im_end|>\n"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str) + "\n")
    tmp.replace(path)


@dataclass
class EpisodeState:
    pending: list = field(default_factory=list)      # queued follow-ups: "write" or a literal tool call
    recent: list = field(default_factory=list)       # executed action labels for the decision state
    last_url: str = ""


class DecisionActorPolicy:
    browser_coordinate_space = "normalized_1000"

    def __init__(self, config: dict, output: str | Path, *, client=None, tokenizer=None, decider=None, typing=None):
        self.config = config
        self.harness = harness_from_config(config)
        self.decider = decider or make_decider(config["decider"], client=client)
        self.typing = typing or ActorTypingModel(**(config.get("typing") or {}))
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=True)
        self._tokenizer = tokenizer
        self._actor_path = config.get("actor_tokenizer")
        self.episodes: dict[str, EpisodeState] = {}
        self.halted = False

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._actor_path)
        return self._tokenizer

    def _finish(self, action: dict, meta: dict):
        text = response_text(action)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return (text, ids, [0.0] * len(ids), "stop"), meta

    async def __call__(self, *, infer, url, input_text, sampling_params, images, observation, history, task,
                       task_id, turn, timeout, **unused):
        if turn == 0 or task_id not in self.episodes:
            self.episodes[task_id] = EpisodeState()
        ep = self.episodes[task_id]
        started = time.monotonic()
        page_url = observation.get("active_tab_url") or ""
        record = dict(task_id=task_id, turn=turn, url=page_url, harness=self.harness.digest(),
                      decider=getattr(self.decider, "kind", type(self.decider).__name__))
        typing_args = dict(infer=infer, url=url, input_text=input_text, images=images, timeout=timeout)

        if ep.pending:
            item = ep.pending.pop(0)
            if item == "write":
                message, info = await self.typing("write", **typing_args)
                action = dict(name="write", arguments=dict(message=message))
                record.update(source="typing_model", typing=info)
                if self.harness.submit_after_type == "always":
                    ep.pending.insert(0, dict(name="press_keys", arguments=dict(keys=["Enter"])))
            else:
                action = item
                record.update(source="queued")
            return self._emit(ep, action, f"(follow-up) {describe_action(action)}", record, started)

        options = build_options(observation, self.harness)
        by_id = {o.id: o for o in options}
        state = DecisionState(task=task, url=page_url, title=observation.get("title", ""),
                              recent_actions=ep.recent[-self.harness.history_actions:], options=options,
                              screenshot=observation.get("screenshot"), infer=infer, actor_url=url,
                              input_text=input_text, sampling_params=sampling_params, images=images,
                              timeout=timeout, task_id=task_id, turn=turn)
        decision = await self.decider(state, pending_write=False)
        top = sorted(decision.probabilities.items(), key=lambda kv: -kv[1])[:5]
        record.update(n_options=len(options), n_elements=sum(o.element is not None for o in options),
                      option_kinds=_count(o.kind for o in options), decision=decision.option_id,
                      top5=top, decision_info=decision.info)
        if decision.passthrough is not None:          # projected_actor: actor typed / finished / escaped
            action = decision.passthrough
            record.update(source="actor_passthrough")
            return self._emit(ep, action, f"(actor) {describe_action(action)}", record, started)
        option: Option = by_id[decision.option_id]
        record.update(source="decision", chosen=option_summary(option))
        if option.needs_text == "done":
            response, info = await self.typing("done", **typing_args)
            record["typing"] = info
            action = dict(name="done", arguments=dict(response=response))
        else:
            action = option.action
            if option.needs_text == "write":
                ep.pending.append("write")
        return self._emit(ep, action, option.label, record, started)

    def _emit(self, ep: EpisodeState, action: dict, label: str, record: dict, started: float):
        ep.recent.append(label)
        record.update(action=action, seconds=time.monotonic() - started)
        key = hashlib.sha256(str(record["task_id"]).encode()).hexdigest()[:16]
        write_json(self.root / "turns" / key / f"turn-{record['turn']:03d}.json", record)
        # One line per browser step in the crossbench trace format (read_traces / records / summaries).
        chosen = record.get("chosen") or {}
        unmapped = record.get("source") == "actor_passthrough" and not (record.get("decision_info") or {}).get("mapped", True)
        line = dict(turn=record["turn"], kind="decision_actor", task_id=record["task_id"],
                    selected_index=chosen.get("kind") or record.get("source"), fallback="unmapped_actor_action" if unmapped else None,
                    selector_telemetry=dict(seconds=(record.get("decision_info") or {}).get("seconds"),
                                            n_options=record.get("n_options")),
                    action=action, harness=record["harness"])
        with (self.root / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        meta = dict(mode="decision_actor", decision=record.get("decision"), source=record.get("source"),
                    n_options=record.get("n_options"), seconds=record["seconds"])
        if action.get("name") == "done":
            self.episodes.pop(record["task_id"], None)
        return self._finish(action, meta)


def _count(items):
    out = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out
