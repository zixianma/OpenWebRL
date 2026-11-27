# Decision actor: acting by choosing from the page's closed action set

**Goal.** A model that acts on the web end to end by choosing one option from every action available on the page, with no action generator or proposer. Only free text (a field's message, the final answer) comes from a separate typing model. Status, 2026-11-27: harness implemented and unit-tested on CPU. No live runs yet.

Why it might work, and the risk:
- Kev-27B as a direct policy scored 26/300 on OM2W ([RL_EVALUATION](RL_EVALUATION.md#kev27b-actor-full300-20261004)). That run read DOM text only, had no web training, and delegated typing to GPT-4.1-mini.
- This design adds the screenshot and element geometry, uses the page's own element list, and is meant to be trained on web decisions.
- The open risk is that no reasoning happens before acting.

## How it works

| Piece | Implementation |
| --- | --- |
| Options | `openwebrl/decision_actor/options.py`. The env's clickable elements (`find_all_clickable_elements`: tag, text, 0–1 bbox), deduplicated and in reading order. Each element gets a click option; typable fields also get a "type into" option. Optional extras are hover options and grid cells for targets that aren't DOM elements. Fixed actions: scroll up/down, back, wait, Enter, Escape, switch tab, done. All knobs are in `HarnessConfig`, and a digest of it is logged every step. |
| Execution | Each option is an OpenWebRL tool call (`point_2d` = bbox centre, 0–1000). "Type into X" clicks X now; on the next turn the typing model writes the message from the focused state. "Done" asks the typing model for the final response. The policy returns the actor's `<tool_call>` format, so the rollout loop, history and judges are unchanged. |
| Typing model | `typing_model.py`. The served OpenWebRL-4B-SFT with its own full prompt and a forced prefix (an empty thinking block, then `{"name": "write", "arguments": {"message": "`), greedy, at most 160 tokens. |
| Deciders | `deciders.py`. `systemone` sends one choice question per step to any System One endpoint (Jev, a local Kev server, or our trained model behind the same API). `projected_actor` is the harness-ceiling diagnostic below. |
| Eval | `scripts/run_decision_actor_eval.py` registers a `decision:<config.json>` condition on the canonical crossbench harness ([EVAL](EVAL.md)): same plans, judges, local Chromium, records and summaries. The digest-pinned harness files are untouched. Plans freeze the config and the package's source digests. |
| Report | `scripts/decision_actor_report.py`: success with paired differences against a same-day `sft1`, projection coverage and its failure reasons, options per step, decision and typing cost. |

The projection uses the same element hit-test as the branch-state grouping ([final_arm_action_grouping](final_arm_action_grouping.md)): smallest containing box, then the nearest element within 0.03.

## Iteration protocol

The order separates what the **action space** loses from what the **decision model** loses.

1. **Harness ceiling (no learned decider).** Run `projected_sft` and `projected_sft_strict` together with `sft1`, same day, on all three benchmarks.
   - **Setup:** the SFT actor acts as usual, and each action is snapped to an option and executed in its canonical form.
   - **What the strict variant adds:** unrepresentable actions become `wait` instead of escaping.
   - **Pass bar:** strict projected-SFT within noise of `sft1`, with escape rate and first-sample coverage reported per benchmark.
2. **Close coverage gaps, one knob at a time.** Use the report's reason histogram:
   - `no_element_at_point`: grid on, element extractor changes (partly visible elements, iframes, shadow DOM).
   - `unsupported:goto_url` / `new_tab`: URL options from the task and the start page.
   - `other_keys`: more key options.
   - `scroll_container_as_page`: scroll options per container.
   - Re-run step 1 for each variant. Keep a knob only if strict projected-SFT improves.
3. **Zero-shot deciders on the chosen harness:** `jev_text` (needs `TYPESAFE_API_KEY`) and `kev27b_text` (start the Kev server first). This gives the untrained floor on the improved action space, against the 26/300 Kev direct policy.
4. **Trained decider.** Train a pointer chooser on OpenWebRL-4B-SFT over these options:
   - **Labels, in order:** element-mapped soft targets from the 80-proposal Recipe 3 states, branch outcome labels, then public element-level demonstrations.
   - **Serving:** behind System One, as a `systemone` config.
   - **Then:** online RL over the categorical policy.

```bash
R=/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/decision-actor-$(date +%Y%m%d)
for C in sft1 decision:openwebrl/decision_actor/configs/projected_sft.json decision:openwebrl/decision_actor/configs/projected_sft_strict.json; do
  python scripts/run_decision_actor_eval.py prepare --root $R --condition $C --benchmarks online_mind2web --n 300 --seed 20261018 --temperature 1.0 --top-p 0.9
  python scripts/run_decision_actor_eval.py launch-script --root $R --condition $C --benchmarks online_mind2web   # sbatch needs explicit approval
done
python scripts/decision_actor_report.py --dir $R/decision-projected-sft*/online_mind2web --control $R/sft1/online_mind2web
```

<details><summary>Known gaps before the first live run</summary>

- Native `<select>` options aren't enumerated; clicking a select opens a native menu that doesn't appear in the screenshot.
- The `systemone` decider sends text only. Image input needs the provider's documented image schema.
- The decision state has no page title or visible text by default (`page_text_chars` = 0).
- Typing adds one extra browser step per field (click, then write). This counts against the 30-step limit.
- Tests: `tests/test_decision_actor.py` (10 CPU tests). Prepare and launch-script were smoke-tested with a stand-in actor; no GPU or browser run yet.
</details>
