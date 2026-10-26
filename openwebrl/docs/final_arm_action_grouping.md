# Final ARM action grouping: from sampled proposals to distinct actions

This page records how sampled actor proposals are grouped into distinct actions, how the executed candidate set is chosen,
and the known weak spots of the rule. The teacher labels built on these actions are in
[final_arm_teacher_design.md](final_arm_teacher_design.md).

## Summary

- **Rule (v2):** two proposals are the same action when every tool call in their bundles has the same type and target.
  Clicks are identified by the page element under the click point (its visible text, else its element index), typed
  text by the text, scrolls by direction, URLs by the URL, and `done`, `wait`, `go_back`, `go_forward`, `new_tab` by the
  tool name alone.
- **Executed set:** of 16 samples, the first proposal of each distinct action in sampling order, up to 5. States with
  fewer than 2 distinct actions are skipped before any execution.
- **Main weak spot: `done` merges different answers.** Among the 357 dataset states with at least two `done` proposals,
  about a quarter (8 of 30 read by hand) contain answers that disagree on the asked fact. That is roughly 3% of all
  2,886 states, and mostly on Wikipedia tasks (7 of the 8).

## Grouping rule

A proposal is parsed into its list of tool calls (`action_key`). It is left out of grouping if generation was cut off
(`finish_reason != stop`), has no tool call after `</think>`, or a call is malformed. Each call maps to a target
(`_target`); a bundle's identity is the list of targets in order.

| Tool | Target |
| --- | --- |
| `click` | the page element under the point (below), identified by its visible text, else its element index |
| `write`, `type` | typed text, lowercased, first 200 characters |
| `scroll` | direction only (amount and point ignored) |
| `goto_url` | URL, lowercased, trailing `/` removed (query strings kept) |
| `done` | one target, whatever the answer |
| `wait`, `go_back`, `go_forward`, `new_tab` | tool name only (e.g. the wait duration is ignored) |
| anything else (`press_keys`, `switch_tab`, …) | tool name plus exact arguments |

Bundles are compared call by call: `click search + write "x" + press Enter` and `click search + write "x" + click Go`
are different actions, as are `click dropdown` and `click dropdown + write "cs.CV" + click Go`.

### How click targets are identified

1. **Element list.** At the branch state the browser captures the screenshot and, in the same capture, a list of
   clickable elements (`openwebrl/env/utils/js.py`, `find_clickable_elements_js_code`; the capture is rejected if the
   page changes between the two). An element is listed when it is
   - an `a`, `button`, `select`, `textarea`, `option` or `input`, has `onclick`, or has role `button`, `option` or `tab`;
     or it has a pointer cursor and is not nested in or around an element already listed;
   - and it is wider and taller than 5 px, fully inside the viewport, not hidden or disabled, and on top at its centre.

   Each entry has a normalized box (0–1) and its text fields (`innerText`, `aria-label`, `textContent`, `placeholder`,
   `value`, `title`, …).
2. **Hit test** (`_target` in `openwebrl/arm_branch_protocol.py`). The click point (0–1000 scale) is divided by 1000.
   Among the elements whose box contains it, the **smallest** box wins.
3. **Identity.** The element's `innerText`, else `aria-label`, else `textContent`, lowercased, first 80 characters.
   Elements with the same text are therefore one target (a search box and its label). An element with none of these
   (for example a bare text input, whose `placeholder` is not used) is identified by its index in the list, which is
   stable within one state.
4. **Snap.** If no element contains the point, the nearest element within 0.03 (30 units) is used.
5. **Grid.** Otherwise the target is the 100 × 100-unit grid cell, e.g. `click(270, 14)` → `grid(3, 0)`.

Grouping is per state, so element indices and grid cells are only ever compared within one page capture.

## Examples (pilot, first 16 samples per state)

`[n✱]` = action executed (first 5 in sampling order); counts are samples.

<details open><summary>Seven states</summary>

**Same element, different coordinates (Liberty Tax, 4 actions)**
```
[0✱] ×7  go_back
[1✱] ×3  click "products"          ← click(470,141) ×2, (469,141)
[2✱] ×5  click "filter services"   ← (810,535), (808,537), (805,535), (810,534), …
[3✱] ×1  click "taxes"             ← (349,141)
```

**Scroll amount ignored, direction kept (Namelix, 5 actions)**
```
[0✱] ×5  scroll down   ← amount 0.7 ×5
[1✱] ×8  scroll up     ← amount 0.7 ×6, 0.8, 0.5
[2✱] ×1  go_back
[3✱] ×1  goto_url businessnamegenerator.com
[4✱] ×1  click "create a unique, professional logo… try brandmark.io"
```

**Bundles differ call by call; an untexted input is identified by its index (arXiv, 7 actions, 2 not executed)**
```
[0✱] ×6  click element#7
[1✱] ×2  click subject dropdown
[2✱] ×4  click "search" + write "computer vision" + click element#7
[3✱] ×1  click dropdown + write "cs.CV" + click element#7
[4✱] ×1  click element#7 + write "computer vision" + press Enter
[5 ] ×1  click dropdown + write "computer vision" + click element#7
[6 ] ×1  click element#7 + write "computer vision" + click element#7
```

**`done` merges answers (dx.com, 2 actions; here the answers agree)**
```
[0✱] ×15 done   ← "dx.com is not currently an active e-commerce w…" ×9, "dx.com is not an active e-commerce website whe…" ×3, …
[1✱] ×1  scroll down
```

**Grid fallback when no element is near (Hugging Face blog, 5 actions)**
```
[0✱] ×11 go_back
[1✱] ×1  click "nvidia nemotron 3.5 embed, rerank, and parse…"
[2✱] ×1  scroll down
[3✱] ×2  click "luo-yihang • 3 days ago • 55"   ← (860,42), (845,42)
[4✱] ×1  click grid(3,0) + write "podcast" + press Enter
```

**`wait` duration ignored; a URL query is a new action (browser error page, 8 actions; excerpt)**
```
[1✱] ×6  new_tab + goto_url scholar.google.com
[2✱] ×3  wait   ← 30 s ×2, 15 s
[5 ] ×1  new_tab + goto_url scholar.google.com/scholar?q=generative+adversarial+networks
[7 ] ×1  done
```

**Skipped: one action only.** Rotten Tomatoes (16/16 click cookie "continue"), Food Network (16/16 "agree"), a blank
page (16/16 `goto_url google.com`).
</details>

## From 16 samples to the executed set

1. 16 proposals are sampled in parallel at temperature 1.0, each with a fixed seed, indexed 0–15.
2. Fewer than 2 distinct actions: the state is skipped before any execution.
3. The executed set is the first proposal of each distinct action in index order, up to 5. When fewer than 5 actions
   exist, the remaining slots are **filled with the next proposals in index order, which repeat chosen actions**
   (`filler`). Each executed candidate is continued once; the teacher sees each distinct action once (its first
   proposal's full text and that execution), never a filler.
4. Proposals of a 6th or later distinct action are not executed and are dropped from the data.

The labels-only collection (planned) uses one round of 16 samples (`diverse_max_draws = 16`; earlier collections
resampled up to 64 until 5 actions were found), first-5, no execution of fillers (about a quarter of executions in the
pilot), and no head test tasks.

## How often `done` hides conflicting answers

`done` proposals in the published branch-state dataset (2,886 states, own-checkpoint proposals):

| States | Count |
| --- | ---: |
| All | 2,886 |
| With ≥ 1 `done` proposal | 498 |
| With ≥ 2 `done` proposals | 357 |
| … all answers identical after normalization | 6 |
| … on Wikipedia tasks | 213 |

Text overlap does not separate rewording from disagreement, so 30 of the 357 states were read by hand (random, seed 1):

| Reading | States |
| --- | ---: |
| Same answer, reworded or with more detail | 20 |
| Conflicting answer to the asked fact | 8 |
| Partly different (same conclusion, different supporting facts or coverage) | 2 |

Conflicts were 7 of 16 sampled Wikipedia states and 1 of 14 others. Examples:
- Vasco da Gama reached India "in 1498" versus "in 1492".
- Village census: "29 inhabitants, 12 households" versus "12 people, 29 households".
- Stockholm population: 1,312,624 (2020), 901,379 (2028) and 1,351,953 (2016) at one state.
- Žilina population: 41,397 versus 80,020.
- Pangasinan tourist spots: the second spot differs between answers.

When answers conflict, the merged action carries only the first `done` proposal's answer: the teacher judges it, and
its continuation outcome scores it, while other proposals of the same action may say something else. At about 3% of
states (≈ 357 × 8/30 of 2,886) this is a small share of the data, and the planned labels-only collection excludes
Wikipedia, where most conflicts occur.

<details><summary>Method and code</summary>

- Grouping: `openwebrl/arm_branch_protocol.py` (`action_key`, `_target`, `diverse_candidates`); sampling rounds:
  `openwebrl/arm_branch_worker.py`.
- Element list: `openwebrl/env/utils/js.py`; boxes normalized in `openwebrl/env/web_env.py`
  (`find_all_clickable_elements`); captured with the screenshot in `openwebrl/controlled_browser_server.py`.
- `done` count: every proposal's last `done` call after `</think>`; answer = its `response` argument. The hand reading
  showed up to five distinct answers per state; a state counts as conflicting when two answers give different values
  for the asked fact.
- The `diverse_candidates` docstring mentions a 50-unit grid; the code uses 100.
</details>
