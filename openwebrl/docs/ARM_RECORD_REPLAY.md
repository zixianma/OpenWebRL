# ARM branch states: prefixes, live replay and record-and-replay

How a branch state is reached, and how the 40 branches (5 actions × 8 continuation draws) re-open it, under the
standard live-replay protocol and the record-and-replay (RR) prototype used by two SFT cohorts.

## Takeaways

- **Every prefix is the actor checkpoint's own rollout.** No state uses SFT-demonstration (teacher) trajectories. Each
  opportunity record gives only a task, a decision depth and seeds. The checkpoint generates every prefix action live,
  and the branches replay those saved outputs. The SFT data only chooses which tasks the Piotr-matched pool contains.
- **RR changes only how the branches re-open the page.** The branch pages are served from a recording of the discovery
  traffic in a deterministic browser, instead of the live web. Candidates, continuations, judging and the strict replay
  check are unchanged.
- **RR helped, but did not fix replay.** The controlled test had 38 completed states. 19 replayed reproducibly from the
  recording, compared with 15 on the live web. In production, 14.3% of RR states that reached replay were released,
  compared with 7.2% under live replay. Two caveats apply. Most replays still fail under RR. The production comparison
  is not controlled.
- **RR was used only by `piotr_sft_rr` and `piotr_sft_rr_diverse`.** These were jobs rr00–rr09, which produced 49 and
  207 unique states. Every later collection used live replay.

<a id="branch-state-prefix-source"></a>

## Where a state's prefix comes from

A collection job reads *opportunities* that have these fields:

- `task_id`, `split` and `task_group`;
- the branch `decision_position` (equal to `depth` + 1);
- the seed namespaces for the prefix, candidates and continuations;
- `archived_teacher_labels_inherited: false`.

An opportunity contains no actions. One episode has three phases:

1. **Discovery (prefix).** The actor checkpoint starts at the task's start URL. For turns 0 … `depth` − 1 it generates
   an action from the screenshot and history, with a seed reserved for the prefix. The browser executes each action
   live. The worker saves each turn's model output and observation as `anchor['prefix']` in
   `openwebrl/arm_branch_worker.py`.
2. **Candidates.** At turn `depth`, the checkpoint samples a fixed number of proposals. The first distinct ones become
   the 5 fixed candidates. No outcome, teacher or probability screening is applied. In diverse cohorts, a second server
   proposes half the draws: the other checkpoint for mixed cohorts, or the same RL checkpoint for run 8.
3. **Branches.** Each of the 40 (candidate, draw) contexts forces the saved prefix outputs and then the candidate.
   It must reconstruct the anchor page before it continues with its own sampling seed.

The checkpoint that reached a state is therefore the cohort's actor. The exception is replay and re-panel cohorts,
which replay a prefix that the other checkpoint generated earlier. The cohort page's *Initial states* column records
this, and so does the final datasets' `prefix_checkpoint` field.

<a id="branch-state-live-replay"></a>

## Live replay (standard protocol)

All 40 branch contexts replay the prefix against the live web, and each must pass a strict check against the anchor.
The check compares the visible-snapshot projection exactly, and the screenshot within a raster tolerance. A state is
released only when all 40 contexts are ready at the same time. The candidate action and the continuations then run.

Live sites change between discovery and replay, through ads, clocks, recommendations, A/B tests and animations. This
makes reconstruction failure the most common rejection.

<a id="branch-state-record-replay"></a>

## Record-and-replay

RR keeps the protocol above. It puts a per-context [mitmproxy](https://mitmproxy.org/) between Chromium and the web,
and makes the browser deterministic.

**Discovery: recording.** Code: `rr-prototype-20261014/collector/rr_worker_entrypoint.py`.

- `mitmdump -w states/<sid>/flows.mitm` records every HTTP(S) response of the prefix rollout.
- The discovery wall-clock time is saved in `states/<sid>/rr-time.json`.
- Chromium is launched with `--proxy-server=127.0.0.1:<port>`, `--proxy-bypass-list=<-loopback>`,
  `--ignore-certificate-errors` and `--disable-quic`.
- The proxy records traffic but does not change it, so the actor sees the live site.

**Branches: replay, then live.** Code: `collector/mitm_rr.py`.

- Each branch context gets its own proxy, loaded with that state's recording.
- **Before release**, every request is answered only from the recording. The match key is the method, scheme, host,
  port, path including the query string, and a SHA-256 of the body. The first recorded response for a key is reused
  for repeated requests. A request with no recorded match is killed and never reaches the live web.
- **After release**, the proxy passes all traffic to the live web for the candidate action and the continuation. The
  switch happens when the controller writes `states/<sid>/release.json`, after all 40 contexts are ready together.

**Determinism.** RR uses a patched source copy, `rr-prototype-20261014/source`, with `RR_DETERMINISM=1`. The patch
applies both to discovery and to the branches.

- **Fixed clock.** The clock is fixed at the discovery time with Playwright `context.clock.set_fixed_time`. The fixed
  clock stays on during continuations.
- **Seeded randomness.** A seeded `Math.random` replaces the browser's own (an LCG with seed 20261014).
- **No motion.** CSS animations and transitions are disabled, the caret is hidden, and `reduced_motion` is set.
- **Settling before capture.** Each capture waits up to 5 s for network idle. It then polls the page state up to 20
  times, 0.3 s apart, until two polls match. Screenshots are taken with `animations: "disabled"`.

**Unchanged from live replay:** the actor and candidate rules, the strict replay check, the 40-ready release barrier,
the continuation sampling and the o4-mini OM2W judge.

<details><summary>Evidence: feasibility test and production counts</summary>

**Stage 1 feasibility test** (`rr-prototype-20261014/rr_stage1.py`, no model or GPU). The test sampled fast1k discovery
attempts and replayed their logged actions in two modes: three times on the live web, and three times from a recording.
A replay counts as reproducible when every step matches the snapshot and the screenshot passes the raster tolerance.

| Run | Browser | States | Completed | Live reproducible | Recording reproducible |
| --- | --- | --- | --- | --- | --- |
| stage1 (361043) | standard | 15 | 13 | 5 | 4 |
| stage1b (361143) | deterministic | 15 | 12 | 5 | 6 |
| stage1c (362196) | deterministic | 44 | 38 | 15 | 19 |

**Production outcomes per discovered state.** The live-replay figures are all fast1k jobs; the RR figures are jobs
rr00–rr09. These counts are not a controlled comparison, for three reasons:

- the two sets ran at different times on different task subsets;
- rr02–rr09 also rejected states for having too few distinct actions;
- both sets ran on the SFT checkpoint and Piotr-matched tasks.

| Outcome | Live replay | % | RR | % |
| --- | --- | --- | --- | --- |
| Discovered | 52,422 | 100.0 | 3,508 | 100.0 |
| Anchor not captured | 18,849 | 36.0 | 714 | 20.4 |
| Too few distinct actions | 0 | 0.0 | 569 | 16.2 |
| Reconstruction failed or timed out before 40-ready release | 28,330 | 54.0 | 1,645 | 46.9 |
| Released | 2,191 | 4.2 | 275 | 7.8 |
| Released among states reaching replay (released + reconstruction failed) | 2,191 of 30,521 | 7.2 | 275 of 1,920 | 14.3 |

Smaller reasons are omitted from the table: duplicates, no accepted training state, malformed panels, worker failures
and missing outcome files. The counts come from every `states/*/release.json` and `reject.json`, counted on
2026-10-24.

</details>
