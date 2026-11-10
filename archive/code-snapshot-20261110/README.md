# Archived code snapshot (November 10, 2026)

These files are one-off controllers, preparation, recovery, watch and audit scripts for specific cluster jobs,
plus code from side projects (harmful-state collection, Luna/Qwen and reasoning-actor evaluations, WebArena probes,
review templates) and their tests. They were pushed from the shared cluster checkout on November 10, 2026 and moved
here so that `scripts/`, `openwebrl/` and `tests/` keep only maintained code. Paths below this folder mirror the original
repository paths.

They are kept for provenance and are not maintained. Imports between them assume their original locations, so most will
not run from here without adjusting `sys.path`. pytest skips this folder (`conftest.py`).

For continuing RL runs, see `openwebrl/docs/RL_HANDOFF.md`. Each run's exact code is in its `source.tar.gz` in the
`zixianma/openwebrl-rl-checkpoints` Hugging Face repo.
