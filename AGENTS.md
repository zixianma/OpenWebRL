# Repository preferences

- Put project documents, analysis notes, experiment plans, and metric references in `openwebrl/docs/`, unless the user specifies another location. This preference was explicitly requested on 2026-09-07.
- Keep this `AGENTS.md` at the repository root so future sessions discover these instructions; it is an agent instruction file, not a project document.
- Before submitting `sbatch`, requesting a new paid allocation, or extending a compute budget, obtain explicit user approval for the exact resource request and budget. Authorization to test within an existing allocation does not authorize new paid jobs.

## Quick baseline resume and supervision

- For a user-authorized **existing** H200 allocation, use `python3 scripts/resume_baseline.py --job-id JOB_ID --dry-run`, inspect the plan, then use `--launch`. See `openwebrl/docs/RESUMING_BASELINE.md` for recovery and monitoring details. Never obtain a new allocation or extend its budget without explicit resource/budget approval.
- The persistent run pointer is `/gpfs/scrubbed/zixianma/openwebrl-runtime/current_baseline.json`. Use its preserved reference source and W&B identity (currently `qcq7i4ug`); do not replace the baseline recipe with experimental working-tree files.
- Resume from a verified completed checkpoint. Replay a saved next batch with the correct submitted-group cursor advance before fresh collection. Keep collected reward observations, completed Adam updates and durable checkpoints distinct.
- Actively inspect the authorized job's progress, logs, GPU/memory use, W&B and saves about every 30–60 seconds. Give progress updates and investigate failures promptly; a background health recorder alone is not active supervision. Refresh the pointer after verifying each new durable checkpoint.
