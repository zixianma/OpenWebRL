# v9d (2026-10-26): job 390607 (cms-a, 2 workers) died with TimeoutError in can_start -> broker.status(): the native
# launcher hard-codes the broker's handler pool to max_concurrency=1 (fine for the 1-worker Reddit packages), so a
# worker's in-flight model call occupied the only handler and the 5 s status read timed out. The earlier Magento
# launcher used max_concurrency=max_workers; v9d uses max_workers+1 so the admission status read always has a handler.
# exec'd after patch_v9_launcher.py with t = launcher text.
a = "max_concurrency=1,financial_contract=financial"
assert t.count(a) == 1, 'v9d launcher anchor'
t = t.replace(a, "max_concurrency=min(8,limits['max_workers']+1),financial_contract=financial")
