# v9e (2026-10-26). Job 390607 (2 workers) died: the launcher's broker.status() read is bounded to 5 s, but the broker
# serves one request at a time (max_concurrency=1), so another worker's in-flight model call starved it ->
# TimeoutError -> campaign failure (admission and post-cell closure both read status). v9d raised max_concurrency to
# max_workers+1, which let WebArbiter's concurrent pairwise tournament calls from one cell overlap and be rejected
# ("Native tool failure or unresolved broker dispatch", job 390653). v9e keeps max_concurrency=1 (per-cell calls stay
# serialized, as in the working Reddit v9b jobs) and lets the read-only status request wait up to 180 s for a handler.
# exec'd after patch_v9_broker.py with NEW = package root.
bp = NEW/'source/openwebrl/webprm_concurrent_broker.py'
bt = bp.read_text()
for a, b in (("    until=min(_deadline(deadline),time.time()+5)\n", "    until=min(_deadline(deadline),time.time()+180)  # v9e: wait behind in-flight calls\n"),
             ("        conn.settimeout(_timeout(until,5));conn.connect(str(_path(socket_path)))\n",
              "        conn.settimeout(_timeout(until,180));conn.connect(str(_path(socket_path)))\n")):
    assert bt.count(a) == 1, ('v9e broker anchor', a)
    bt = bt.replace(a, b)
bp.write_text(bt)
