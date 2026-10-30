"""Read-only sampled accounting for one owned env_server process tree.

CPU sums each observed process identity's maximum own utime+stime, excluding
child-CPU counters to prevent double counting. Processes that start and exit
between samples are missed, so CPU and peak RSS are sampled lower bounds.
The scope includes env_server, its driver and browser descendants, never the
calling model worker. RSS sums shared pages across processes (not PSS).
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import time


def parse_proc_stat(text, *, uid, page_size):
    """Linux stat comm may contain spaces or parentheses; split after its end."""
    close = text.rfind(')')
    fields = text[close + 2:].split()
    return dict(pid=int(text[:text.find('(')].strip()), uid=uid, ppid=int(fields[1]),
                pgrp=int(fields[2]), session=int(fields[3]),
                own_ticks=int(fields[11]) + int(fields[12]),
                start_ticks=int(fields[19]), rss_bytes=max(0, int(fields[21])) * page_size)


def read_uid_processes(uid, proc_root='/proc'):
    rows, errors = [], 0
    page_size = os.sysconf('SC_PAGE_SIZE')
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid == uid:
                rows.append(parse_proc_stat((directory / 'stat').read_text(), uid=uid,
                                            page_size=page_size))
        except FileNotFoundError:
            # Ordinary process churn is expected during read-only sampling.
            continue
        except (PermissionError, OSError, ValueError, IndexError):
            errors += 1
    return rows, errors


class BrowserResourceSampler:
    interval_seconds = 1.0

    def __init__(self, root_pid, receipt_path, *, uid=None, clock=time.monotonic,
                 reader=read_uid_processes, ticks_per_second=None):
        self.root_pid = root_pid
        self.uid = os.getuid() if uid is None else uid
        self.receipt_path = Path(receipt_path)
        self.clock, self.reader = clock, reader
        self.ticks_per_second = ticks_per_second or os.sysconf('SC_CLK_TCK')
        self.started = clock()
        self.ended = None
        self.root_identity = None
        self.process_ticks = {}
        self.samples = 0
        self.read_errors = 0
        self.peak_rss = 0
        self.peak_processes = 0
        self.integrated_rss = 0.0
        self.previous = None
        self.task = None
        self.stopping = asyncio.Event()
        self.terminal_snapshot = None
        self.browser_phase_metrics = None

    def observe(self, rows, now=None, errors=0):
        """Pure accumulator entry point, also used by deterministic offline tests."""
        now = self.clock() if now is None else now
        self.read_errors += errors
        rows = {r['pid']: r for r in rows if r['uid'] == self.uid}
        root = rows.get(self.root_pid)
        if self.root_identity is None and root is not None:
            self.root_identity = (root['pid'], root['start_ticks'])
        root_live = root is not None and (root['pid'], root['start_ticks']) == self.root_identity
        owned = set()
        for pid, row in rows.items():
            ident = (pid, row['start_ticks'])
            if ident in self.process_ticks or (root_live and
                    (pid == self.root_pid or row['session'] == self.root_pid)):
                owned.add(pid)
        # Include descendants even when they start a separate process session.
        while True:
            added = {pid for pid, row in rows.items() if row['ppid'] in owned} - owned
            if not added:
                break
            owned.update(added)
        rss = 0
        for pid in owned:
            row = rows[pid]
            ident = (pid, row['start_ticks'])
            self.process_ticks[ident] = max(self.process_ticks.get(ident, 0), row['own_ticks'])
            rss += row['rss_bytes']
        if self.previous is not None:
            prev_time, prev_rss = self.previous
            self.integrated_rss += max(0., now - prev_time) * (prev_rss + rss) / 2
        self.previous = now, rss
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_processes = max(self.peak_processes, len(owned))
        self.samples += 1

    async def sample(self):
        try:
            rows, errors = await asyncio.to_thread(self.reader, self.uid)
            self.observe(rows, errors=errors)
        except Exception:
            # Instrumentation never changes browser execution or cleanup policy.
            self.read_errors += 1

    async def _run(self):
        while not self.stopping.is_set():
            try:
                await asyncio.wait_for(self.stopping.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                await self.sample()

    async def start(self):
        await self.sample()
        self.task = asyncio.create_task(self._run())
        return self

    def snapshot(self):
        now = self.ended if self.ended is not None else self.clock()
        return dict(schema='browser-process-tree-sampled-v1',
            scope='Owned env_server + browser/driver descendants; excludes model worker',
            root_pid=self.root_pid, root_start_ticks=self.root_identity[1] if self.root_identity else None,
            cpu_seconds_sampled_lower_bound=sum(self.process_ticks.values()) / self.ticks_per_second,
            peak_aggregate_rss_bytes_sampled=self.peak_rss,
            rss_byte_seconds_sampled_trapezoid=self.integrated_rss,
            session_duration_seconds=max(0., now - self.started),
            sample_count=self.samples, sample_interval_seconds=self.interval_seconds,
            read_error_count=self.read_errors, observed_process_identities=len(self.process_ticks),
            peak_owned_process_count_sampled=self.peak_processes,
            finished=self.ended is not None,
            browser_phase_metrics=self.browser_phase_metrics,
            limitations=['CPU and peak RSS miss processes alive only between samples',
                        'RSS sum double-counts shared pages; not PSS or allocation peak memory',
                        'Session duration includes browser waiting for actor/selector/judge',
                        'RSS-time is sampled trapezoid estimate, not exact occupancy'])

    async def finish(self):
        if self.ended is not None:
            return self.terminal_snapshot
        self.stopping.set()
        if self.task is not None:
            await self.task
        await self.sample()
        self.ended = self.clock()
        self.terminal_snapshot = self.snapshot()
        try:
            # Adjacent to the already-owned browser log under the run root.
            self.receipt_path.write_text(json.dumps(self.terminal_snapshot, indent=2) + '\n')
        except Exception as exc:
            self.terminal_snapshot['receipt_write_error_type'] = type(exc).__name__
        return self.terminal_snapshot
