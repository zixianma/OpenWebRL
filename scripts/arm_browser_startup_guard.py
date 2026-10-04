"""Stop sustained browser startup failures even when some rollouts succeed.

The guard reads only incremental parent-process log records. It does not submit
jobs, change rewards, or classify unsuccessful web tasks as startup failures.
"""
from datetime import datetime
from pathlib import Path
import re
import time


STAMP = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
START = re.compile(r"\[LocalProcessCreate\] start pid=(\d+) port=")
READY = re.compile(r"\[LocalProcessCreate\] ready pid=(\d+) port=")


class BrowserStartupGuard:
    def __init__(self):
        self.offset = 0
        self.partial = b""
        self.attempts = {}
        self.first_start = None
        self.bad_since = None

    def feed(self, line):
        start, ready = START.search(line), READY.search(line)
        if not (start or ready):
            return
        stamp = STAMP.search(line)
        if not stamp:
            return
        at = datetime.strptime(stamp[1], "%Y-%m-%d %H:%M:%S").timestamp()
        if start:
            pid = int(start[1])
            self.attempts[pid] = (at, False)
            if self.first_start is None:
                self.first_start = at
        elif int(ready[1]) in self.attempts:
            pid = int(ready[1])
            self.attempts[pid] = (self.attempts[pid][0], True)

    def check(self, now):
        # Ignore recent in-flight startups, and old collections. A 30s startup
        # normally finishes well inside this 120s allowance.
        self.attempts = {p: v for p, v in self.attempts.items() if now-v[0] <= 900}
        settled = sorted((v for v in self.attempts.values() if now-v[0] >= 120),
                         key=lambda v: v[0])[-100:]
        missing = sum(not ready for _, ready in settled)
        rate = missing / len(settled) if settled else 0.
        bad = (self.first_start is not None and now-self.first_start >= 600
               and len(settled) >= 64 and rate >= .3)
        if not bad:
            self.bad_since = None
        elif self.bad_since is None:
            self.bad_since = now
        stop = bad and now-self.bad_since >= 60
        return dict(mature_startups=len(settled), missing_ready=missing,
                    missing_ready_fraction=rate, sustained_seconds=(
                        now-self.bad_since if self.bad_since is not None else 0),
                    stop=stop, issue=(
                        'Sustained local browser startup degradation: '
                        f'{missing}/{len(settled)} mature attempts lack readiness'
                        if stop else None))

    def poll(self, path, now=None):
        path = Path(path)
        if path.exists():
            if path.stat().st_size < self.offset:
                self.__init__()
            with path.open('rb') as handle:
                handle.seek(self.offset)
                data = self.partial + handle.read()
                self.offset = handle.tell()
            lines = data.split(b'\n')
            self.partial = lines.pop()
            for line in lines:
                self.feed(line.decode(errors='replace'))
        return self.check(time.time() if now is None else now)


def instrument_controller(source):
    """Apply to an inactive controller copy, preserving its scientific recipe."""
    if 'BrowserStartupGuard' in source:
        raise ValueError('Controller already has a startup guard')
    changes = [
        ('import urllib.request\n',
         'import urllib.request\nfrom arm_browser_startup_guard import BrowserStartupGuard\n'),
        ("    children=[]; handles=[]; selector=None; worker=None; validated={}; stage='actor-startup'; current=None\n",
         "    env['SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR']=str(root/'browser-server-logs')\n"
         "    startup_guard=BrowserStartupGuard()\n"
         "    children=[]; handles=[]; selector=None; worker=None; validated={}; stage='actor-startup'; current=None\n"),
        ('            if issue:=health_issue(live,recent): raise RuntimeError(issue)\n',
         "            browser_health=startup_guard.poll(root/'collection.log')\n"
         "            write_json(root/'browser-startup-health.json',browser_health)\n"
         "            if browser_health['stop']: raise RuntimeError(browser_health['issue'])\n"
         '            if issue:=health_issue(live,recent): raise RuntimeError(issue)\n'),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('Unexpected controller shape; do not patch blindly')
        source = source.replace(old, new, 1)
    compile(source, '<startup-guard-controller>', 'exec')
    return source
