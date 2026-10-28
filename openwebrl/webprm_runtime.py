"""Operational preflight and explicit live-site navigation for WebPRM workers."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time

_PROCESS_RUNTIME = None


def load_api_credentials(env_file=None):
    """Validate credentials before any reservation, journal request or browser."""
    if env_file is not None:
        from dotenv import dotenv_values
        path = Path(env_file)
        if not path.is_file():
            raise ValueError('Explicit API credential file is missing')
        for name, value in dotenv_values(path).items():
            if value is not None and name not in os.environ:
                os.environ[name] = value
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        raise ValueError('OPENAI_API_KEY is absent; no request or reservation created')
    return key


def prepare_browser_runtime():
    """Keep Chromium profiles/sockets and caches in a short, writable local path.

    Chromium startup failed under a long shared-filesystem TMPDIR and passed
    under /tmp. An owning campaign may set WEBPRM_BROWSER_TMP_ROOT to an
    existing canonical directory below /tmp so its storage accounting includes
    profiles and caches. Configure before BrowserGym initializes Playwright.
    """
    global _PROCESS_RUNTIME
    configured = Path(os.environ.get('WEBPRM_BROWSER_TMP_ROOT', '/tmp'))
    parent = configured.resolve()
    if (not configured.is_absolute() or str(configured) != str(parent) or configured.is_symlink()
            or (parent != Path('/tmp') and Path('/tmp') not in parent.parents)
            or not parent.is_dir()):
        raise ValueError('Browser temporary root must be an existing canonical directory under /tmp')
    if _PROCESS_RUNTIME is None or _PROCESS_RUNTIME[0] != os.getpid():
        root = Path(tempfile.mkdtemp(prefix='wprm-', dir=str(parent)))
        _PROCESS_RUNTIME = (os.getpid(), root)
    root = _PROCESS_RUNTIME[1]
    if root.parent != parent or root.resolve() != root or root.is_symlink() or not root.is_dir():
        raise ValueError('Browser runtime root changed after initialization')
    os.environ['TMPDIR'] = str(root)
    tempfile.tempdir = str(root)
    for variable, name in (('XDG_CACHE_HOME', 'cache'), ('XDG_CONFIG_HOME', 'config'),
                           ('XDG_DATA_HOME', 'data')):
        directory = root / name
        if directory.resolve() != directory or directory.is_symlink():
            raise ValueError('Browser cache path must not redirect through a symlink')
        directory.mkdir(exist_ok=True)
        os.environ[variable] = str(directory)
    return root


def navigation_timeout_ms(deadline, limit_ms=30000):
    remaining_ms = int((deadline - time.time()) * 1000)
    if remaining_ms <= 0:
        raise TimeoutError('Pilot deadline reached before initial navigation')
    return min(limit_ms, remaining_ms)


def live_task_type():
    """Declare navigation adaptation without modifying installed BrowserGym."""
    from browsergym.core.task import OpenEndedTask

    class LiveWebPRMTask(OpenEndedTask):
        def __init__(self, seed, start_url, goal=None, *, deadline, navigation_ms=30000, reject_http_errors=True):
            super().__init__(seed, start_url, goal)
            self.deadline = deadline
            self.navigation_ms = navigation_ms
            # Live OM2W (v3) keeps going on HTTP>=400, like the OpenWebRL harness: bot-check
            # pages often clear after their script runs, and the judge scores the outcome.
            self.reject_http_errors = reject_http_errors

        def setup(self, page):
            # 'load' waits for unrelated subresources. A failed navigation still
            # raises; no reload retries, page substitution or success inference.
            # Released WebArena uses this exact delimiter to request multiple
            # initial tabs. Passing the combined string to goto creates a bogus
            # product URL. Keep the existing first page and one shared deadline.
            urls = self.start_url.split(' |AND| ')
            if any(not url.strip() for url in urls):
                raise ValueError('Empty initial URL in WebArena tab list')
            navigations = []
            for index, url in enumerate(urls):
                target = page if index == 0 else page.context.new_page()
                response = target.goto(url, wait_until='domcontentloaded',
                    timeout=navigation_timeout_ms(self.deadline, self.navigation_ms))
                if self.reject_http_errors and response is not None and response.status >= 400:
                    raise RuntimeError(f'Initial website HTTP status {response.status}')
                navigations.append(dict(wait_until='domcontentloaded', timeout_ms=self.navigation_ms,
                    status=None if response is None else response.status,
                    reject_http_errors=self.reject_http_errors, final_url=target.url))
            info = dict(initial_navigation=navigations[0])
            if len(urls) > 1:
                # BrowserGym's load/pageshow hooks can activate a background
                # tab after DOMContentLoaded. Drain initial loads before the
                # final first-tab restoration, as the released viewer does.
                # Each wait consumes the same existing navigation deadline.
                for target in page.context.pages:
                    target.wait_for_load_state('load',
                        timeout=navigation_timeout_ms(self.deadline, self.navigation_ms))
                # The released environment makes the first tab current.
                page.bring_to_front()
                # BrowserGym tracks the active page through a JS binding.
                # Chromium focus alone does not reliably update that pointer
                # in headless mode; await the existing binding explicitly.
                page.evaluate("() => typeof window.browsergym_page_activated === 'function' "
                              "? window.browsergym_page_activated() : undefined")
                info['initial_navigations'] = navigations
            return self.goal, info

    return LiveWebPRMTask
