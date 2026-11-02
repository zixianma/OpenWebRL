"""Close temporary Playwright connections even when native page setup fails.

Healthy startup uses the frozen connection, viewport, navigation and target
selection operations. A failed startup remains a failed task.
"""
from contextlib import contextmanager

VERSION = "temporary-playwright-session-cleanup-v1"


def live_page(ws, navigate=None):
    from playwright.sync_api import sync_playwright

    manager = sync_playwright().start()
    browser = None
    try:
        browser = manager.chromium.connect_over_cdp(ws, timeout=30000)
        pages = [page for context in browser.contexts for page in context.pages]
        page = pages[-1]
        page.set_viewport_size({"width": 1280, "height": 1800})
        if navigate is not None:
            page.goto(navigate, wait_until="domcontentloaded", timeout=30000)
        return page.context.new_cdp_session(page).send("Target.getTargetInfo")["targetInfo"]["targetId"]
    finally:
        try:
            if browser is not None:
                browser.close()
        finally:
            manager.stop()


@contextmanager
def scoped(worker):
    """Bind before setup/SDK initialization and restore on every exit path."""
    missing = object()
    previous = getattr(worker, "live_page", missing)
    worker.live_page = live_page
    try:
        yield
    finally:
        if previous is missing:
            del worker.live_page
        else:
            worker.live_page = previous
