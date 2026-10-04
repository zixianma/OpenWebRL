"""Explicit, opt-in fixes to the pinned upstream DOM harness.

Keep upstream-v1 results immutable. actionable-v2 aligns offered targets with
the executor's existing center-point hit test; it does not bypass that guard.
"""
from contextlib import contextmanager

REVISIONS = ('upstream-v1', 'actionable-v2')


def actionable_snapshot(source):
    anchor = "    if (rname==='gridcell' && e.querySelector('button,[role=\"button\"]')) continue;"
    if source.count(anchor) != 1:
        raise ValueError('Unexpected upstream snapshot source')
    return source.replace(anchor,
        "    if (!e.contains(document.elementFromPoint(x,y))) continue;\n" + anchor)


@contextmanager
def revision(browser_module, name):
    if name not in REVISIONS:
        raise ValueError('Unknown Jev harness revision')
    original = browser_module.READ_STATE, browser_module.MARKER
    if name == 'actionable-v2':
        browser_module.READ_STATE = actionable_snapshot(original[0])
        browser_module.MARKER = (
            '(() => { const state=' + browser_module.READ_STATE + '; return state?.marker ?? null; })()')
    try:
        yield
    finally:
        browser_module.READ_STATE, browser_module.MARKER = original
