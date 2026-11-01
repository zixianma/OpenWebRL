"""Heuristic effect signature of a BrowserCode proposal, used to decide agreement.

Exact code strings almost never match across samples (92% of historical N5
decisions had five distinct strings), so agreement compares what an action does:
its side-effect categories, navigation targets and click/type target literals.
All read-only probes share one signature: two observations never change the page.
A final answer never counts as agreement; termination always escalates.
"""
import json
import re

WS = re.compile(r'\s+')
Q = r'["\'`]'
NAV = re.compile(r'(?:Page\.navigate\(\s*\{\s*url\s*:|location\.href\s*=|location\.assign\(|goto\()\s*' + Q + r'([^"\'`]+)')
CLICK = re.compile(r'\.click\(\)|dispatchMouseEvent|mousePressed')
TYPE = re.compile(r'insertText|\.value\s*=|Input\.dispatchKeyEvent\(\{[^}]*type\s*:\s*' + Q + r'char')
KEY = re.compile(r'dispatchKeyEvent|requestSubmit|\.submit\(\)')
SCROLL = re.compile(r'scrollBy|scrollTo|scrollIntoView|mouseWheel|window\.scroll')
LITS = [re.compile(r'includes\(\s*' + Q + r'([^"\'`]{2,})'),
        re.compile(r'querySelector(?:All)?\(\s*' + Q + r'([^"\'`]{2,})'),
        re.compile(r'insertText\(\s*\{\s*text\s*:\s*' + Q + r'([^"\'`]+)'),
        re.compile(r'\.value\s*=\s*' + Q + r'([^"\'`]+)')]


def norm_url(url):
    url = re.sub(r'^https?://', '', url.strip().lower())
    return re.sub(r'^www\.', '', url).rstrip('/')


def effect(code):
    urls = tuple(sorted({norm_url(u) for u in NAV.findall(code)}))
    cats = set()
    if urls:
        cats.add('nav')
    for name, pattern in (('type', TYPE), ('click', CLICK), ('key', KEY), ('scroll', SCROLL)):
        if pattern.search(code):
            cats.add(name)
    if not cats:
        cats.add('read')
    lits = ()
    if cats & {'click', 'type'}:
        lits = tuple(sorted({re.sub(r'[^a-z0-9]', '', m.lower()) for rx in LITS for m in rx.findall(code)}))
    return tuple(sorted(cats)), urls, lits


def signature(proposal):
    """proposal: {'command': BrowserCode JSON string, 'done': bool, 'final_response': str}."""
    if proposal.get('done'):
        return ('done',)
    code = proposal.get('command') or ''
    try:
        parsed = json.loads(code)
        if isinstance(parsed, dict):
            code = parsed.get('code', code)
    except ValueError:
        pass
    return ('act',) + effect(str(code))


def agree(proposals):
    """True when every proposal acts (none terminates) with one shared effect signature."""
    keys = [signature(p) for p in proposals]
    return bool(keys) and keys[0] != ('done',) and all(k == keys[0] for k in keys)
