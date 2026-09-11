"""Stable links and atomic section updates for consolidated project documents."""
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

DOCS = Path(__file__).resolve().parents[1] / 'openwebrl/docs'


@lru_cache(maxsize=1)
def document_map():
    return json.loads((DOCS / 'document_map.json').read_text())


def slug(title):
    title = re.sub(r'<[^>]*>', '', title).lower().strip()
    title = re.sub(r'[^\w\-\s]', '', title)
    return re.sub(r'\s', '-', title)


def document_path(legacy_path):
    path = Path(legacy_path)
    return path.with_name(document_map()['documents'][path.name]['target'])


def rewrite_links(markdown, source_name):
    """Retarget Markdown/HTML links, including old per-document fragments."""
    entries = document_map()['documents']

    def target(url):
        wrapped = url.startswith('<') and url.endswith('>')
        raw = url[1:-1] if wrapped else url
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc:
            if not (parsed.netloc in {'github.com', 'raw.githubusercontent.com'}
                    and '/zixianma/OpenWebRL/' in parsed.path):
                return url
        name = Path(unquote(parsed.path)).name if parsed.path else source_name
        if name not in entries:
            return url
        entry = entries[name]
        fragment = unquote(parsed.fragment)
        anchor = entry['anchor']
        if fragment and fragment != slug(entry['title']):
            anchor += '--' + fragment
        path = parsed.path
        if path:
            path = path.rsplit('/', 1)[0] + '/' + entry['target'] if '/' in path else entry['target']
        else:
            path = entry['target']
        if parsed.scheme:
            path = parsed.scheme + '://' + parsed.netloc + path
        if parsed.query:
            path += '?' + parsed.query
        result = path + '#' + anchor
        return '<' + result + '>' if wrapped else result

    # Preserve labels, optional link titles, and external URLs exactly.
    markdown = re.sub(r'(\]\()(<[^>]+>|[^\s)]+)([^)]*\))',
                      lambda m: m[1] + target(m[2]) + m[3], markdown)
    markdown = re.sub(r'(\bhref=["\'])([^"\']+)(["\'])',
                      lambda m: m[1] + target(m[2]) + m[3], markdown)
    return markdown


def render_section(source_name, markdown):
    entry = document_map()['documents'][source_name]
    lines = markdown.splitlines()
    if lines and lines[0].startswith('# '):
        lines = lines[1:]
    output = [f'<!-- document:{source_name}:start -->',
              f'<a id="{entry["anchor"]}"></a>',
              f'## {entry["title"]}', '',
              f'_Source record: `{source_name}`. Dated entries retain their historical context._', '']
    fence = None
    counts = {}
    for line in lines:
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            token = match[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        heading = re.match(r'^(#{1,6})\s+(.+?)\s*#*$', line) if fence is None and not match else None
        if heading:
            key = slug(heading[2])
            n = counts.get(key, 0)
            counts[key] = n + 1
            suffix = f'-{n}' if n else ''
            output.append(f'<a id="{entry["anchor"]}--{key}{suffix}"></a>')
            line = '#' * min(6, len(heading[1]) + 1) + ' ' + heading[2]
        output.append(line)
    output += ['', f'<!-- document:{source_name}:end -->']
    return rewrite_links('\n'.join(output), source_name)


def write_document_section(legacy_path, markdown):
    """Replace only one source record, preserving neighboring experiment records.

    Historical filenames are keys in document_map.json, never output files.
    A shared directory lock serializes independent SFT/DPO report writers.
    """
    legacy_path = Path(legacy_path)
    target = document_path(legacy_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / '.documentation.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        entry = document_map()['documents'][legacy_path.name]
        old = target.read_text() if target.exists() else '# ' + document_map()['targets'][entry['target']]['title'] + '\n'
        start = f'<!-- document:{legacy_path.name}:start -->'
        end = f'<!-- document:{legacy_path.name}:end -->'
        section = render_section(legacy_path.name, markdown)
        if start in old or end in old:
            if old.count(start) != 1 or old.count(end) != 1 or old.index(start) > old.index(end):
                raise ValueError(f'Malformed documentation section in {target}')
            updated = old[:old.index(start)] + section + old[old.index(end) + len(end):]
        else:
            updated = old.rstrip() + '\n\n' + section + '\n'
        temporary = target.with_suffix('.md.tmp')
        temporary.write_text(updated)
        os.replace(temporary, target)
    return target
