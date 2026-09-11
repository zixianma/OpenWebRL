#!/usr/bin/env python3
"""Validate consolidated topic documents, anchors, and local Markdown links."""
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from project_docs import DOCS, document_map, slug


def anchors(markdown):
    found = set(re.findall(r'<a id="([^"]+)"', markdown))
    fence = None
    counts = {}
    for line in markdown.splitlines():
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            token = match[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if heading:
            key = slug(heading[1])
            number = counts.get(key, 0)
            counts[key] = number + 1
            found.add(key + (f'-{number}' if number else ''))
    return found


def inspect():
    mapping = document_map()
    canonical = list(mapping['targets']) + mapping['retained'] + ['README.md']
    markdown = {name: (DOCS / name).read_text() for name in canonical}
    known_anchors = {name: anchors(text) for name, text in markdown.items()}
    retired = set(mapping['documents'])
    issues = []
    for source, entry in mapping['documents'].items():
        target = entry['target']
        start = f'<!-- document:{source}:start -->'
        end = f'<!-- document:{source}:end -->'
        if markdown[target].count(start) != 1 or markdown[target].count(end) != 1:
            issues.append(f'{source}: expected exactly one marked section in {target}')
        if entry['anchor'] not in known_anchors[target]:
            issues.append(f'{source}: missing canonical anchor in {target}')
        if (DOCS / source).exists():
            issues.append(f'{source}: retired file still exists')
    roots = [DOCS.parent.parent / 'README.md', *[DOCS / name for name in canonical]]
    for path in roots:
        for match in re.finditer(r'\[[^\]]*\]\((<[^>]+>|[^\s)]+)', path.read_text()):
            raw = match.group(1).strip('<>')
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc:
                continue
            name = Path(unquote(parsed.path)).name
            if name in retired:
                issues.append(f'{path}: link targets retired file {raw}')
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path.resolve()
            if parsed.path.endswith('.md') and not target.exists():
                issues.append(f'{path}: missing Markdown target {raw}')
            if target.parent == DOCS and target.name in known_anchors and parsed.fragment:
                if unquote(parsed.fragment) not in known_anchors[target.name]:
                    issues.append(f'{path}: missing anchor {raw}')
    return issues


def main():
    issues = inspect()
    print(json.dumps({'canonical_documents': 11,
                      'retired_source_records': len(document_map()['documents']),
                      'issues': issues}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
