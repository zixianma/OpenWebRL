#!/usr/bin/env python3
"""Group the expansion reservoir by known benchmark sites; never edit training data.

Requires tldextract==5.3.0. All inputs are local/frozen; no network or browser use.
WebTailBench supplies no start URLs. Use explicit domains and reviewed named-site
aliases from its V1/V2 instructions, never task-ID prefixes or rubric suggestions.
No-match is provisional for that open-web benchmark, not a leakage guarantee.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import tldextract

REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = ('webvoyager', 'om2w', 'webtailbench', 'deepshop')
DOMAIN_PATTERN = re.compile(r'(?<![\w@.-])(?:https?://)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,63}(?![\w-]|\.[\w-])', re.I)


def normalized_text(value):
    return value.replace('\u2019', "'").replace('\u2018', "'")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + '\n')


class Sites:
    def __init__(self, psl):
        self.extract = tldextract.TLDExtract(
            suffix_list_urls=[Path(psl).resolve().as_uri()], cache_dir=None,
            fallback_to_snapshot=False, include_psl_private_domains=True)

    def parse(self, value):
        url = value if '://' in value else 'https://' + value
        host = (urlsplit(url).hostname or '').lower().rstrip('.')
        host = host.encode('idna').decode('ascii')
        if not host:
            raise ValueError('Missing hostname: ' + value)
        host = host.removeprefix('www.')
        ext = self.extract(host)
        # Private-suffix roots (e.g. tumblr.com) are their own explicit host,
        # not an instruction to match every separately hosted tenant.
        return host, ext.top_domain_under_public_suffix or host, bool(ext.suffix)

    def explicit_domains(self, text):
        return sorted({self.parse(m.group())[0] for m in DOMAIN_PATTERN.finditer(text)
                       if self.parse(m.group())[2]})


def inventory(repo, sources, aliases, sites):
    evidence = []
    source_info = []
    counts = {}
    for benchmark, name, field in [
            ('webvoyager', 'webvoyager_fara.jsonl', 'web'),
            ('om2w', 'online-mind2web.jsonl', 'website'),
            ('deepshop', 'deepshop.jsonl', 'website')]:
        path = repo / 'openwebrl/data/eval' / name
        task_rows = list(rows(path))
        source_info.append(dict(path=str(path.relative_to(repo)), sha256=sha(path)))
        counts[benchmark] = len(task_rows)
        for row in task_rows:
            h, site, _ = sites.parse(row[field])
            evidence.append(dict(benchmark=benchmark, host=h, site=site,
                                 task_id=str(row.get('task_id', row.get('id'))),
                                 source=name, evidence_type='released_start_url'))

    unmatched, matched_by_version = [], {}
    for filename in ['WebTailBench.tsv', 'WebTailBench-v2-rubrics.tsv']:
        path = sources / filename
        source_info.append(dict(path=filename, sha256=sha(path),
            url='https://huggingface.co/datasets/microsoft/WebTailBench/resolve/'
                '50cc93ed008a4d54271ca3e5779e83d4a39ae647/' + filename))
        with path.open() as f:
            tasks = list(csv.DictReader(f, delimiter='\t'))
        assert len({r['id'] for r in tasks}) == len(tasks) == 609
        matched = 0
        for row in tasks:
            text = normalized_text(row['task_summary'])
            hits = [(d, 'instruction_domain', d) for d in sites.explicit_domains(text)]
            for alias in aliases:
                for name in alias['names']:
                    if re.search(r'(?<!\w)' + re.escape(normalized_text(name)) + r'(?!\w)', text, flags=re.I):
                        hits.append((alias['domain'], 'named_site_alias', name))
                        break
            unique_hits = sorted(set(hits))
            if unique_hits:
                matched += 1
            else:
                unmatched.append(dict(source=filename, task_id=row['id'], category=row['benchmark'],
                                      reason='no_explicit_domain_or_reviewed_alias'))
            for domain, kind, mention in unique_hits:
                h, site, known = sites.parse(domain)
                if not known:
                    raise ValueError('Alias has unknown suffix: ' + domain)
                evidence.append(dict(benchmark='webtailbench', host=h, site=site,
                    task_id=row['id'], source=filename, evidence_type=kind, mention=mention))
        matched_by_version[filename] = dict(total_tasks=len(tasks),
            tasks_with_at_least_one_site=matched, no_extracted_site=len(tasks)-matched)
    counts['webtailbench'] = 609
    return evidence, source_info, counts, matched_by_version, unmatched


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool-root', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--aliases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sites = Sites(args.sources / 'public_suffix_list.dat')
    evidence, sources, task_counts, webtail_coverage, unresolved = inventory(
        REPO, args.sources, json.loads(args.aliases.read_text())['aliases'], sites)
    by_site, by_host = defaultdict(set), defaultdict(set)
    for row in evidence:
        by_site[row['site']].add(row['benchmark'])
        by_host[row['host']].add(row['benchmark'])
    write_json(args.output / 'benchmark-site-evidence.json', evidence)
    write_json(args.output / 'webtail-unresolved.json', unresolved)
    disposition = args.pool_root / 'live-browser-full-20260929/task-dispositions.jsonl'
    source_hash = sha(disposition)
    website_counts = defaultdict(Counter)
    host_counts = defaultdict(Counter)
    cohort_counts, group_counts = Counter(), Counter()
    sources_by_group = defaultdict(Counter)
    benchmark_counts = Counter()
    cohort_sites = defaultdict(set)
    available_hosts, seen, unknown_suffixes = set(), set(), set()
    # The three per-task files contain IDs/labels, not full benchmark or page text.
    with (args.output / 'task-groups.jsonl').open('w') as all_out, \
            (args.output / 'available-benchmark-tasks.jsonl').open('w') as yes_out, \
            (args.output / 'available-nonbenchmark-tasks.jsonl').open('w') as no_out:
        for row in rows(disposition):
            key = (row['source'], str(row['task_id']))
            if key in seen:
                raise ValueError('Duplicate task identity: ' + repr(key))
            seen.add(key)
            host, site, known = sites.parse(row['start_url'])
            if not known:
                unknown_suffixes.add(host)
            benchmarks = sorted(by_site[site])
            group = 'benchmark' if benchmarks else 'nonbenchmark_no_known_match'
            item = dict(task_id=key[1], source=key[0], start_url=row['start_url'],
                        host=host, site=site, group=group, benchmarks=benchmarks,
                        exact_host_benchmarks=sorted(by_host[host]),
                        page_availability=row['page_availability'])
            encoded = json.dumps(item, sort_keys=True) + '\n'
            all_out.write(encoded)
            status = row['page_availability']
            website_counts[site][status] += 1
            host_counts[host][status] += 1
            cohort_counts[status] += 1
            if status == 'page_available':
                (yes_out if benchmarks else no_out).write(encoded)
                group_counts[group] += 1
                sources_by_group[group][key[0]] += 1
                cohort_sites[group].add(site)
                available_hosts.add(host)
                benchmark_counts.update(benchmarks)
    assert sum(cohort_counts.values()) == len(seen)
    assert sum(group_counts.values()) == cohort_counts['page_available']
    assert not (cohort_sites['benchmark'] & cohort_sites['nonbenchmark_no_known_match'])
    assert source_hash == sha(disposition), 'Input changed during grouping'
    columns = ['site', 'group', *BENCHMARKS, 'page_available', 'inconclusive', 'unavailable_after_retry', 'total_tasks']
    with (args.output / 'website-groups.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for site, ct in sorted(website_counts.items(), key=lambda kv: (-kv[1]['page_available'], kv[0])):
            writer.writerow(dict(site=site,
                group='benchmark' if by_site[site] else 'nonbenchmark_no_known_match',
                **{b: int(b in by_site[site]) for b in BENCHMARKS},
                **{c: ct[c] for c in ['page_available', 'inconclusive', 'unavailable_after_retry']}, total_tasks=sum(ct.values())))
    for label, group in [('benchmark', 'benchmark'), ('nonbenchmark', 'nonbenchmark_no_known_match')]:
        with (args.output / f'available-{label}-websites.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['site', *BENCHMARKS, 'available_tasks'])
            writer.writeheader()
            for site in sorted(cohort_sites[group], key=lambda s:(-website_counts[s]['page_available'],s)):
                writer.writerow(dict(site=site, **{b:int(b in by_site[site]) for b in BENCHMARKS},
                    available_tasks=website_counts[site]['page_available']))
    with (args.output / 'host-groups.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['host', 'site', 'benchmarks', 'exact_host_benchmarks',
            'page_available', 'inconclusive', 'unavailable_after_retry'])
        writer.writeheader()
        for host, ct in sorted(host_counts.items()):
            site = sites.parse(host)[1]
            writer.writerow(dict(host=host, site=site, benchmarks=';'.join(sorted(by_site[site])),
                exact_host_benchmarks=';'.join(sorted(by_host[host])),
                **{c:ct[c] for c in ['page_available','inconclusive','unavailable_after_retry']}))
    with (args.output / 'benchmark-sites.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['site', *BENCHMARKS, 'available_pool_tasks', 'evidence_types'])
        writer.writeheader()
        for site in sorted(s for s,bs in by_site.items() if bs):
            writer.writerow(dict(site=site, **{b:int(b in by_site[site]) for b in BENCHMARKS},
                available_pool_tasks=website_counts[site]['page_available'],
                evidence_types=';'.join(sorted({r['evidence_type'] for r in evidence if r['site']==site}))))
    stats = dict(
        schema_version=1, date='2026-09-30', classification_only=True,
        matching='PSL registrable domain including private suffixes; www stripped; country domains distinct; start URL only',
        limitations=[
            'WebTailBench has no released start URLs; V1/V2 explicit instruction domains and reviewed aliases are used.',
            'No-known-match is provisional: open-ended tasks can use websites not named in the instruction.',
            'Task-ID prefixes and rubric-only suggestions are not site evidence. Refusals split is excluded.',
            'No brand-family expansion, redirect-destination matching, task-semantic exclusion or training-pool changes.'],
        input=dict(path=str(disposition), sha256=source_hash, tasks=len(seen)),
        inputs=sources + [dict(path='public_suffix_list.dat', sha256=sha(args.sources/'public_suffix_list.dat'),
                             url='https://publicsuffix.org/list/public_suffix_list.dat'),
                        dict(path=str(args.aliases.resolve().relative_to(REPO)), sha256=sha(args.aliases))],
        parser_version=tldextract.__version__,
        available=dict(tasks=cohort_counts['page_available'], websites=sum(map(len, cohort_sites.values())),
            normalized_hosts=len(available_hosts),
            groups={g: dict(tasks=n, websites=len(cohort_sites[g]), sources=dict(sources_by_group[g]))
                    for g,n in sorted(group_counts.items())}),
        all_page_statuses=dict(cohort_counts),
        benchmarks={b:dict(tasks_in_benchmark=task_counts[b],
            inventory_sites=sum(b in bs for bs in by_site.values()),
            available_pool_sites=sum(b in by_site[s] for s in cohort_sites['benchmark']),
            available_pool_tasks=benchmark_counts[b]) for b in BENCHMARKS},
        webtail_extraction_coverage=webtail_coverage,
        unknown_suffix_hosts=sorted(unknown_suffixes),
        largest_sites={g:[dict(site=s,tasks=website_counts[s]['page_available'],benchmarks=sorted(by_site[s]))
            for s in sorted(cohort_sites[g], key=lambda s:(-website_counts[s]['page_available'],s))[:20]]
            for g in sorted(group_counts)},
        artifacts={f.name:sha(f) for f in sorted(args.output.iterdir()) if f.is_file()})
    write_json(args.output / 'summary.json', stats)
    print(json.dumps({k:stats[k] for k in ['available','benchmarks','webtail_extraction_coverage','unknown_suffix_hosts']},indent=2))


if __name__ == '__main__':
    main()
