#!/usr/bin/env python3
"""Download the public web/CUA action datasets listed in the 2026-11-27 catalog.

The catalog (openwebrl/docs/arm_results/rl_integration/public-action-data-catalog-20261127.json)
marks which sources to download and which HF paths to skip because they are seeded
from evaluation benchmarks. Each download is pinned: the resolved commit SHA is
written to <out>/manifest.json so later conversions can cite an exact revision.

Usage:
  python scripts/download_public_action_data.py --out /path/to/storage --dry-run
  python scripts/download_public_action_data.py --out /path/to/storage --tier 1 2
  python scripts/download_public_action_data.py --out /path/to/storage --only webchain webstar
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CATALOG = (Path(__file__).resolve().parents[1] / "openwebrl" / "docs" / "arm_results" /
           "rl_integration" / "public-action-data-catalog-20261127.json")


def select(sources, tiers, only):
    chosen = []
    for source in sources:
        if only:
            if source["id"] in only:
                chosen.append(source)
        elif source.get("download") and (not tiers or source["tier"] in tiers):
            chosen.append(source)
    unknown = set(only or ()) - {s["id"] for s in sources}
    if unknown:
        raise SystemExit(f"Unknown source ids: {sorted(unknown)}")
    excluded = [s["id"] for s in chosen if s["tier"] == 0]
    if excluded:
        raise SystemExit(f"Refusing eval-contaminated sources: {excluded}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tier", type=int, nargs="*", default=[])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    catalog = json.loads(CATALOG.read_text())
    chosen = select(catalog["sources"], set(args.tier), set(args.only))
    for source in chosen:
        print(f"{source['id']:22s} tier {source['tier']}  {source['hf']}  "
              f"license={source.get('license')}  ignore={source.get('ignore', [])}")
    if args.dry_run:
        return

    from huggingface_hub import HfApi, snapshot_download  # imported late so --dry-run needs no deps

    api = HfApi()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for source in chosen:
        sha = api.dataset_info(source["hf"]).sha
        local = snapshot_download(repo_id=source["hf"], repo_type="dataset", revision=sha,
                                  local_dir=args.out / source["id"],
                                  ignore_patterns=source.get("ignore") or None)
        manifest[source["id"]] = dict(hf=source["hf"], revision=sha, local_dir=str(local),
                                      ignore=source.get("ignore", []), license=source.get("license"),
                                      downloaded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"done {source['id']} @ {sha}")


if __name__ == "__main__":
    main()
