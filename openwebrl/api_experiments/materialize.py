"""Restore one campaign's original source filenames; never launch a worker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent


def materialize(campaign: str, output: Path, *, source_root: Path = SOURCE_ROOT) -> list[Path]:
    """Validate every source before writing a new, code-only output directory."""
    source_root = source_root.resolve()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    index = json.loads((source_root / "source-provenance.json").read_text())
    campaigns = index["snapshots"]
    if campaign not in campaigns:
        raise ValueError(f"Unknown campaign {campaign!r}; choose {', '.join(sorted(campaigns))}")
    files: list[tuple[str, bytes]] = []
    names: set[str] = set()
    for entry in campaigns[campaign]["files"]:
        name = entry["file"]
        if not name or Path(name).name != name or name in names:
            raise ValueError(f"Unsafe or duplicate original filename: {name!r}")
        names.add(name)
        source = (source_root / entry["published_path"]).resolve()
        if not source.is_relative_to(source_root):
            raise ValueError("Published source path escapes the code directory")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {entry['published_path']}")
        files.append((name, data))
    output.mkdir(parents=True, exist_ok=False)
    written = []
    for name, data in files:
        target = output / name
        with target.open("xb") as stream:
            stream.write(data)
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", help="Campaign key from source-provenance.json")
    parser.add_argument("--output", type=Path, required=True, help="New code-only directory")
    args = parser.parse_args()
    files = materialize(args.campaign, args.output)
    print(f"Restored {len(files)} source files in {args.output}. No workers or API calls started.")
    print("Private configuration, task data, credentials and excluded recovery helpers remain local.")


if __name__ == "__main__":
    main()
