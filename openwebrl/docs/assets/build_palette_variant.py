#!/usr/bin/env python3
"""Build a separate palette-variant gallery from an isolated, pinned source copy.

Use --render to reproduce, --workspace PATH to collect an already-rendered copy,
or --check to verify the published preview against its pinned source snapshot.
Requires the same local plotting dependencies as the canonical renderers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

DOCS = Path(__file__).resolve().parents[1]
REPO = DOCS.parents[1]
RECIPE = DOCS / "assets/palette_variants/salmon_adm.json"
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FORMATS = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_bytes(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:openwebrl/docs/{path}"], cwd=REPO)


def pinned_text(ref, path):
    return pinned_bytes(ref, path).decode()


def make_workspace(recipe):
    work = Path(tempfile.mkdtemp(prefix="openwebrl-palette-variant-", dir="/tmp"))
    # Keep the archived copy self-contained and reject links to external paths.
    command = ["git", "archive", recipe["source_ref"], "openwebrl/docs", "scripts"]
    process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            target = work / member.name
            if not target.resolve().is_relative_to(work):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream, target.open("wb") as dest:
                    shutil.copyfileobj(stream, dest)
            else:
                raise ValueError(f"Unsupported archive entry: {member.name}")
    if process.wait():
        raise RuntimeError("Could not archive the pinned source")
    palette = work / "openwebrl/docs/assets/paper_plot_palette.json"
    data = json.loads(palette.read_text())
    data["trial"]["colors"].update(recipe["palette_overrides"])
    data["proposal"]["paper_roles"].update(recipe["palette_overrides"])
    data["proposal"]["base_swatches"].update(
        adm=recipe["palette_overrides"]["ADM_DEFAULT"],
        lookahead=recipe["palette_overrides"]["LOOKAHEAD_TEACHER"],
        action=recipe["palette_overrides"]["ACTION"],
        attention=recipe["palette_overrides"]["ATTENTION_ALLOWED"])
    palette.write_text(json.dumps(data, indent=2) + "\n")
    board = work / "openwebrl/docs/assets/preview_paper_palette.py"
    board.write_text(board.read_text().replace('"Shared paper palette"', '"Alternative paper palette"')
        .replace("Purple ADM · yellow states · green actions · salmon teacher · amber oracle / eval",
                 "Salmon ADM · yellow states · purple actions / teacher · amber oracle / eval"))
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLCONFIGDIR=str(work / ".mpl-cache"), XDG_CACHE_HOME=str(work / ".cache"),
               PYTHONDONTWRITEBYTECODE="1")
    for i, args in enumerate(recipe["commands"], 1):
        print(f"[{i}/{len(recipe['commands'])}] {args[0]}", flush=True)
        subprocess.run([sys.executable, *args], cwd=work, env=env, check=True)
    return work


def collect(work, recipe):
    original = pinned_text(recipe["source_ref"], "ARM_FIGURES.md")
    source_docs = work / "openwebrl/docs"
    colors = json.loads((source_docs / "assets/paper_plot_palette.json").read_text())["trial"]["colors"]
    assert all(colors[k] == v for k, v in recipe["palette_overrides"].items())
    root = DOCS / recipe["asset_directory"]
    root.mkdir(parents=True, exist_ok=True)
    mapping, originals, outputs = {}, {}, {}
    for relative in dict.fromkeys(LINK.findall(original)):
        path = Path(relative)
        if path.suffix.lower() not in FORMATS:
            continue
        # Preserve the source taxonomy without repeating arm_results twice.
        tail = Path(*path.parts[1:]) if path.parts[0] == "arm_results" else path
        target = root / tail
        source = source_docs / path
        assert source.is_file(), source
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".svg":
            target.write_text("\n".join(line.rstrip() for line in source.read_text().splitlines()) + "\n")
        else:
            shutil.copyfile(source, target)
        mapping[relative] = target.relative_to(DOCS).as_posix()
        originals[relative] = hashlib.sha256(pinned_bytes(recipe["source_ref"], relative)).hexdigest()
        outputs[target.relative_to(DOCS).as_posix()] = sha(target)
    for ext in ("png", "pdf", "svg"):
        source = source_docs / f"assets/paper_plot_palette_preview.{ext}"
        target = root / f"palette.{ext}"
        if ext == "svg":
            target.write_text("\n".join(line.rstrip() for line in source.read_text().splitlines()) + "\n")
        else:
            shutil.copyfile(source, target)
        outputs[target.relative_to(DOCS).as_posix()] = sha(target)
    text = original.replace("# Figure gallery\n", f"# {recipe['title']}\n", 1)
    text = re.sub(r"<!-- Generated.*?-->\n", "", text, count=1)
    text = text.replace(f"# {recipe['title']}\n", f"# {recipe['title']}\n\n"
        f"[Current palette](ARM_FIGURES.md) · [Alternative color key]({recipe['asset_directory']}/palette.png)\n", 1)
    for old, new in mapping.items():
        text = text.replace(f"]({old})", f"]({new})")
    # Source/reproduction links continue to point at the canonical native code;
    # the explicit recipe supplies the color-only overrides and pinned snapshot.
    text = text.replace("[Reproduce](arm_results/methods/adm_overview/README.md)",
                        "[Reproduce this palette](assets/palette_variants/salmon_adm.json)")
    (DOCS / recipe["document"]).write_text(text)
    manifest = {"schema": "paper-palette-variant-exports-v1", "source_ref": recipe["source_ref"],
                "palette_overrides": recipe["palette_overrides"],
                "figure_count": len(re.findall(r"^!\[", text, re.MULTILINE)),
                "canonical_sha256": originals, "output_sha256": outputs}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check(recipe):
    root = DOCS / recipe["asset_directory"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["palette_overrides"] == recipe["palette_overrides"]
    for path, digest in manifest["output_sha256"].items():
        assert sha(DOCS / path) == digest, path
    for path, digest in manifest["canonical_sha256"].items():
        assert hashlib.sha256(pinned_bytes(recipe["source_ref"], path)).hexdigest() == digest, path
    text = (DOCS / recipe["document"]).read_text()
    original = pinned_text(recipe["source_ref"], "ARM_FIGURES.md")
    expected = len(re.findall(r"^!\[", original, re.MULTILINE))
    assert len(re.findall(r"^!\[", text, re.MULTILINE)) == manifest["figure_count"] == expected
    source_paths = {path for path in LINK.findall(original) if Path(path).suffix.lower() in FORMATS}
    assert source_paths == set(manifest["canonical_sha256"]), "Source gallery coverage changed"
    for relative in LINK.findall(text):
        if "://" not in relative and not relative.startswith("#"):
            assert (DOCS / relative).is_file(), relative
    print(f"Verified {manifest['figure_count']} preview figures against their pinned source snapshot.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--workspace", type=Path)
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(RECIPE.read_text())
    if not args.check:
        original = pinned_text(recipe["source_ref"], "ARM_FIGURES.md")
        current = {p: sha(DOCS / p) for p in LINK.findall(original)
                   if Path(p).suffix.lower() in FORMATS}
        collect(make_workspace(recipe) if args.render else args.workspace.resolve(), recipe)
        assert all(sha(DOCS / p) == digest for p, digest in current.items()), "Canonical figures changed during the variant build"
    check(recipe)


if __name__ == "__main__":
    main()
