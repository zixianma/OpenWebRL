"""Render the proposed palette and a copy of the label comparison for review.

Reads only public aggregate results. The main result figures and the active
palette are not changed; output names explicitly identify these as previews.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
sys.path.insert(0, str(DOCS))
import paper_plot_style  # noqa: E402,F401

CONFIG = json.loads((HERE / "paper_plot_palette.json").read_text())
PROPOSAL = CONFIG["proposal"]
BASE = PROPOSAL["base_swatches"]
ROLE = PROPOSAL["paper_roles"]


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        metadata = ({"Date": None} if ext == "svg" else
                    {"CreationDate": None, "ModDate": None} if ext == "pdf" else {})
        path = HERE / f"{stem}.{ext}"
        fig.savefig(path, dpi=250, facecolor="white", metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def palette_board():
    fig = plt.figure(figsize=(7.4, 4.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 7.4), ylim=(0, 4.4))
    ax.axis("off")
    def text(x, y, s, size=8, **kwargs):
        return ax.text(x, y, s, fontsize=size, va="center", color=ROLE["INK"], **kwargs)
    def chip(x, y, color, width=.34, height=.17):
        ax.add_patch(Rectangle((x, y - height / 2), width, height,
                               facecolor=color, edgecolor="none"))
    text(.18, 4.10, "Proposed paper palette", 12, weight="semibold")
    text(.18, 3.83, "Purple for ADM · salmon for lookahead · yellow for oracle · mint for outcomes / attention", 8)
    text(.18, 3.53, "ROLE", 7, weight="semibold")
    text(2.45, 3.53, "SOURCE SWATCH", 7, weight="semibold")
    text(4.12, 3.53, "PAPER MARKS / VARIANTS", 7, weight="semibold")
    rows = [
        ("ADM", "adm", [("ADM_22M", "default / 22M"), ("ADM_4B", "4B")]),
        ("Lookahead horizons", "lookahead", [("LOOKAHEAD_H1", "H=1"), ("LOOKAHEAD_H4", "H=4"), ("LOOKAHEAD_H6", "H=6")]),
        ("Oracle / train / eval", "oracle", [("TRAIN", "oracle / train"), ("EVAL", "eval")]),
        ("Single-call scaling", "single_call", [("SINGLE_CALL_MAIN", "single call")]),
        ("Success", "success", [("SUCCESS", "small marks")]),
        ("Failure", "failure", [("FAILURE", "hollow circle")]),
        ("Attention allowed", "attention", [("ATTENTION_ALLOWED", "grid fill")]),
    ]
    for i, (label, family, variants) in enumerate(rows):
        y = 3.16 - .38 * i
        text(.18, y, label)
        chip(2.45, y, BASE[family])
        text(2.89, y, BASE[family], 7.5)
        for j, (key, description) in enumerate(variants):
            x = 4.12 + 1.08 * j
            if key == "FAILURE":
                ax.plot(x + .08, y + .025, "o", ms=7, mfc="white", mec=ROLE[key], mew=1.3)
            else:
                chip(x, y + .025, ROLE[key], .22, .13)
            text(x + .29, y + .025, description, 6.7)
            text(x, y - .12, ROLE[key], 6.5)
    text(.18, .34, "Neutrals", 7.5)
    for x, key, label in [(1.05, "INK", "text"), (2.38, "BASELINE_DARK", "controls"),
                          (3.83, "BASELINE_LIGHT", "light control"), (5.39, "BORDER", "borders")]:
        chip(x, .34, ROLE[key], .24, .15)
        text(x + .33, .34, f"{label}  {ROLE[key]}", 6.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in ax.texts:
        box = artist.get_window_extent(renderer)
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), artist.get_text()
    save(fig, "paper_plot_palette_preview")
    plt.close(fig)


def label_comparison():
    source = DOCS / "arm_results/rl_integration/selection-label-setups-kfold-20261101-plot.py"
    code = source.read_text()
    changes = {"#4B5563": ROLE["PRE_EXEC"], "#D8877E": ROLE["LOOKAHEAD_H1"],
               "#6250BE": ROLE["LOOKAHEAD_H4"], "#33246E": ROLE["LOOKAHEAD_H6"],
               "#397FA3": ROLE["ORACLE"], "#222222": ROLE["INK"],
               "#666666": ROLE["MUTED"], "#555555": ROLE["BASELINE_DARK"]}
    for old, new in changes.items():
        code = code.replace(old, new)
    old_output = 'output = HERE / "selection-label-setups-kfold-20261101"'
    assert code.count(old_output) == 1
    code = code.replace(old_output, f"output = Path({str(HERE / 'selection_label_palette_preview')!r})")
    plt.rcParams.update({"text.color": ROLE["INK"], "axes.labelcolor": ROLE["INK"],
                         "axes.edgecolor": ROLE["MUTED"], "xtick.color": ROLE["INK"],
                         "ytick.color": ROLE["INK"]})
    # Use the existing renderer unchanged apart from its display colors/output.
    namespace = {"__file__": str(source), "__name__": "palette_review"}
    exec(compile(code, str(source), "exec"), namespace)
    assert [row[2] for row in namespace["ROWS"]] == ["teacher_pre", "teacher_post", "teacher_k3", "teacher_k5", "outcome"]


if __name__ == "__main__":
    palette_board()
    label_comparison()
