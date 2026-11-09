"""Render the proposed palette and a copy of the label comparison for review.

Reads only public aggregate results and writes separately named previews.
Method and result renderers share the same named semantic color roles.
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
    fig = plt.figure(figsize=(6.8, 3.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 6.8), ylim=(0, 3.15))
    ax.axis("off")
    def text(x, y, s, size=8, **kwargs):
        kwargs.setdefault("color", ROLE["INK"])
        return ax.text(x, y, s, fontsize=size, va="center", **kwargs)
    def chip(x, y, color, width=.34, height=.17):
        ax.add_patch(Rectangle((x, y - height / 2), width, height,
                               facecolor=color, edgecolor="none"))
    text(.18, 2.88, "Four accent colors", 12, weight="semibold")
    text(.18, 2.62, "Purple ADM/actions · salmon lookahead · green state/success · yellow outcome labels", 8)
    rows = [
        ("ADM and actions", "adm", "ADM, candidates, selected actions"),
        ("Lookahead", "lookahead", "future states and teacher evidence"),
        ("State and success", "success", "current context; successful outcomes"),
        ("Outcome labels", "oracle", "oracle, train and eval"),
    ]
    for i, (label, family, description) in enumerate(rows):
        y = 2.18 - .38 * i
        chip(.20, y, BASE[family], .34, .20)
        text(.70, y, label, 8, weight="medium")
        text(2.30, y, description, 7.3, color=ROLE["MUTED"])
    text(.20, .48, "Neutral gray", 7.5, weight="medium")
    chip(1.18, .48, ROLE["BASELINE_DARK"], .26, .15)
    text(1.55, .48, "controls, failures and structure", 7)
    text(.20, .20, "Use line style, marker shape and light tints to separate series within a color family.", 6.7,
         color=ROLE["MUTED"])
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
    palette_import = "from paper_plot_palette import COLORS as C"
    assert code.count(palette_import) == 1
    code = code.replace(palette_import, "C = _PALETTE_ROLE")
    old_output = 'output = HERE / "selection-label-setups-kfold-20261101"'
    assert code.count(old_output) == 1
    code = code.replace(old_output, f"output = Path({str(HERE / 'selection_label_palette_preview')!r})")
    plt.rcParams.update({"text.color": ROLE["INK"], "axes.labelcolor": ROLE["INK"],
                         "axes.edgecolor": ROLE["MUTED"], "xtick.color": ROLE["INK"],
                         "ytick.color": ROLE["INK"]})
    # Use the existing renderer unchanged apart from its display colors/output.
    namespace = {"__file__": str(source), "__name__": "palette_review", "_PALETTE_ROLE": ROLE}
    exec(compile(code, str(source), "exec"), namespace)
    assert [row[2] for row in namespace["ROWS"]] == ["teacher_pre", "teacher_post", "teacher_k3", "teacher_k5", "outcome"]
    return namespace


if __name__ == "__main__":
    palette_board()
    label_comparison()
