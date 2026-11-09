"""Reproducible typography for paper plots; import before creating any figures.

Standalone renderers add this docs directory to sys.path and import this module.
Fonts are bundled under the SIL Open Font License; no system install is needed.
"""
from pathlib import Path
from matplotlib import font_manager, rcParams
from paper_plot_palette import COLORS

FONT_DIRECTORY = Path(__file__).resolve().parent / "assets/fonts/albert-sans"
FONT_FAMILY = "Albert Sans"


def use_paper_style():
    for path in sorted(FONT_DIRECTORY.glob("AlbertSans-*.ttf")):
        # Static instances give Matplotlib explicit regular/bold/italic weights.
        if "[" not in path.name:
            font_manager.fontManager.addfont(str(path))
    rcParams.update({
        "font.family": FONT_FAMILY,
        "text.color": COLORS["INK"],
        "axes.labelcolor": COLORS["INK"],
        "axes.edgecolor": COLORS["BORDER"],
        "xtick.color": COLORS["MUTED"],
        "ytick.color": COLORS["MUTED"],
        "grid.color": COLORS["BORDER"],
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": FONT_FAMILY + ":italic",
        "mathtext.bf": FONT_FAMILY + ":bold",
        "mathtext.bfit": FONT_FAMILY + ":bold:italic",
        "mathtext.sf": FONT_FAMILY,
        "mathtext.tt": "DejaVu Sans Mono",
        "mathtext.cal": "STIXGeneral:italic",
        "mathtext.fallback": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


use_paper_style()
