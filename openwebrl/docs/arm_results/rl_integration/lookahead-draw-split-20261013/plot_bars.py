"""Dot plot (file name kept from the earlier bar chart): success of each selector's pick on held-out rollouts 3-7 (labels and the oracle use rollouts 0-2), main set,
valid draws only (curve_fixedfit_main.json, window 3-7). Error bars: paired 95% task-clustered interval of each selector's
difference from the actor's first sample, drawn around the actor's rate plus that difference (absolute intervals are about
+-5 pp for every bar because tasks differ much more than selectors). The actor's first sample, which has no interval of its
own, is the dashed reference line. Position, not bar length, encodes the value, so the cropped axis is honest; prior
formulations reproduced with Luna are drawn as diamonds (hatched bars before)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style
from paper_plot_palette import COLORS as C


sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import STYLE  # noqa: E402

D = Path(__file__).resolve().parent
r = json.loads((D / "curve_fixedfit_main.json").read_text())["valid_only"]["3-7"]["pooled"]
NAMES = {"Best on observed draws": "Oracle\n(rollouts 0–2)", "Lookahead K=3": "Lookahead\nH=4", "Lookahead K=1": "Lookahead\nH=2",
         "Before execution": "Pre-\nexecution", "WebArbiter formulation": "Web-\nArbiter",
         "WebShepherd formulation": "Web-\nShepherd", "PRO-CUA formulation": "PRO-\nCUA", "Uniform": "Uniform",
         "Actor first sample": "Actor\n1st sample"}
keys = [k for k in STYLE if k in r and k != "Actor first sample"]
a = r["Actor first sample"][0]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.axhline(a, color=C["BASELINE_DARK"], lw=1, ls="--", zorder=0)
ax.text(-0.45, a - .5, f"Actor, first sample: {a:.1f}", ha="left", va="top", fontsize=7.5, color=C["BASELINE_DARK"])
for i, k in enumerate(keys):
    m = r[k][0]; d = r["minus_actor"][k]; lo, hi = a + d[1], a + d[2]
    ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="D" if STYLE[k][2] == "-." else "o", ms=7.5, color=STYLE[k][1],
                ecolor=C["MUTED"], capsize=3, lw=1, markeredgecolor="white", markeredgewidth=1, zorder=3)
    ax.text(i, hi + .3, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(keys)), [NAMES[k] for k in keys], fontsize=8); ax.set_xlim(-0.5, len(keys) - 0.5)
ax.set_ylabel("Success on rollouts 3–7 (%)")
ax.set_ylim(25, 50); ax.set_yticks(range(25, 51, 5))  # wide range: intervals read at the scale of the rates
ax.grid(axis="y", alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
for ext in ("png", "svg"):
    path = D / f"lookahead_success_bars_valid_only_main.{ext}"
    fig.savefig(path, dpi=160)
    if ext == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
