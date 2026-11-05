"""Valid-only held-out success vs recorded Luna cost per labelled state (421 fast1k states; success on the 372 states
where every action has a valid held-out draw). Variants share their base condition's colour with a hollow marker."""
import json
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


V = Path(__file__).resolve().parent
succ = json.loads((V / "history-effect.json").read_text())["valid_only"]["success"]
cost = {r["root"].split("/")[-1]: r["total_usd"] for r in json.loads((V.parent / "lookahead-costs.json").read_text())
        if r["root"].startswith("lookahead-fast1k")}
STATES = 421
# root, success key, label, colour, marker, filled
POINTS = [("before", "Pre-execution, 3 past steps", "Pre-execution", "#7b4fd0", "o", True),
          ("before-h0", "Pre-execution, no history", "Pre, no history", "#7b4fd0", "o", False),
          ("immediate", "Post-execution, 3 past steps", "Post-execution", "#eb6834", "o", True),
          ("k1", "K=1, 3 past steps", "K=1", "#93bdef", "o", True),
          ("k1-h0", "K=1, no history", "K=1, no history", "#93bdef", "o", False),
          ("k2", "K=2, 3 past steps", "K=2", "#4f8fdc", "o", True),
          ("k3", "K=3, 3 past steps", "K=3", "#1f5fae", "o", True),
          ("k3-h0", "K=3, no history", "K=3, no history", "#1f5fae", "o", False),
          ("k3-nofb", "K=3, no tool feedback in lookahead", "K=3, no tool feedback", "#1f5fae", "s", False),
          ("k3-img1", "K=3, current screenshot only", "K=3, current screenshot only", "#1f5fae", "D", False)]
# label positions in data coordinates (x = USD per 1,000 states, y = %), joined to points by thin leader lines
LABEL_AT = {"before": (8.2, 32.75), "before-h0": (6.6, 32.0), "immediate": (17.0, 34.55), "k1-h0": (17.0, 35.35),
            "k1": (26.4, 36.4), "k2": (25.5, 34.7), "k3": (31.6, 36.3), "k3-h0": (31.6, 35.8), "k3-nofb": (31.6, 35.3),
            "k3-img1": (8.0, 36.3)}
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for root, key, label, color, marker, filled in POINTS:
    if key not in succ or root not in cost:
        continue
    x, y = 1000 * cost[root] / STATES, succ[key][0]
    ax.plot(x, y, marker=marker, ms=9, color=color, mfc=color if filled else "white", mec=color, mew=2, ls="none", zorder=3)
    lx, ly = LABEL_AT[root]
    ax.annotate(label, (x, y), xytext=(lx, ly), textcoords="data", fontsize=8.5, color="#0b0b0b", va="center",
                ha="left", arrowprops=dict(arrowstyle="-", color="#9a9890", lw=0.7, shrinkA=2, shrinkB=6))
for key, label, color, ls in (("Uniform", "Uniform pick", "#9a9890", "--"), ("Actor (first sample)", "Actor (first sample)", "#b04a9b", ":")):
    y = succ[key][0]
    ax.axhline(y, color=color, ls=ls, lw=1.4, zorder=1)
    ax.text(1.0, y + 0.15, f"{label}  (no Luna calls)", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8.5, color="#52514e")
ax.set_xlabel("Luna cost per 1,000 labelled states (USD, 4 calls per state)")
ax.set_ylabel("Held-out success, valid draws only (%)")
ax.set_xlim(0, 40)
ax.grid(alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(V / f"lookahead_success_vs_cost.{ext}", dpi=170)
