"""Plot curve.json: one figure per outcome convention; panels pooled / new / earlier states; six conditions."""
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


D = Path(__file__).resolve().parent
res = json.loads((D / "curve.json").read_text())
STYLE = {"Uniform": ("Uniform pick", C["BASELINE_LIGHT"], "--"), "Before execution": ("Before execution", C["PRE_EXEC"], "-"),
         "Lookahead K=1": ("Lookahead H = 2", C["LOOKAHEAD_TEACHER"], ":"),
         "Lookahead K=3": ("Lookahead H = 4", C["LOOKAHEAD_TEACHER"], "-"), "Best on observed draws": ("Best on observed draws 0..n−1", C["ORACLE"], "-")}
PANELS = [("pooled", "Pooled"), ("lookahead-fast1k-20261013", "New fast1k states"), ("lookahead-v2-20261013", "Earlier states")]
TITLES = {"invalid_as_failure": "A. Invalid draws count as failures",
          "valid_only": "B. Valid draws only (state kept when every action has a valid draw on both sides)"}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else D
for mode, title in TITLES.items():
    r = {int(k): v for k, v in res[mode].items()}
    ns = sorted(r)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    for ax, (key, name) in zip(axes, PANELS):
        for i, (k, (label, color, ls)) in enumerate(STYLE.items()):
            x = [n + (i - 2.5) * 0.045 for n in ns]
            m = [r[n][key][k][0] for n in ns]
            ax.errorbar(x, m, yerr=[[a - r[n][key][k][1] for a, n in zip(m, ns)], [r[n][key][k][2] - a for a, n in zip(m, ns)]],
                        color=color, ls=ls, marker="o", ms=5, lw=1.2 if k == "Uniform" else 1.8, capsize=2, elinewidth=.8, label=label)
        counts = [r[n][key]["states"] for n in ns]
        ax.set_title(name + (f" ({counts[0]} states)" if len(set(counts)) == 1 else f" ({min(counts)}–{max(counts)} states)"), fontsize=11)
        ax.set_xticks(ns, [f"0–{n - 1} | {n}–7\n{c} states" for n, c in zip(ns, counts)], fontsize=8.5)
        ax.set_xlabel("Observed draws (reference picks on these) | scored draws", fontsize=9)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Success of the picked action (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.5, frameon=False)
    fig.suptitle(f"Lookahead labels vs selecting on observed outcomes, as the draw split varies\n{title}", fontsize=11.5)
    fig.text(.5, .06, "Teacher conditions see only draw 0, so the split only changes which draws score them. Bars: 95% task-clustered bootstrap. "
             "New states are only ones whose candidates' outcomes differ, so their rates are lower.", ha="center", fontsize=8, color=C["MUTED"])
    fig.tight_layout(rect=(0, .1, 1, .93))
    for ext in ("png", "svg"):
        fig.savefig(out / f"lookahead_draw_split_{mode}.{ext}", dpi=160)
        if ext == "svg":
            target = out / f'lookahead_draw_split_{mode}.{ext}'
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
