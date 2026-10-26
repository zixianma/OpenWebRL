"""Fixed-split figure: success of each selector's pick on held-out rollouts 3-5, 3-6 and 3-7 (labels and the oracle use
rollouts 0-2 throughout), main set, valid draws only. Lines only; paired task-clustered intervals are in the JSON."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import STYLE  # noqa: E402

D = Path(__file__).resolve().parent
r = {w: v["pooled"] for w, v in json.loads((D / "curve_fixedfit_main.json").read_text())["valid_only"].items()}
ws = list(r)
xs = range(len(ws))
fig, ax = plt.subplots(figsize=(7.2, 4.3))
for k, (label, color, ls) in STYLE.items():
    if k not in r[ws[0]]:
        continue
    if k == "Best on observed draws":
        label = "Oracle (best on rollouts 0–2)"
    ax.plot(xs, [r[w][k][0] for w in ws], color=color, ls=ls, marker="o", ms=4,
            lw=1.3 if k == "Uniform" else 2.2 if k == "Actor first sample" else 2, label=label)
ax.set_xticks(list(xs), [f"Rollouts {w.replace('-', '–')}" for w in ws])
ax.set_xlim(-.25, len(ws) - .75)
ax.set_xlabel("Held-out evaluation rollouts (labels and oracle use rollouts 0–2)")
ax.set_ylabel("Success (%)")
lo, hi = ax.get_ylim()
ax.set_ylim(min(30, int(lo)), max(40, int(hi) + 1))
ax.grid(alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=9, frameon=False)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(D / f"lookahead_fixed_split_valid_only_main.{ext}", dpi=160)
