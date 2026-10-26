"""Bar chart: success of each selector's pick on held-out rollouts 3-7 (labels and the oracle use rollouts 0-2), main set,
valid draws only (curve_fixedfit_main.json, window 3-7). Error bars: paired 95% task-clustered interval of each selector's
difference from the actor's first sample, drawn around the actor's rate plus that difference (absolute intervals are about
+-5 pp for every bar because tasks differ much more than selectors)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import STYLE  # noqa: E402

D = Path(__file__).resolve().parent
r = json.loads((D / "curve_fixedfit_main.json").read_text())["valid_only"]["3-7"]["pooled"]
NAMES = {"Best on observed draws": "Oracle\n(rollouts 0–2)", "Lookahead K=3": "Lookahead\nK=3", "Lookahead K=1": "Lookahead\nK=1",
         "Immediate result": "Post-\nexecution", "Before execution": "Pre-\nexecution", "WebArbiter formulation": "Web-\nArbiter",
         "WebShepherd formulation": "Web-\nShepherd", "PRO-CUA formulation": "PRO-\nCUA", "Uniform": "Uniform",
         "Actor first sample": "Actor\n1st sample"}
keys = [k for k in STYLE if k in r]
fig, ax = plt.subplots(figsize=(7.6, 3.6))
for i, k in enumerate(keys):
    m = r[k][0]
    if k == "Actor first sample":
        lo = hi = m
    else:
        d = r["minus_actor"][k]; a = r["Actor first sample"][0]; lo, hi = a + d[1], a + d[2]
    ax.bar(i, m, color=STYLE[k][1], width=.7, hatch="//" if STYLE[k][2] == "-." else None, edgecolor="white", lw=0)
    if k != "Actor first sample":
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="#333333", capsize=3, lw=1)
    ax.text(i, hi + .4, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(keys)), [NAMES[k] for k in keys], fontsize=8)
ax.set_ylabel("Success on rollouts 3–7 (%)")
ax.set_ylim(30, 44)
ax.grid(axis="y", alpha=.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(D / f"lookahead_success_bars_valid_only_main.{ext}", dpi=160)
