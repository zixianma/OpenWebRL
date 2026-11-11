"""Selection-head model scale (November 11): 3-benchmark mean overall success vs head parameters, with same-window SFT 4B alone.
Joint transformer, soft CE, Recipe 3 GPT-5.5 H = 4 labels, lr 3e-4; 2 runs per point. Error bars: paired 95% bootstrap interval of
each head's difference from SFT 4B alone, drawn around the head's point. Data: selection-head-scale-20261111.json."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/openwebrl-selection-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _s
_s.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _style  # noqa: F401
from paper_plot_palette import COLORS as C

HERE = Path(__file__).parent
d = json.loads((HERE / "selection-head-scale-20261111.json").read_text())["conditions"]
P = {"head5-5fbddd8b2031": 4.36, "head5-fa31796f1fe0": 9.58, "head5-91e7a5a67417": 22.56, "head5-938477dffad2": 272.76}
NAME = {"head5-5fbddd8b2031": "d128-L2", "head5-fa31796f1fe0": "d256-L2", "head5-91e7a5a67417": "d512-L2 (default)", "head5-938477dffad2": "d2048-L4"}
sft = 100 * d["sft1"]["mean_overall"]
fig, ax = plt.subplots(figsize=(4.4, 3.2))
xs = list(P.values()); ys = [100 * d[k]["mean_overall"] for k in P]
ax.plot(xs, ys, color=C["ADM_22M"], linewidth=1.4, zorder=2)
for k, x in P.items():
    y = 100 * d[k]["mean_overall"]; lo, hi = d[k]["minus_sft"]["mean_overall_pp"][1:]
    diff = d[k]["minus_sft"]["mean_overall_pp"][0]
    ax.errorbar(x, y, yerr=[[diff - lo], [hi - diff]], fmt="o", color=C["ADM_22M"], markersize=6, capsize=2, linewidth=0.9, zorder=3)
    dx, dy = {"head5-938477dffad2": (-7, 5), "head5-91e7a5a67417": (7, -5)}.get(k, (7, 5))   # off the line and the error bar
    ax.annotate(f"{NAME[k]}\n{y:.1f}", (x, y), textcoords="offset points", xytext=(dx, dy), ha="right" if dx < 0 else "left",
                va="bottom" if dy > 0 else "top", fontsize=6.5,
                color=C["ADM_22M"], linespacing=1.0)
ax.axhline(sft, color=C["BASELINE_LIGHT"], linestyle="--", linewidth=0.9, zorder=0)
ax.text(400, sft - 0.3, f"SFT 4B alone {sft:.1f}", fontsize=6.5, color=C["MUTED"], va="top", ha="right")
ax.set_xscale("log"); ax.set_xticks([4, 10, 25, 100, 300]); ax.set_xticklabels(["4M", "10M", "25M", "100M", "300M"]); ax.minorticks_off()
ax.set_xlim(2.8, 450); ax.set_ylim(sft - 2.5, max(ys) + 6)
ax.set_xlabel("Head parameters"); ax.set_ylabel("Mean overall success (%)")
ax.grid(axis="y", color=C["BORDER"], linewidth=0.6, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    out = HERE / f"selection-head-scale-20261111.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
    if ext == "svg": out.write_text("\n".join(l.rstrip() for l in out.read_text().splitlines()) + "\n")
