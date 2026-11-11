"""Build the editable, vector Figure 1 overview.

Run: python openwebrl/docs/arm_results/methods/adm_overview/plot_figure1.py
Requires matplotlib, numpy and PyMuPDF. PNGs are review previews.

Sources: sec/1_intro.tex, sec/3_action_decision_models.tex, sec/5_policy_improvement.tex;
https://github.com/zixianma/OpenWebRL/tree/arm/openwebrl/docs
ARM_FORMULATIONS.md: matched branching and execution-evidence labels;
ARM_SELECTION_HEAD.md: pre-action selection and teacher supervision;
ARM_SUMMARY.md: inference, offline SFT, and action-guided online RL.
Insets read the published Recipe 3 and RL validation aggregates.
The offline inset uses the current paper Table 7 overall-success snapshot. RL uses both outcome controls and the published pooled
trailing-three-evaluation smoother over iterations 20--60. Source hashes and
coordinates are exported in figure1_results.json. No new evaluations are run.
Source screenshots and their provenance are in web_states/. The selected
teacher response and model scores are schematic, not case-study results.
"""

import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import fitz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web_states" / "dinka"
DOCS = ROOT.parents[2]
SOURCE_PATHS = {
    "inference": DOCS / "arm_results/rl_integration/selection-recipe3-vs-cost-20261104.json",
    "offline": ROOT / "figure1_offline_results.json",
    "online": DOCS / "rl_results/rl_eval_learning_curves.json",
}
SOURCE = {key: json.loads(path.read_text()) for key, path in SOURCE_PATHS.items()}
sys.path.insert(0, str(DOCS))
from paper_plot_style import use_paper_style
from paper_plot_palette import COLORS as C, tint
use_paper_style()
# Embed fonts in PDF; outline SVG text so exports need no local font install.
plt.rcParams.update({"font.family": "Albert Sans", "font.size": 6.8,
                     "mathtext.fontset": "custom", "mathtext.rm": "Albert Sans",
                     "mathtext.it": "Albert Sans:italic",
                     "mathtext.bf": "Albert Sans:weight=bold",
                     "mathtext.cal": "STIXGeneral:italic",
                     "mathtext.sf": "Albert Sans", "mathtext.fallback": "stixsans",
                     "pdf.fonttype": 42, "svg.fonttype": "path",
                     "svg.hashsalt": "wdm-figure1"})
# Semantic roles match the collection, architecture and result figures.
INK, MUTED, LINE = C["INK"], C["MUTED"], C["BORDER"]
GRAY, PALE_GRAY = C["BASELINE_DARK"], tint("BASELINE_LIGHT", .92)
STATE_INK, STATE_EDGE, PALE_STATE = C["STATE_INK"], C["STATE_BORDER"], tint("STATE", .65)
ACTION_EDGE, PALE_ACTION = C["ACTION_INK"], tint("ACTION", .65)
TEACHER, PALE_TEACHER = C["LOOKAHEAD_TEACHER"], tint("LOOKAHEAD_EVIDENCE", .90)
PURPLE, PALE_PURPLE = C["ADM_DEFAULT"], tint("ADM_DEFAULT", .92)
FIG_HEIGHT = 3.14
CANVAS_HEIGHT = 790
fig = plt.figure(figsize=(7.2, FIG_HEIGHT), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 1800), ylim=(CANVAS_HEIGHT, 0))
ax.axis("off")


# Give the applications column more room while retaining common gutters.
# These anchors redistribute horizontal space without shrinking the type.
LAYOUT_SOURCE = [0, 638, 664, 1179, 1206, 1800]
LAYOUT_TARGET = [0, 550, 576, 1000, 1028, 1800]


def layout_x(x):
    return float(np.interp(x, LAYOUT_SOURCE, LAYOUT_TARGET))


def layout_y(x, y):
    """Place application headings and dividers in three equal-height bands."""
    if x >= 1206 and y >= 100:
        return float(np.interp(y, [100, 148, 376, 412, 632, 668, 955],
                               [100, 143, 332, 359, 548, 575, 764]))
    return y


def chart_axes(x, y, w, h):
    top, bottom = layout_y(x, y), layout_y(x, y+h)
    return fig.add_axes([layout_x(x)/1800, (CANVAS_HEIGHT-bottom)/CANVAS_HEIGHT,
                         (layout_x(x+w)-layout_x(x))/1800,
                         (bottom-top)/CANVAS_HEIGHT])


def text(x, y, label, size=6.8, color=INK, weight="normal", ha="center", **kw):
    return ax.text(layout_x(x), layout_y(x, y), label, fontsize=size, color=color, weight=weight,
                   ha=ha, va="center", **kw)


def box(x, y, w, h, fill="white", edge=LINE, radius=10, lw=0.65):
    patch = FancyBboxPatch((layout_x(x), layout_y(x, y)),
                          layout_x(x+w)-layout_x(x), layout_y(x, y+h)-layout_y(x, y),
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(patch)
    return patch


def line(points, color=LINE, lw=0.7, dashed=False):
    x, y = zip(*points)
    ax.plot([layout_x(value) for value in x],
            [layout_y(xx, yy) for xx, yy in zip(x, y)], color=color, lw=lw,
            linestyle=(0, (3, 3)) if dashed else "-", zorder=1)


def arrow(start, end, color=MUTED, lw=0.75, dashed=False, scale=6):
    start = (layout_x(start[0]), layout_y(*start))
    end = (layout_x(end[0]), layout_y(*end))
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=scale,
                            color=color, lw=lw, shrinkA=0, shrinkB=0,
                            linestyle=(0, (3, 3)) if dashed else "-")
    ax.add_patch(patch)


def browser(x, y, w=25, h=36, label=None):
    box(x-w/2, y, w, h, PALE_STATE, STATE_EDGE, radius=3, lw=0.5)
    line([(x-w/2+2, y+9), (x+w/2-2, y+9)], STATE_EDGE, lw=0.4)
    if label is not None:
        text(x, y+23, label, size=5.2, color=STATE_INK)
        return
    for dy, frac in [(17, 0.6), (25, 0.4)]:
        line([(x-w/2+5, y+dy), (x-w/2+5+w*frac, y+dy)],
             STATE_EDGE, lw=0.45)


def web_frame(x, y, w, filename):
    """Place an original screenshot without cropping or altering its contents."""
    pixels = plt.imread(WEB / filename)
    x, w = layout_x(x), layout_x(x+w)-layout_x(x)
    h = w * pixels.shape[0] / pixels.shape[1] * (7.2/1800) / (FIG_HEIGHT/CANVAS_HEIGHT)
    ax.imshow(pixels, extent=(x, x+w, y+h, y), aspect="auto",
              interpolation="lanczos", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none",
                           edgecolor=LINE, linewidth=0.55, zorder=4))
    return h


def panel(x, w, title, color=INK, size=8.25):
    text(x+20, 48, title, size=size, weight="bold", ha="left", color=color)


panel(12, 626, "(a) Training data")
panel(664, 515, "(b) Model", color="black")
panel(1206, 582, "(c) Applications")

# (a) Conceptual supervision: proposals + privileged evidence -> teacher choice.
# Detailed rollout/evaluation mechanics belong to the separate data figure.
box(34, 116, 582, 162, PALE_STATE, STATE_EDGE)
web_frame(49, 132, 230, "state.jpg")
text(447, 171, r"Web state $s_t$", size=8.5, color=STATE_INK)
text(447, 224, "Task + browser history", size=6.5, color=MUTED)
arrow((322, 278), (322, 317), GRAY)
text(369, 298, r"$\pi_\theta$", size=7.7, color=GRAY)

box(34, 317, 582, 105, tint("ACTION", .93), ACTION_EDGE, radius=7, lw=0.5)
text(54, 341, r"Actions $\mathcal{A}_t$", size=6.8, ha="left")
centers = [122, 322, 522]
for cx, i in zip(centers, ["1", "2", "K"]):
    box(cx-66, 366, 132, 49, PALE_ACTION, ACTION_EDGE, radius=6)
    text(cx, 391, rf"$a_t^{{({i})}}$", size=8.3, color=ACTION_EDGE)
    arrow((cx, 422), (cx, 445), TEACHER, scale=5)
text(422, 391, r"$\cdots$", size=6.5, color=MUTED)

box(34, 445, 582, 105, PALE_TEACHER, TEACHER, radius=8, lw=1.1)
text(322, 470, r"Privileged lookahead $\mathcal{E}_t$", size=7.8, color=TEACHER, weight="bold")
for cx in centers:
    browser(cx, 492, 68, 42)
arrow((322, 550), (322, 572), TEACHER, lw=1.0, scale=6)

# The teacher sees the state, existing proposals, and execution evidence.
box(157, 572, 330, 88, PALE_TEACHER, TEACHER, radius=9, lw=0.8)
text(322, 598, "Teacher", size=8.5, color=TEACHER, weight="bold")
text(322, 637, r"$\mathcal{T}(s_t,\mathcal{A}_t,\mathcal{E}_t)$", size=9.5, color=TEACHER)
arrow((322, 660), (322, 700), ACTION_EDGE, scale=4)
box(243, 700, 158, 64, PALE_ACTION, ACTION_EDGE, radius=8)
text(322, 732, r"$a_t^{+}$", size=12.2, color=ACTION_EDGE)

# (b) Only state and proposals are available to the student.
for cx, label, fill, color in [
    (804, r"$s_t$", PALE_STATE, STATE_EDGE),
    (1040, r"$\mathcal{A}_t$", PALE_ACTION, ACTION_EDGE),
]:
    box(cx-90, 139, 180, 84, fill, color, radius=7)
    text(cx, 181, label, size=11.5, color=STATE_INK if cx == 804 else ACTION_EDGE)
    line([(cx, 223), (cx, 264), (922, 264)], PURPLE)
arrow((922, 264), (922, 286), PURPLE, lw=0.85)
box(676, 286, 492, 186, PALE_PURPLE, PURPLE, lw=1.3)
text(922, 335, "Action Decision Model", size=9.4, color=PURPLE, weight="bold")
text(922, 412, r"$f_\phi(s_t,\mathcal{A}_t)$", size=13.0, color=PURPLE)
arrow((922, 472), (922, 497), color=PURPLE, scale=4)
text(922, 519, r"$p_\phi$", size=9.5, color=PURPLE)
# The selector has K classes; omit intermediate candidates as in panel (a).
for cx, score, label in [(745, 0.12, "1"), (850, 0.24, "2"), (1100, 0.42, "K")]:
    ax.add_patch(Rectangle((layout_x(cx-26), 618-score*180),
                           layout_x(cx+26)-layout_x(cx-26), score*180,
                           facecolor=PURPLE if label == "K" else tint("ADM_DEFAULT", .50), edgecolor="none"))
    text(cx, 642, rf"$a_t^{{({label})}}$", size=7.2, color=PURPLE if label == "K" else MUTED)
text(975, 642, r"$\cdots$", size=7.2, color=MUTED)
line([(711, 620), (1135, 620)], LINE, lw=0.55)
box(843, 700, 158, 64, PALE_ACTION, ACTION_EDGE)
text(922, 732, r"$\hat a_t$", size=12.2, color=ACTION_EDGE)
arrow((922, 666), (922, 700), color=PURPLE, lw=0.85, scale=5)
text(944, 683, "argmax", size=6.2, color=PURPLE, ha="left")

# (c) Model uses and representative results from separate paper studies.
def application_card(y, height, heading, color):
    text(1246, y+27, heading, size=7.7, weight="bold", ha="left", color=color)


application_card(121, 245, "Inference-time scaling", INK)
# The actor proposes candidates; the ADM selects one for execution.
box(1256, 190, 65, 42, PALE_GRAY, GRAY, radius=5)
text(1288, 211, r"$\pi_\theta$", size=7.6, color=GRAY)
arrow((1321, 211), (1339, 211), GRAY, scale=4)
for cx, label in [(1363, r"$a_t^{(1)}$"), (1433, r"$\cdots$"), (1503, r"$a_t^{(K)}$")]:
    box(cx-24, 190, 48, 42, PALE_ACTION, ACTION_EDGE, radius=5, lw=0.5)
    text(cx, 211, label, size=6.2, color=ACTION_EDGE)
    line([(cx, 232), (cx, 247)], ACTION_EDGE, lw=0.55)
line([(1310, 247), (1503, 247)], ACTION_EDGE, lw=0.55)
box(1264, 263, 92, 43, PALE_PURPLE, PURPLE, radius=5)
text(1310, 284, r"$f_\phi$", size=8.5, color=PURPLE)
arrow((1310, 247), (1310, 263), PURPLE, scale=4)
arrow((1356, 284), (1391, 284), PURPLE, scale=4)
box(1392, 263, 74, 43, PALE_ACTION, ACTION_EDGE, radius=5)
text(1429, 284, r"$\hat a_t$", size=8.0, color=ACTION_EDGE)
arrow((1466, 284), (1496, 284), ACTION_EDGE, scale=4)
browser(1518, 265, 34, 43)
text(1518, 334, "Execute", size=5.8, color=MUTED)
# Selection outputs a decision; the browser icon denotes its execution.

# November 4 SFT-only comparison: three-benchmark valid-only means pooled
# over three runs. Use the Figure 7 PFLOP coordinates with Luna variants omitted.
# Retain its rounded per-benchmark rates and exclusion of pre-execution labels.
def inference_mean(key):
    rows = [r for r in SOURCE["inference"]["rows"] if r["key"] == key]
    assert {r["benchmark"] for r in rows} == {"online_mind2web", "webvoyager", "deepshop"}
    return sum(r["valid_only"] for r in rows) / 3


def inference_pflops(key):
    rows = [r for r in SOURCE["inference"]["rows"] if r["key"] == key]
    return sum(r["pflop_per_episode"] for r in rows) / 3


best_adm4b_key = "lora-gpt55-k3"
inference_methods = [
    ("sft1", "4B", C["BASELINE_LIGHT"], "s", (3, -6), "left"),
    ("sft8b", "8B", GRAY, "s", (3, -7), "left"),
    ("lora-gpt55-k3", "ADM-4B" if best_adm4b_key == "lora-gpt55-k3" else "", "#8246E5", "D", (-3, -9), "right"),
    ("head-gpt55-k3", "ADM-23M", "#8246E5", "o", (5, 2), "left"),
]
inference_points = [(inference_pflops(key), inference_mean(key), label, color, marker, offset, align)
                    for key, label, color, marker, offset, align in inference_methods]
inference_point_keys = {(point[0], point[1]): method[0]
                        for method, point in zip(inference_methods, inference_points)}
parameter_chart = chart_axes(1585, 218, 180, 95)
parameter_chart.set_title("Mean success (%)", fontsize=5.6, color=MUTED, pad=4)
# Frontier of the four displayed Figure 7 systems after omitting Luna variants.
inference_frontier = []
for point in sorted(inference_points, key=lambda p: (p[0], -p[1])):
    if not inference_frontier or point[1] > inference_frontier[-1][1]:
        inference_frontier.append(point)
parameter_chart.plot([p[0] for p in inference_frontier],
                     [p[1] for p in inference_frontier],
                     color=C["FRONTIER"], linestyle="--", linewidth=.8, zorder=1)
parameter_chart.scatter([p[0] for p in inference_frontier], [p[1] for p in inference_frontier],
                        s=36, facecolors="none", edgecolors=C["FRONTIER"], linewidths=.5, zorder=2)
for x, y, label, color, marker, offset, align in inference_points:
    parameter_chart.scatter(x, y, s=25 if marker == "D" else 17, marker=marker,
                            color=color, edgecolors="white", zorder=3, linewidths=0.5)
    if label:
        parameter_chart.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                            ha=align, fontsize=5.6, color=color)
parameter_chart.set(xlim=(.15, 1.62), ylim=(38, 50), xticks=[.3, .9, 1.5], yticks=[40, 45])
parameter_chart.set_xlabel("PFLOPs / episode", fontsize=5.2, labelpad=2, color=MUTED)
parameter_chart.tick_params(axis="both", labelsize=5.5, length=1.8, width=0.45, pad=1.8, colors=MUTED)
parameter_chart.spines[["top", "right"]].set_visible(False)
for spine in parameter_chart.spines.values():
    spine.set_color(LINE)
    spine.set_linewidth(0.45)

application_card(385, 237, "Offline improvement", INK)
# One ADM supplies demonstrations (SFT) and preferences (DPO/RPO).
box(1256, 501, 70, 48, PALE_PURPLE, PURPLE, radius=5)
text(1291, 525, r"$f_\phi$", size=8.0, color=PURPLE)
line([(1326, 525), (1337, 525)], PURPLE)
line([(1337, 479), (1337, 572)], PURPLE)
for yy, method, label in [
    (455, "SFT", r"$\hat a_t$"),
    (548, "DPO / RPO", r"$\hat a_t\succ a_t^{-}$"),
]:
    text(1400, yy-15, method, size=6.0, color=PURPLE)
    arrow((1337, yy+24), (1347, yy+24), PURPLE, scale=3)
    box(1348, yy, 104, 48, PALE_ACTION, ACTION_EDGE, radius=5)
    text(1400, yy+24, label, size=6.7, color=ACTION_EDGE)
    arrow((1452, yy+24), (1473, yy+24), ACTION_EDGE, scale=4)
    box(1474, yy, 64, 48, PALE_GRAY, GRAY, radius=5)
    text(1506, yy+24, r"$\pi_\theta$", size=7.6, color=GRAY)
offline_chart = chart_axes(1585, 468, 180, 118)
offline_chart.set_title("OM2W success (%)", fontsize=5.6, color=MUTED, pad=4)
offline_keys = ["actor", "dpo", "rpo"]
offline_values = [SOURCE["offline"]["rates"][key]["overall"] for key in offline_keys]
offline_chart.bar(range(3), offline_values, width=0.66,
                  color=[GRAY, tint("ADM_DEFAULT", .35), PURPLE],
                  edgecolor="none")
for x, value in enumerate(offline_values):
    offline_chart.text(x, value+1.7, f"{value:.1f}", ha="center", va="bottom", size=5.5,
                       color=PURPLE if x == 2 else MUTED)
offline_chart.set(xlim=(-0.65, 2.65), ylim=(0, 49), xticks=[0, 1, 2], yticks=[0, 40])
offline_chart.set_xticklabels([SOURCE["offline"]["rates"][key]["label"] for key in offline_keys])
offline_chart.tick_params(axis="both", labelsize=5.5, length=1.8, width=0.45, pad=1.8, colors=MUTED)
offline_chart.tick_params(axis="x", length=0)
offline_chart.spines[["top", "right"]].set_visible(False)
for spine in offline_chart.spines.values():
    spine.set_color(LINE)
    spine.set_linewidth(0.45)

application_card(641, 274, "Online RL", INK)
# The same ADM supplies the decision signal used by the online RL bonus.
box(1256, 704, 70, 48, PALE_PURPLE, PURPLE, radius=5)
text(1291, 730, r"$f_\phi$", size=8.0, color=PURPLE)
arrow((1326, 730), (1347, 730), PURPLE, scale=4)
box(1348, 704, 104, 48, PALE_ACTION, ACTION_EDGE, radius=5)
text(1400, 730, r"$\hat a_t$", size=8.0, color=ACTION_EDGE)
arrow((1452, 730), (1473, 730), ACTION_EDGE, scale=4)
box(1474, 704, 64, 48, PALE_GRAY, GRAY, radius=5)
text(1506, 730, r"$\pi_\theta$", size=7.6, color=GRAY)
arrow((1400, 752), (1400, 790), ACTION_EDGE, scale=4)
box(1256, 790, 282, 85, PALE_PURPLE, PURPLE, radius=6)
text(1397, 833, "ADM reward", size=6.6, color=PURPLE)
# Show iterations 20--60 with the published pooled trailing-three smoother.
# success/valid counts, with both outcome-only controls and raw checkpoint dots.
RL_START, RL_STOP = 20, 60
chart = chart_axes(1606, 730, 138, 175)
chart.set_title("OM2W success (%)", fontsize=5.6, color=MUTED, pad=4)
rl_styles = [("outcome48", C["BASELINE_LIGHT"], "48", "--"),
             ("outcome56", GRAY, "56", "-"),
             ("gate-b", PURPLE, "+ ADM", "-")]
for key, color, label, linestyle in rl_styles:
    method = next(m for m in SOURCE["online"]["methods"] if m["id"] == key)
    rows = [r for r in method["points"] if RL_START <= r["iteration"] <= RL_STOP]
    x = np.array([r["iteration"] for r in rows], dtype=float)
    raw = np.array([100*r["successes"]/r["valid"] for r in rows])
    smoothed = []
    for i in range(len(rows)):
        window = rows[max(0, i - 2):i + 1]
        smoothed.append(100 * sum(r["successes"] for r in window) / sum(r["valid"] for r in window))
    assert np.allclose(smoothed, [r["percent"] for r in method["smoothed"]
                                  if RL_START <= r["iteration"] <= RL_STOP])
    chart.scatter(x, raw, color=color, s=5, alpha=.25, edgecolors="none")
    chart.plot(x, smoothed, color=color, lw=1.2, linestyle=linestyle,
               marker="o", ms=1.9, markeredgewidth=0)
chart.set(xlim=(RL_START - 2, RL_STOP + 2), ylim=(35, 57),
          xticks=[RL_START, RL_STOP], yticks=[40, 50])
chart.set_xlabel("Iteration", fontsize=5.6, labelpad=1.5, color=MUTED)
chart.tick_params(axis="both", labelsize=5.8, length=1.8, width=0.45, pad=1.8, colors=MUTED)
chart.spines[["top", "right"]].set_visible(False)
for spine in chart.spines.values():
    spine.set_color(LINE)
    spine.set_linewidth(0.45)
text(1248, 925, "Outcome:", size=5.3, color=MUTED, ha="left")
for x, (_, color, label, linestyle) in zip([1320, 1390, 1460], rl_styles):
    line([(x, 925), (x + 16, 925)], color, lw=1.1, dashed=linestyle == "--")
    text(x + 24, 925, label, size=5.6, color=color, ha="left")

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
right_labels = [t for t in ax.texts if t.get_position()[0] >= layout_x(1206)]
for inset in (parameter_chart, offline_chart, chart):
    right_labels.extend([inset.title, inset.xaxis.label, inset.yaxis.label, *inset.texts])
    right_labels.extend(inset.get_xticklabels() + inset.get_yticklabels())
right_labels = [t for t in right_labels if t.get_visible() and t.get_text()]
for i, label in enumerate(right_labels):
    bounds = label.get_window_extent(renderer)
    assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), label.get_text()
    for other in right_labels[:i]:
        assert not bounds.overlaps(other.get_window_extent(renderer)), (label.get_text(), other.get_text())
    for guide in ax.lines:
        if (min(guide.get_xdata()) >= layout_x(1206)
                and np.ptp(guide.get_ydata()) < 1e-8):
            assert not bounds.overlaps(guide.get_window_extent(renderer).padded(.25)), (
                label.get_text(), list(guide.get_xdata()), list(guide.get_ydata()))

result = {
    "sources": {key: {"path": str(path.relative_to(DOCS)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for key, path in SOURCE_PATHS.items()},
    "inference": {"metric": "Equal-weight three-benchmark mean valid-only success (%)",
                  "best_adm4b_key": best_adm4b_key,
                  "figure7_reference": {"image_blob": "adebff6875ecafe30c4e7874fe40f933a8eec90c",
                                        "source_ref": "d7d4a1d",
                                        "renderer_blob": "3be727b64855167b0e649d123ec58643c73838bf"},
                  "frontier_scope": "Four displayed Figure 7 SFT-actor systems; Luna and pre-execution-label variants omitted",
                  "frontier": [{"key": inference_point_keys[(p[0], p[1])],
                                "label": p[2], "pflop_per_episode": p[0], "percent": p[1]}
                               for p in inference_frontier],
                  "points": [{"key": method[0], "label": point[2],
                              "pflop_per_episode": point[0], "percent": point[1]}
                             for method, point in zip(inference_methods, inference_points)]},
    "offline": {"metric": SOURCE["offline"]["metric"],
                "source": SOURCE["offline"]["source"],
                "rates": {key: SOURCE["offline"]["rates"][key] for key in offline_keys}},
    "online": {"protocol": SOURCE["online"]["protocol"],
               "smoothing": SOURCE["online"]["smoothing"],
               "display": {"first_iteration": RL_START, "last_iteration": RL_STOP,
                           "smoothing_population": "Only evaluations in the displayed 20--60 interval; no pre-20 warmup"},
               "methods": [{"id": m["id"], "points": [p for p in m["points"]
                            if RL_START <= p["iteration"] <= RL_STOP],
                            "smoothed": [p for p in m["smoothed"]
                                         if RL_START <= p["iteration"] <= RL_STOP]}
                           for m in SOURCE["online"]["methods"]]},
}
(ROOT / "figure1_results.json").write_text(json.dumps(result, indent=2) + "\n")

metadata = {"CreationDate": None, "ModDate": None, "Creator": "plot_figure1.py"}
fig.savefig(ROOT / "figure1.pdf", dpi=300, metadata=metadata)
svg_path = ROOT / "figure1.svg"
fig.savefig(svg_path, dpi=300, metadata={"Date": None, "Creator": "plot_figure1.py"})
# Matplotlib emits trailing spaces in multi-line SVG paths.
svg_path.write_text("\n".join(s.rstrip() for s in svg_path.read_text().splitlines()) + "\n")
fig.savefig(ROOT / "figure1.png", dpi=300)
plt.close(fig)
print("Wrote fig/figure1.pdf, fig/figure1.svg, fig/figure1.png")

# Keep the existing counterclockwise previews in sync with the editable source.
# Rotation preserves the normal figure's values, geometry and vector content.
with fitz.open(ROOT / "figure1.pdf") as rotated:
    for page in rotated:
        page.set_rotation((page.rotation - 90) % 360)
    rotated.save(ROOT / "figure1-ccw.pdf", garbage=4, deflate=True)
    rotated[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False).save(
        ROOT / "figure1-ccw.png")

svg_ns = "http://www.w3.org/2000/svg"
for prefix, uri in [("", svg_ns),
                    ("xlink", "http://www.w3.org/1999/xlink"),
                    ("dc", "http://purl.org/dc/elements/1.1/"),
                    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")]:
    ET.register_namespace(prefix, uri)
original_svg = ET.parse(svg_path).getroot()
_, _, width, height = map(float, original_svg.attrib["viewBox"].split())
attrs = dict(original_svg.attrib)
attrs.update(width=f"{height:g}pt", height=f"{width:g}pt",
             viewBox=f"0 0 {height:g} {width:g}")
rotated_svg = ET.Element(f"{{{svg_ns}}}svg", attrs)
group = ET.SubElement(rotated_svg, f"{{{svg_ns}}}g",
                      transform=f"translate(0 {width:g}) rotate(-90)")
for child in original_svg:
    group.append(child)
rotated_svg_path = ROOT / "figure1-ccw.svg"
ET.ElementTree(rotated_svg).write(rotated_svg_path, encoding="utf-8",
                                xml_declaration=True)
rotated_svg_path.write_text("\n".join(line.rstrip() for line in
                                   rotated_svg_path.read_text().splitlines()) + "\n")
print("Wrote fig/figure1-ccw.pdf, fig/figure1-ccw.svg, fig/figure1-ccw.png")
