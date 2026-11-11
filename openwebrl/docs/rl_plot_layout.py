"""Shared compact square-panel layout for RL learning and repetition figures."""
import matplotlib.pyplot as plt
from paper_plot_palette import COLORS as C

FIGSIZE = (4.9, 2.10)
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 8, 7.5, 7, 7
LEGEND_LABELS = ("Outcome only\n48 groups", "Outcome only\n56 groups", "Outcome + ADM\n(ours)")


def make_axes():
    plt.rcParams.update({"font.family": "Albert Sans", "font.size": TICK_SIZE,
                         "text.color": C["INK"], "axes.labelcolor": C["MUTED"],
                         "xtick.color": C["MUTED"], "ytick.color": C["MUTED"]})
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=.09, right=.72, top=.84, bottom=.22, wspace=.35)
    for ax in axes:
        ax.set_box_aspect(1)
    return fig, axes


def style_axis(ax, title):
    ax.set_title(title, fontsize=TITLE_SIZE, loc="left", color=C["INK"], pad=6)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["BORDER"])
        ax.spines[side].set_linewidth(.7)
    ax.tick_params(labelsize=TICK_SIZE, colors=C["MUTED"], width=.7, length=2.5, pad=2)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontsize(LABEL_SIZE)
        label.set_color(C["MUTED"])


def add_legend(fig, handles):
    return fig.legend(handles=handles, labels=LEGEND_LABELS, loc="center left",
                      bbox_to_anchor=(.76, .53), ncol=1, frameon=False,
                      fontsize=LEGEND_SIZE, labelcolor=C["INK"], handlelength=1.4,
                      handletextpad=.5, labelspacing=1.0, borderaxespad=0)
