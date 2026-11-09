"""Shared compact paper styling; preserves every plotted coordinate and axis range."""
# Shared semantic palette; source measurements and geometry are unchanged.
from pathlib import Path as _PalettePath
import sys as _palette_sys
_palette_sys.path.insert(0, str(next(p for p in _PalettePath(__file__).resolve().parents
                                     if (p / "paper_plot_palette.py").is_file())))
from paper_plot_palette import COLORS as PALETTE, tint as palette_tint
FOCUS = "parallel5_select_high_60steps"
LONG = {"luna_single_high_60steps", "luna_single_high_120steps", FOCUS, "sol_single_high60"}
FOOTER = "GPT-6 Luna & Sol. Luna baseline: high effort · 12,288 max tokens · 30 steps. N=candidates."


def display_label(label, method):
    if method == FOCUS:
        return "Luna ADM\nN=5, 60 steps"
    return label.replace("Luna 0.5× max tokens", "0.5× max tokens").replace("Luna 2× max tokens", "2× max tokens")


def apply_style(fig, methods_per_panel, styles, label_to_method, *, weighted):
    assert len(fig.axes) == len(methods_per_panel)
    for j, (ax, methods) in enumerate(zip(fig.axes, methods_per_panel)):
        assert len(ax.collections) == len(methods) + 1  # method points plus frontier rings
        for method, collection in zip(methods, ax.collections[:len(methods)]):
            color = styles[method][2]
            collection.set_facecolor(color if method in LONG else "white")
            collection.set_edgecolor(color)
            collection.set_sizes([27])
            collection.set_linewidths([.9])
            if method == FOCUS:
                collection.set_zorder(5)
                offset, align = ((7, 5), "left") if j % 3 == 1 else ((-6, 5), "right")
                if weighted or j % 3 == 2:
                    ax.annotate("Luna ADM", collection.get_offsets()[0], xytext=offset,
                                textcoords="offset points", color=color, fontsize=9,
                                ha=align, va="bottom", zorder=6)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=9)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        if weighted or j >= 6:
            ax.xaxis.label.set_text(("Actor cost ($, log)", "Browser steps", "Actor latency (s)")[j % 3])
        if not weighted:
            ax.title.set_fontsize(10)
            for text in ax.texts:
                if text.get_transform() == ax.transAxes:
                    text.set_fontsize(9)
                    text.set_y(1.45)
                    text.set_text(text.get_text().replace("Browser-code batches", "Browser steps"))
    if weighted:
        fig.set_size_inches(7, 2.55)
        fig.subplots_adjust(left=.085, right=.985, top=.97, bottom=.53, wspace=.12)
        fig.axes[0].yaxis.label.set_text("Valid-only success (%)")
        fig.axes[0].yaxis.set_label_coords(-.17, .40)
    else:
        fig.set_size_inches(7, 4.3)
        fig.subplots_adjust(left=.085, right=.985, top=.89, bottom=.32,
                            hspace=.60, wspace=.12)
    old_legends = list(fig.legends)
    for legend in old_legends:
        legend.remove()
    for j, old in enumerate(old_legends[:2]):
        labels = []
        for line, text in zip(old.get_lines(), old.texts):
            method = label_to_method[text.get_text()]
            line.set_markerfacecolor(styles[method][2] if method in LONG else "white")
            line.set_markersize(4.8)
            line.set_markeredgewidth(.9)
            labels.append(display_label(text.get_text(), method))
        title = old.get_title().get_text() if j == 0 else "Best-of-N action decision scaling"
        fig.legend(handles=old.get_lines(), labels=labels, title=title, loc="upper left",
                   bbox_to_anchor=(.085 if j == 0 else .65, .93 / fig.get_figheight()),
                   ncol=3 if j == 0 else 2, fontsize=8, title_fontsize=9,
                   frameon=False, handlelength=.8, handletextpad=.4,
                   columnspacing=.7, labelspacing=.3, borderaxespad=0, alignment="left")
    old = old_legends[2]
    fig.legend(handles=old.get_lines(), labels=[text.get_text() for text in old.texts],
               loc="lower left", bbox_to_anchor=(.085, .04 / fig.get_figheight()),
               frameon=False, fontsize=8, handlelength=1.8, handletextpad=.35, borderaxespad=0)
    for text in list(fig.texts):
        text.remove()
    fig.text(.32, .065 / fig.get_figheight(), FOOTER, fontsize=7.2, color=PALETTE["MUTED"], va="bottom")
