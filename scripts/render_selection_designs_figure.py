#!/usr/bin/env python3
"""Render the selector-design comparison figure (Piotr ARM, actor-feature heads, Kev-style variants, Kev).

Synthetic method illustration; contains no task data. Writes SVG, and PNG with --png (fails if any
text leaves its box or the canvas). Reuses the drawing helpers of render_selection_head_figure.py.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_selection_head_figure import DEST, INK, MUTED, Figure, render_png  # noqa: E402

COLUMNS = [
    ('Piotr SelectionARM', 'baseline', 'plain'),
    ('One-at-a-time head', 'frozen actor (+ top-8 adapter)', 'frozen'),
    ('Joint top layers', 'Kev-inspired, inside the actor', 'train'),
    ('Kev-style pass', 'Kev design on the actor\'s weights', 'train'),
    ('Kev', 'external reference', 'plain'),
]
ROWS = [
    ('Model', [['Separate 4B', 'vision-language selector'], ['Frozen 4B actor', '+ small head; adapter:', 'LoRA on top 8 layers'],
               ['4B actor, layers 1–28 frozen', '+ LoRA on top 8 layers'], ['4B actor + LoRA on all', '36 layers, as a separate pass'],
               ['Separate 0.8–27B', 'decision model']]),
    ('Reads', [['screenshot + task +', '5 candidate texts'], ["actor's prompt + screenshot;", 'its own candidate states'],
               ['same as left; candidate', 'states from generation'], ['prompt + screenshot +', 'candidates, re-encoded'],
               ['text-only page JSON +', 'candidate texts']]),
    ('Interaction', None),
    ('Order effects', [['yes: picks candidate 1', 'on 62/124 states'], ['none'], ['none: shared positions'],
                       ['none: shared positions'], ['yes in Kev-27B;', 'we shuffle per state']]),
    ('Readout', [['generates', '{"selection": N}'], ['pooled states', '→ MLP per candidate'],
                 ['pooled head', '+ gated pointer'], ['pointer only:', 'q(<decide>) · k(end)'],
                 ['pointer', '+ fitted temperature']]),
    ('Added cost', [['full 4B pass:', '≈3.8k tokens + image'], ['≈0 (frozen head);', '≈8/36 pass (adapter)'],
                    ['≈8/36 pass over', 'candidate tokens'], ['≈ one full 4B pass', '(prompt + candidates)'],
                    ['full 27B pass over', 'page text + candidates']]),
    ('Status', [['Luna 73.2%; held-out', 'success +2.5 pp'], ['Luna 61.7%; held-out', 'success +2.0 pp'],
                ['pointer alone: Luna', 'agreement 57.5%'], ['Luna 66.1% (step 600);', 'held-out success +1.1 pp'],
                ['not run on these states']]),
]
# Attention among [cand 1, cand 2, cand 3, <decide>]: rows = queries, columns = keys.
MASKS = [
    ([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], False, ['one causal sequence:', 'later see earlier']),
    ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], False, ['none: each', 'candidate alone']),
    ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]], True, ['<decide> reads all,', 'top 8 layers only']),
    ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]], True, ['<decide> reads all,', 'in every layer']),
    ([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], True, ['later options see', 'earlier + <decide>']),
]
FILL = {'frozen': '#2563b0', 'train': '#ad5b13', 'plain': '#7d8fa3'}


def mask_glyph(f, x, y, cells, decide, color):
    size, labels = 22, ['c1', 'c2', 'c3', 'D']
    n = 4 if decide else 3
    for i in range(n):
        f.text(x - 8, y + i * size + 16, labels[i], 12, MUTED, 400, 'end')
        f.text(x + i * size + size / 2, y - 6, labels[i], 12, MUTED, 400, 'middle')
        for j in range(n):
            on = cells[i][j]
            f.parts.append(f'<rect x="{x + j * size}" y="{y + i * size}" width="{size - 2}" height="{size - 2}" rx="3" '
                           f'fill="{color if on else "#eef2f7"}" stroke="none"/>')


def figure():
    col_w, label_w, x0, top = 300, 150, 30, 140
    width = x0 * 2 + label_w + col_w * len(COLUMNS)
    heights = [86, 86, 178, 74, 74, 74, 74]
    height = top + 84 + sum(heights) + 20 * len(heights) + 30
    f = Figure(width, height, 'Five ways to choose among sampled actions',
               'Compares Piotr SelectionARM, the one-at-a-time actor-feature head, the joint top-layer variant, '
               'a Kev-style decision pass on the actor and Kev itself: weights, inputs, how candidates interact, '
               'order effects, readout, added cost and current status (Luna agreement on 124 states; held-out continuation success minus uniform on 138 states).')
    f.text(x0, 52, 'Five ways to choose among sampled actions', 34, INK, 700)
    f.text(x0, 86, 'From a separate selector to scoring inside the actor; the small grids show which candidates '
                   'each one can attend to.', 19, MUTED)
    f.text(x0, 116, 'Grid rows are queries, columns are keys: c1–c3 candidates, D the <decide> token; '
                    'filled = may attend.', 15, MUTED)
    y = top
    for c, (name, sub, kind) in enumerate(COLUMNS):
        x = x0 + label_w + c * col_w
        f.box(x + 6, y, col_w - 12, 72, kind, name, [sub], title_size=19, size=14)
    y += 84
    for (label, cells), h in zip(ROWS, heights):
        f.text(x0, y + h / 2 + 6, label, 17, INK, 700)
        for c, (_, _, kind) in enumerate(COLUMNS):
            x = x0 + label_w + c * col_w
            if cells is None:
                grid, decide, caption = MASKS[c]
                f.box(x + 6, y, col_w - 12, h, 'panel', radius=10)
                n = 4 if decide else 3
                mask_glyph(f, x + col_w / 2 - n * 11 + 6, y + 30, grid, decide, FILL[kind])
                for k, line in enumerate(caption):
                    f.text(x + col_w / 2, y + 138 + k * 19, line, 14, MUTED, 400, 'middle', box=(x + 6, y, col_w - 12, h))
            else:
                f.box(x + 6, y, col_w - 12, h, 'plain', None, cells[c], size=14)
        y += h + 20
    return f.write('selection_designs_compared')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    path = figure()
    if args.png:
        render_png(path)
    print(path)


if __name__ == '__main__':
    main()
