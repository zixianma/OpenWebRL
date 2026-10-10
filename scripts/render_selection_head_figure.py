#!/usr/bin/env python3
"""Render the selection-head architecture figure (inference and training workflows).

Synthetic method illustration; contains no task data. Writes SVG, and PNG with --png.
The PNG pass fails if any text leaves its box or the canvas.
"""
import argparse
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'openwebrl/docs/arm_results/methods'
INK, MUTED, LINE, ARROW = '#182b44', '#50647a', '#d6dfeb', '#7d8fa3'
STYLES = {  # fill, stroke, dashed
    'frozen': ('#eaf1fb', '#2563b0', False),
    'train': ('#fdf0e4', '#ad5b13', False),
    'cache': ('#eef8f6', '#14776c', True),
    'data': ('#f2edf8', '#7350a2', False),
    'plain': ('#f8fafc', '#b7c4d4', False),
    'panel': ('#ffffff', LINE, False),
}


class Figure:
    def __init__(self, width, height, title, description):
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<style>text{font-family:Arial,Helvetica,sans-serif}</style>',
            f'<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
            f'<path d="M0,0 L8,4 L0,8" fill="{ARROW}"/></marker></defs>',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>']

    def text(self, x, y, value, size=15, color=INK, weight=400, anchor='start', box=None):
        attr = f' data-box="{",".join(map(str, box))}"' if box else ''
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
                          f'text-anchor="{anchor}"{attr}>{escape(value)}</text>')

    def box(self, x, y, w, h, kind, title=None, lines=(), title_size=17, size=14, center=True, radius=10):
        fill, stroke, dashed = STYLES[kind]
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="2"{dash}/>')
        rows = ([(title, title_size, 700, INK)] if title else []) + [(l, size, 400, MUTED) for l in lines]
        heights = [s * 1.35 for _, s, _, _ in rows]
        cy = y + (h - sum(heights)) / 2
        for (value, s, weight, color), lh in zip(rows, heights):
            cy += lh
            tx, anchor = (x + w / 2, 'middle') if center else (x + 14, 'start')
            self.text(tx, cy - lh * 0.28, value, s, color, weight, anchor, box=(x, y, w, h))

    def arrow(self, points, label=None, dashed=False, label_at=None):
        d = 'M' + ' L'.join(f'{px},{py}' for px, py in points)
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="2.5"{dash} marker-end="url(#arrow)"/>')
        if label:
            lx, ly = label_at or ((points[0][0] + points[-1][0]) / 2, (points[0][1] + points[-1][1]) / 2 - 8)
            self.text(lx, ly, label, 13, MUTED, 400, 'middle')

    def write(self, name):
        path = DEST / (name + '.svg')
        path.write_text('\n'.join(self.parts + ['</svg>']) + '\n')
        return path


def figure():
    f = Figure(1600, 1190, 'Selection head inside the actor: inference and training workflows',
               'Inference: the frozen SFT actor samples five candidates; a small head scores them from the actor\'s own '
               'hidden states, read directly or after a top-layer scoring adapter, instead of a separate 4B selector. '
               'Training: stage 1 trains the head on cached frozen features; stage 2 trains LoRA on the top eight '
               'language layers plus the head, with the prompt encoded by the unadapted actor.')
    f.text(40, 52, 'Selection head inside the actor', 34, INK, 700)
    f.text(40, 86, 'Generation always uses the frozen SFT actor; selection reads that actor\'s hidden states instead of '
                   'running a second 4B selector.', 19, MUTED)
    legend = [('frozen', 'Frozen SFT actor'), ('train', 'Learned / trainable'), ('cache', 'Reused / cached'),
              ('data', 'Teacher data'), ('plain', 'Other step')]
    x = 40
    for kind, label in legend:
        f.box(x, 106, 26, 20, kind, radius=4)
        f.text(x + 36, 122, label, 15, INK)
        x += 60 + len(label) * 8

    # Panel A: inference.
    f.box(24, 142, 1552, 438, 'panel', radius=14)
    f.text(46, 176, 'A   INFERENCE · one browser step', 19, '#2563b0', 700)
    top, h = 196, 196
    f.box(46, top, 196, h, 'plain', 'Browser state', ['task', 'action history', 'current screenshot'])
    f.box(272, top, 268, h, 'frozen', 'Frozen SFT actor', ['Qwen3-VL-4B, 36 layers', '1. prefill prompt once', '→ prompt KV cache',
                                                           '2. sample 5 candidates', '(reasoning + action)'])
    f.box(570, top, 236, h, 'cache', 'Hidden states', ['of all 5 candidates', 'by-product of generation', 'every token, every layer', 'no extra forward pass'])
    f.box(836, top, 280, 90, 'frozen', 'Frozen head', ['read layers 18 / 27 / 36 as is'])
    f.box(836, top + 106, 280, 90, 'train', 'Scoring adapter', ['re-run top 8 layers on candidate', 'tokens with LoRA; prompt KV reused'])
    f.box(1146, top, 200, h, 'plain', 'Pool per candidate', ['reasoning mean', 'action mean', 'end token', '+ prompt last token'])
    f.box(1376, top, 178, h, 'train', 'Selection head', ['standardize', '→ MLP per candidate', '→ 5 scores', '→ argmax', '→ execute'])
    mid = top + h / 2
    f.arrow([(242, mid), (268, mid)])
    f.arrow([(540, mid), (566, mid)])
    f.arrow([(806, mid), (820, mid), (820, top + 45), (832, top + 45)])
    f.arrow([(806, mid), (820, mid), (820, top + 151), (832, top + 151)])
    f.arrow([(1116, top + 45), (1131, top + 45), (1131, mid), (1142, mid)])
    f.arrow([(1116, top + 151), (1131, top + 151), (1131, mid), (1142, mid)])
    f.arrow([(1346, mid), (1372, mid)])
    f.box(46, 418, 760, 140, 'plain', 'Added cost per step', [
        'Frozen head: no extra 4B forward — features come from generation',
        'Adapter: ≈ 8/36 of one forward over the candidate tokens only',
        'One set of 4B weights to serve'], center=False)
    f.box(836, 418, 718, 140, 'plain', 'Baseline: Piotr SelectionARM', [
        'A separate 4B model re-reads the screenshot and all 5 candidates',
        '(≈ 3.8k input tokens per step) and outputs {"selection": N}',
        'Two sets of 4B weights to serve'], center=False)

    # Panel B: training.
    f.box(24, 604, 1552, 568, 'panel', radius=14)
    f.text(46, 638, 'B   TRAINING · teacher-labeled panels; the generating actor is never updated', 19, '#ad5b13', 700)
    f.box(46, 664, 250, 190, 'data', 'Teacher-labeled panels', ['Piotr release, GPT-5.5 choice', 'prompt + screenshot',
                                                                '5 sampled candidates', '35,916 train · 3,759 held-out', '(split by task group)'])
    f.box(46, 878, 250, 130, 'data', 'Soft target', ['uniform over candidates', 'action-equivalent to', "the teacher's pick"])

    f.text(330, 668, 'Stage 1 · frozen features', 16, INK, 700)
    s1 = 682
    f.box(330, s1, 270, 120, 'frozen', 'Frozen actor', ['teacher-forced', 'prompt once → KV cache', '→ 5 candidates per panel'])
    f.box(630, s1, 240, 120, 'cache', 'Feature cache (once)', ['39,675 panels · 12 GB', '≈ 2 GPU-hours'])
    f.box(900, s1, 250, 120, 'train', 'Train head only', ['soft cross-entropy', '≈ 3 s per epoch'])
    f.box(1180, s1, 374, 120, 'plain', 'Choose checkpoint', ['held-out GPT-5.5 agreement', '≈ 65.5% (most common action 50.1%)'])
    f.arrow([(296, 742), (326, 742)])
    f.arrow([(600, s1 + 60), (626, s1 + 60)])
    f.arrow([(870, s1 + 60), (896, s1 + 60)])
    f.arrow([(1150, s1 + 60), (1176, s1 + 60)])

    f.text(330, 846, 'Stage 2 · top-layer scoring adapter (warm-started from stage 1)', 16, INK, 700)
    s2 = 860
    f.box(330, s2, 196, 130, 'cache', 'Prompt', ['adapter off, no gradient', '(as at generation)', '→ KV cache'])
    f.box(556, s2, 196, 130, 'frozen', 'Layers 1–28', ['candidate tokens', 'frozen'])
    f.box(782, s2, 216, 130, 'train', 'Layers 29–36', ['+ LoRA, rank 16', 'trainable'])
    f.box(1028, s2, 150, 130, 'plain', 'Pool', ['per candidate'])
    f.box(1208, s2, 170, 130, 'train', 'Head', ['trainable', 'starts at stage 1'])
    f.box(1408, s2, 146, 130, 'plain', 'Soft CE loss', ['vs soft target'])
    m2 = s2 + 65
    f.arrow([(296, 810), (313, 810), (313, m2), (326, m2)])
    f.arrow([(526, m2), (552, m2)])
    f.arrow([(752, m2), (778, m2)])
    f.arrow([(998, m2), (1024, m2)])
    f.arrow([(1178, m2), (1204, m2)])
    f.arrow([(1378, m2), (1404, m2)])
    f.text(330, 1022, 'Gradients reach only the LoRA weights and the head; the vision tower, base weights and generation '
                      'are unchanged.', 15, MUTED)
    f.box(330, 1046, 1224, 104, 'plain', 'Evaluation — never used for training or checkpoint choice', [
        '124 branching states: agreement with Luna (independent teacher) and chosen-action continuation success,',
        'compared with Piotr SelectionARM run offline on the same states'])
    return f.write('selection_head_workflows')


def render_png(path):
    from playwright.sync_api import sync_playwright
    import xml.etree.ElementTree as ET
    chrome = '/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome'
    root = ET.parse(path).getroot()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport=dict(width=int(root.attrib['width']), height=int(root.attrib['height'])),
                                device_scale_factor=2)
        page.set_content('<html><body style="margin:0">' + path.read_text() + '</body></html>')
        bad = page.locator('svg').evaluate("""svg => [...svg.querySelectorAll('text')].filter(t => {
            const b = t.getBBox(), vb = svg.viewBox.baseVal;
            if (b.x < 0 || b.y < 0 || b.x + b.width > vb.width || b.y + b.height > vb.height) return true;
            if (!t.dataset.box) return false;
            const [x, y, w, h] = t.dataset.box.split(',').map(Number);
            return b.x < x + 4 || b.x + b.width > x + w - 4 || b.y < y + 2 || b.y + b.height > y + h - 2;
        }).map(t => t.textContent)""")
        if bad:
            raise ValueError('Text outside its box or the canvas: ' + repr(bad))
        page.locator('svg').screenshot(path=str(path.with_suffix('.png')))
        browser.close()


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
