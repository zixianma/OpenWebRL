#!/usr/bin/env python3
"""Render public, synthetic ARM method illustrations; contains no task data."""
import argparse
from html import escape
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'openwebrl/docs/arm_results/methods'
INK, MUTED, LINE = '#182b44', '#50647a', '#d6dfeb'
BLUE, TEAL, PURPLE, ORANGE = '#2563b0', '#14776c', '#7350a2', '#ad5b13'


class Figure:
    def __init__(self, width, height, title, description):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<style>text{font-family:Arial,Helvetica,sans-serif} .mono{font-family:monospace}</style>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#7d8fa3"/></marker></defs>']
        self.rect(0,0,width,height,'#fff',radius=0)

    def rect(self,x,y,w,h,fill='#fff',stroke='none',radius=14):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def text(self,x,y,value,size=22,color=INK,weight=400,anchor='start',mono=False):
        cls = ' class="mono"' if mono else ''
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{cls}>{escape(value)}</text>')

    def arrow(self,x1,y1,x2,y2):
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#7d8fa3" stroke-width="2.5" marker-end="url(#arrow)"/>')

    def write(self,name):
        path=DEST/(name+'.svg')
        path.write_text('\n'.join(self.parts+['</svg>'])+'\n')
        return path


def stages():
    f=Figure(1500,590,'Three ways to use an action reward model',
        'Inference uses ARM to choose an action now. Offline learning distills saved ARM preferences. Online RL uses ARM to change turn credit during PPO. Only inference-time selection requires ARM at deployment.')
    f.text(42,53,'Three ways to use ARM',36,weight=700)
    f.text(42,88,'Same signal: which reasoning + action response does ARM prefer at this browser state?',22,MUTED)
    cards=[(42,BLUE,'1  INFERENCE','Choose the action now',
        [('Actor samples 5 responses','at the current state'),('ARM selects one response','ScalarRM scores; SelectionARM compares'),('Execute the selected action','repeat at every browser turn')],
        'Deployment: actor + ARM'),
        (532,PURPLE,'2  OFFLINE LEARNING','Learn from saved preferences',
        [('Saved states + preferences','retain eligible training examples'),('Filtered SFT or DPO','imitate chosen / prefer chosen to rejected'),('Train the actor offline','C2, 1A, joint SFT and joint DPO')],
        'Deployment: single trained actor'),
        (1022,TEAL,'3  ONLINE RL','Change credit during learning',
        [('Actor collects fresh rollouts','terminal judge provides task outcome'),('ARM labels sampled turns','executed response + 4 alternatives'),('PPO uses outcome + ARM signal','add a bonus or reweight turn credit')],
        'Deployment: single trained actor')]
    for x,color,label,subtitle,boxes,footer in cards:
        f.rect(x,117,436,432,'#f8fafc',LINE)
        f.text(x+20,152,label,21,color,700)
        f.text(x+20,184,subtitle,24,weight=700)
        for i,(a,b) in enumerate(boxes):
            y=207+i*94
            f.rect(x+18,y,400,72,'#fff',LINE,9)
            f.text(x+218,y+29,a,21,weight=700,anchor='middle')
            f.text(x+218,y+53,b,16,MUTED,anchor='middle')
            if i<2:f.arrow(x+218,y+76,x+218,y+89)
        f.text(x+218,529,footer,20,color,700,anchor='middle')
    f.text(42,578,'ARM judges local preference; the terminal outcome judge determines task success.',19,MUTED)
    return f.write('arm_three_stages')


def rl_choices():
    f=Figure(1600,930,'ARM RL variants: groups, gates and credit',
        'Mixed groups contain varied terminal outcomes; failure groups contain five valid failures. All-failure replaces mixed slots, additive adds up to eight failure groups. Gate B accepts duplicate-containing sets; Gate C also credits action equivalence. Worked examples are synthetic.')
    f.text(40,50,'Online RL: groups, gates and credit',36,weight=700)
    f.text(40,86,'Two different sets of five: 5 trajectories per task  ≠  5 candidate responses at one state.',23,MUTED)

    f.rect(40,112,726,420,'#f8fafc',LINE)
    f.text(62,149,'1  GROUPS — which trajectories train?',25,BLUE,700)
    f.text(62,184,'M = mixed outcomes [1, 0, 1, 0, 0]',21,BLUE)
    f.text(62,215,'F = five valid failures [0, 0, 0, 0, 0]',21,ORANGE)
    rows=[('Outcome-only baseline','48M','terminal outcome only'),
          ('Original / mixed-only pair','48M','ARM on ordinary groups'),
          ('All-failure bonus','(48−N)M + NF','failure groups replace slots'),
          ('Additive / Gate B / Gate C','48M + NF','add N ≤ 8 failure groups')]
    for i,(name,formula,note) in enumerate(rows):
        y=249+i*64
        f.text(62,y,name,20,weight=700)
        f.text(420,y,formula,22,BLUE,700)
        f.text(420,y+25,note,17,MUTED)
    f.text(62,516,'F is eligible only with ≥1 usable ARM label; A = 0 for F.',18,MUTED)

    f.rect(792,112,768,744,'#f8fafc',LINE)
    f.text(816,149,'2  GATE — is this candidate set usable?',25,TEAL,700)
    f.text(816,184,'All five responses must be valid. Letters denote actions.',21,MUTED)
    for i,letter in enumerate('XXYZW'):
        x=824+i*141
        color=BLUE if i==0 else TEAL if i==1 else MUTED
        f.text(x+53,223,f'response {i}',18,color,anchor='middle')
        f.rect(x,238,106,79,'#eaf2fd' if i==0 else '#e6f5ef' if i==1 else '#fff',color)
        f.text(x+53,290,letter,39,color,700,anchor='middle')
        if i<2:f.text(x+53,346,'executed' if i==0 else 'ARM chose',18,color,700,anchor='middle')
    f.text(816,382,'The two X responses can have different reasoning.',19,MUTED)
    f.text(816,420,'Strict gate: 5 distinct actions required → reject (bonus 0)',21,ORANGE)
    f.text(816,456,'Gate B / C: ≥2 distinct actions required → accept',21,TEAL,700)
    f.text(816,510,'3  CREDIT — what counts as an ARM win?',25,PURPLE,700)
    f.rect(816,532,720,119,'#fff',LINE)
    f.text(838,567,'Gate B: exact response index',23,weight=700)
    f.text(838,602,'ARM chose response 1, not the executed response 0.',20,MUTED)
    f.text(838,631,'b = 0.5 × (0 − 1/5) = −0.10',23,ORANGE,700)
    f.rect(816,668,720,150,'#fff',LINE)
    f.text(838,703,'Gate C: equivalent action',23,weight=700)
    f.text(838,738,'Selected action X matches executed action X.',20,MUTED)
    f.text(838,768,'Two of five candidates contain action X.',20,MUTED)
    f.text(838,798,'b = 0.5 × (1 − 2/5) = +0.30',23,TEAL,700)

    f.rect(40,552,726,304,'#f8fafc',LINE)
    f.text(62,589,'BONUS vs REWEIGHT — same 48M + Gate B',24,PURPLE,700)
    f.text(62,624,'Example: A = +1; three turns; valid binary outcomes.',20,MUTED)
    f.text(62,657,'ARM: executed wins / alternative wins / no label',19,MUTED)
    f.text(62,699,'Outcome only',21,weight=700)
    f.text(333,699,'[1.00, 1.00, 1.00]',22,mono=True)
    f.text(62,733,'Add bonus (β = 0.5)',21,weight=700)
    f.text(333,733,'[1.40, 0.90, 1.00]',22,mono=True)
    v=[math.exp(.5*u) for u in (.8,-.2,0.)]
    weights=[x/(sum(v)/3) for x in v]
    assert [round(x,2) for x in weights]==[1.32,.8,.88]
    f.text(62,767,'Reweight (λ = 0.5)',21,weight=700)
    f.text(333,767,'['+', '.join(f'{x:.2f}' for x in weights)+']',22,mono=True)
    f.text(62,813,'Reweight preserves the trajectory’s mean A and its sign.',19,MUTED)
    f.text(40,891,'Shared: K = 5, q = 20% label attempts, β = 0.5. ARM credit covers executed reasoning + action tokens.',21,MUTED)
    f.text(40,920,'These are method illustrations, not recorded trajectories. Exact formulas and error handling are in ARM_SUMMARY.md.',18,MUTED)
    return f.write('arm_rl_method_choices')


def render_png(paths):
    from playwright.sync_api import sync_playwright
    import xml.etree.ElementTree as ET
    browser_path='/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome'
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=browser_path,headless=True,args=['--no-sandbox'])
        for path in paths:
            tree=ET.parse(path).getroot()
            page=browser.new_page(viewport=dict(width=int(tree.attrib['width']),height=int(tree.attrib['height'])),device_scale_factor=2)
            page.set_content('<html><body style="margin:0">'+path.read_text()+'</body></html>')
            # SVG text is measurable; reject overflow outside the figure.
            overflow=page.locator('svg').evaluate("""svg => [...svg.querySelectorAll('text')].filter(t => {
                let b=t.getBBox();return b.x<0||b.y<0||b.x+b.width>svg.viewBox.baseVal.width||b.y+b.height>svg.viewBox.baseVal.height;
            }).map(t=>t.textContent)""")
            if overflow:raise ValueError('Figure text overflow: '+repr(overflow))
            page.locator('svg').screenshot(path=str(path.with_suffix('.png')))
            page.close()
        browser.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png',action='store_true')
    args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    paths=[stages(),rl_choices()]
    if args.png:render_png(paths)
    for path in paths:print(path)


if __name__=='__main__':main()
