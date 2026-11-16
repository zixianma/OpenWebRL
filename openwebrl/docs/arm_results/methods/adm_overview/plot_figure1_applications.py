"""Application-first alternative to Figure 1; leaves the current figure intact.

Run: MPLCONFIGDIR=/tmp/adm-variant-mpl python fig/plot_figure1_applications.py
Defaults to the selected oversized vivid-violet title.
Add --bordered to export a separate figure1_applications_bordered comparison.
Use --header-style with --preview-only to compare title treatments separately.
Add --top-background to export a separate pale-lavender overview variant.
Add --training-overview to make the top panel tell only the training story.
Add --phase-overview to label ADM training and applications, retaining predictions.
Add --refined-overview for an aligned comparison with vertical teacher supervision.
Add --redesigned-overview for horizontal teacher/student rows, their loss, and application bands.
Add --minimal-overview for compact training with all application details retained.
Add --simplify-layout for a separate less crowded student-first comparison.
Add --no-phase-headers to omit Training/Applications while retaining application titles.
Use --conceptual-overview for the new Figure 1: training/model and three workflows, without result plots.
Use --minimal-concept for an abstract training diagram and the same application workflows.
Uses the current Figure 1 aggregate snapshot without collecting new results.
The header uses the paper's Netflix example; future observations are summarized.
Exports editable vector PDF/SVG, a PNG preview and a provenance JSON.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
_style_root = next((parent for parent in Path(__file__).resolve().parents
                    if (parent / 'paper_plot_style.py').is_file()), None)
if _style_root is not None:
    sys.path.insert(0, str(_style_root))
    import paper_plot_style

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as VectorPath
import numpy as np

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--bordered",action="store_true",
                    help="Export the fully bordered comparison without changing the current figure")
parser.add_argument("--header-style",choices=("current","plain","band","tab","badge","large","underline","integrated"),default="current",
                    help="Export a separate title treatment; current uses the selected oversized title")
parser.add_argument("--preview-only",action="store_true",
                    help="Export only a PNG preview")
parser.add_argument("--top-background",action="store_true",
                    help="Export a separate pale-lavender top-panel variant")
parser.add_argument("--training-overview",action="store_true",
                    help="Export a separate training-focused overview without a top action output")
parser.add_argument("--phase-overview",action="store_true",
                    help="Export separate training/application sections with explicit prediction and supervision")
parser.add_argument("--refined-overview",action="store_true",
                    help="Export a new aligned training/application layout, preserving earlier versions")
parser.add_argument("--redesigned-overview",action="store_true",
                    help="Export horizontal teacher/student rows with explicit lookahead and loss")
parser.add_argument("--minimal-overview",action="store_true",
                    help="Export compact conceptual training with the complete application workflows and plots")
parser.add_argument("--swap-teacher-student",action="store_true",
                    help="Export a separate student-first redesigned preview")
parser.add_argument("--simplify-layout",action="store_true",
                    help="Export a separate student-first comparison with a compact CE loss and fewer containers")
parser.add_argument("--phase-bands",action="store_true",
                    help="Export a simplified comparison with neutral Training/Applications heading bands")
parser.add_argument("--no-results",action="store_true",
                    help="Export the banded training and application illustrations without quantitative plots")
parser.add_argument("--conceptual-overview",action="store_true",
                    help="Create figure1_conceptual with training/model and three application workflows, without result plots or phase headings")
parser.add_argument("--minimal-concept",action="store_true",
                    help="Create figure1_conceptual_minimal with shared inputs, teacher lookahead, action-label supervision and all three workflows")
parser.add_argument("--right-legend",action="store_true",
                    help="Export a separate comparison with a method legend beside the inference plots")
parser.add_argument("--no-phase-headers","--no-panel-headers",dest="no_panel_headers",action="store_true",
                    help="Omit Training/Applications phase headings while retaining the (a), (b), (c) application titles")
render_options = parser.parse_args()
if render_options.minimal_concept:
    render_options.conceptual_overview = True
if render_options.conceptual_overview:
    render_options.no_results = True
    render_options.no_panel_headers = True
if render_options.right_legend:
    render_options.phase_bands = True
if render_options.no_results:
    render_options.phase_bands = True
if render_options.phase_bands:
    render_options.simplify_layout = True
if render_options.simplify_layout:
    render_options.swap_teacher_student = True
if render_options.no_panel_headers and not render_options.swap_teacher_student:
    parser.error("--no-panel-headers requires --simplify-layout, --phase-bands, --no-results, or --right-legend")
if render_options.swap_teacher_student and render_options.minimal_overview:
    parser.error("Choose the student-first layout or the minimal training layout")
if render_options.swap_teacher_student:
    render_options.redesigned_overview = True
    render_options.bordered = True
if render_options.minimal_overview:
    render_options.redesigned_overview = True
    render_options.bordered = True
if render_options.refined_overview and render_options.redesigned_overview:
    parser.error("Choose either --refined-overview or --redesigned-overview")
if render_options.refined_overview or render_options.redesigned_overview:
    if render_options.training_overview or render_options.top_background or render_options.header_style != "current":
        parser.error("The new overview variants use their own header and training layout")
    render_options.phase_overview = True

ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "figure1_results.json"
DATA = json.loads(SOURCE_PATH.read_text())
EXTRA_SOURCE_PATH = ROOT / "figure1_applications_sources.json"
EXTRA = json.loads(EXTRA_SOURCE_PATH.read_text())
C = json.loads((ROOT / "figure1_sources.json").read_text())["colors"]
for path in (ROOT / "fonts/albert-sans").glob("*.ttf"):
    font_manager.fontManager.addfont(path)
plt.rcParams.update({
    "font.family": "Albert Sans", "font.size": 7,
    "font.sans-serif": ["Albert Sans","DejaVu Sans"],
    "text.color": C["INK"], "axes.labelcolor": C["INK"],
    "axes.edgecolor": C["BORDER"], "xtick.color": C["MUTED"],
    "ytick.color": C["MUTED"], "grid.color": C["BORDER"],
    "mathtext.fontset": "stix", "mathtext.default": "it",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "path", "svg.hashsalt": "adm-applications-overview",
})
INK, MUTED, BORDER = C["INK"], C["MUTED"], C["BORDER"]
PURPLE, ACTION = C["ADM_DEFAULT"], C["ACTION_INK"]
BRIGHT_TITLE = "#A05AFF"
GRAY, LIGHT_GRAY = C["BASELINE_DARK"], C["BASELINE_LIGHT"]
SALMON, STATE = C["LOOKAHEAD_TEACHER"], C["STATE_BORDER"]
# A related dark salmon keeps small labels readable on stronger salmon fills.
TEACHER_INK = "#944735"
BOX_OUTLINES = render_options.bordered
OUTPUT_STEM = "figure1_applications_bordered" if BOX_OUTLINES else "figure1_applications"
if render_options.header_style == "current":
    render_options.header_style = "large"
else:
    OUTPUT_STEM += f"_header_{render_options.header_style}"
if render_options.top_background:
    OUTPUT_STEM += "_lavender"
if render_options.training_overview:
    OUTPUT_STEM += "_training"
if render_options.redesigned_overview:
    OUTPUT_STEM += "_redesign"
elif render_options.refined_overview:
    OUTPUT_STEM += "_refined"
elif render_options.phase_overview:
    OUTPUT_STEM += "_phases"
if render_options.swap_teacher_student:
    OUTPUT_STEM += "_swapped"
if render_options.simplify_layout:
    OUTPUT_STEM += "_simple"
if render_options.phase_bands:
    OUTPUT_STEM += "_bands"
if render_options.no_results:
    OUTPUT_STEM += "_no_results"
if render_options.right_legend:
    OUTPUT_STEM += "_right_legend"
if render_options.no_panel_headers:
    OUTPUT_STEM += "_no_headers"
CLEAR_TRAINING = render_options.training_overview or render_options.phase_overview
if render_options.minimal_overview:
    OUTPUT_STEM = "figure1_minimal"
if render_options.conceptual_overview:
    OUTPUT_STEM = "figure1_conceptual"
if render_options.minimal_concept:
    OUTPUT_STEM = "figure1_conceptual_minimal"
# Apply the selected neutral-flow palette only to this results-bearing comparison.
NEUTRAL_DIAGRAM = OUTPUT_STEM == "figure1_applications_bordered_redesign_swapped_simple_bands_right_legend_no_headers"
HEADER_DESCRIPTIONS = {
    "plain":"Action Decision Models uses one line with equal font size, weight, and light purple color for all three words, centered above the model box",
    "band":"Equal-emphasis light-purple title on a pale lavender band with a purple accent, centered above the model box",
    "tab":"Equal-emphasis white title on a solid light-purple tab, centered above the model box",
    "badge":"A light-purple ADM badge accompanies the full name, whose three words retain equal size, weight, and light-purple color",
    "large":"An oversized vivid-violet wordmark gives all three words equal emphasis without a surrounding shape",
    "underline":"An equal-emphasis vivid-violet title is anchored by a lavender highlight stroke underneath",
    "integrated":"The full name is integrated into a single wider model card: an attached light-purple title bar above the dark-purple formula body",
}
GALLERY_STYLE = "https://github.com/zixianma/OpenWebRL/blob/arm/openwebrl/docs/ARM_FIGURES.md"
APPLICATION_OFFSET = 34 if render_options.refined_overview else 22 if render_options.phase_overview else 0
TOP_CROP = 0 if render_options.minimal_overview else 38 if render_options.redesigned_overview else 0
if render_options.no_panel_headers:
    TOP_CROP += 16
if render_options.minimal_concept:
    TOP_CROP = 0
W, H = 900, 410 if render_options.minimal_overview else 518-TOP_CROP if render_options.redesigned_overview else 430+APPLICATION_OFFSET
if render_options.phase_bands:
    H += 8  # Extra room for the enlarged inference panels and their axis labels.
if render_options.no_panel_headers:
    H -= 28  # Close the empty phase-heading row; retain application-title space.
fig = plt.figure(figsize=(7.2, 7.2 * H / W), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H+TOP_CROP, TOP_CROP))
ax.axis("off")


def tint(color, white):
    channels = [int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(v * (1-white) + 255*white):02x}" for v in channels)


def text(x, y, label, size=7, color=INK, weight="normal", ha="center", **kw):
    va = kw.pop("va", "center")
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, **kw)


def box(x, y, width, height, fill="white", edge=BORDER, radius=7, lw=.6):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=fill,edgecolor=edge if BOX_OUTLINES else "none",
                          linewidth=lw if BOX_OUTLINES else 0)
    ax.add_patch(patch)
    return patch


def line(points, color=BORDER, lw=.7, dashed=False):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw,
            linestyle=(0, (3, 2)) if dashed else "-", zorder=1)


def arrow(start, end, color=PURPLE, lw=.8, scale=5, dashed=False):
    if NEUTRAL_DIAGRAM:
        color = GRAY
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", color=color,
                            linewidth=lw, mutation_scale=scale, shrinkA=0, shrinkB=0)
    if dashed:
        patch.set_linestyle((0,(3,2)))
    ax.add_patch(patch)


def state_window(x, y, width, height, lookahead=False):
    """A browser frame; the neutral-flow comparison uses green for all states."""
    green = not lookahead or NEUTRAL_DIAGRAM
    fill = tint(C["STATE"],.65) if green else tint(SALMON,.60)
    edge = STATE if green else SALMON
    frame = box(x,y,width,height,fill,edge,radius=3,lw=.65)
    if not lookahead:
        frame.set_edgecolor(STATE)
        frame.set_linewidth(.65)
    chrome = min(12,max(5,height*.17))
    body = box(x+.8,y+chrome,width-1.6,height-chrome-.8,"white","none",radius=2,lw=0)
    line([(x+.8,y+chrome),(x+width-.8,y+chrome)],edge,.35)
    for fraction in [.07,.13,.19]:
        dot = min(1.4,width*.025)
        box(x+width*fraction-dot/2,y+chrome*.50-dot/2,dot,dot,edge,"none",radius=dot/2,lw=0)
    box(x+width*.28,y+chrome*.25,width*.63,chrome*.50,"white","none",radius=chrome*.18,lw=0)
    return body


def policy_badge(x,y,trained=False,size=9):
    """Small emoji-style vector badges stay sharp in both PDF and SVG."""
    ax.add_patch(Circle((x,y),size*.60,facecolor="white",edgecolor="none",zorder=5))
    if trained:
        outer = [(0,-.55),(.50,-.13),(.55,.25),(.28,.45),
                 (.07,.60),(-.36,.52),(-.43,.22),
                 (-.48,-.02),(-.27,-.20),(-.20,-.38),
                 (-.18,-.08),(-.02,.01),(0,-.55),(0,-.55)]
        inner = [(0,-.08),(.30,.20),(.20,.48),(0,.49),
                 (-.24,.47),(-.22,.20),(-.08,.10),
                 (-.10,.26),(.02,.20),(0,-.08),(0,-.08)]
        for vertices,color,zorder in [(outer,"#FF9D00",6),(inner,"#FFE078",7)]:
            codes = [VectorPath.MOVETO]+[VectorPath.CURVE4]*(len(vertices)-2)+[VectorPath.CLOSEPOLY]
            points = np.asarray(vertices)*size+np.array([x,y])
            ax.add_patch(PathPatch(VectorPath(points,codes),facecolor=color,
                                   edgecolor="none",zorder=zorder))
    else:
        center = np.array([x,y])
        radius = size*.47
        for angle in np.arange(0,360,60):
            unit = np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
            normal = np.array([-unit[1],unit[0]])
            paths = [[center,center+unit*radius]]
            joint = center+unit*radius*.52
            for side in (-1,1):
                paths.append([joint,joint+unit*radius*.28+normal*radius*.25*side])
            for points in paths:
                points = np.asarray(points)
                ax.plot(points[:,0],points[:,1],color="#73BCEA",lw=.6,
                        solid_capstyle="round",zorder=6)


def pill(x, y, width, label, actor=False, height=27, size=8, policy_status=None):
    color = GRAY if actor else ACTION
    fill = ("white" if BOX_OUTLINES else tint(GRAY,.90)) if actor else tint(PURPLE,.88)
    box(x,y,width,height,fill,color,radius=4,lw=.6)
    text(x+width/2, y+height/2, label, size=size, color=color)
    if policy_status:
        assert actor and policy_status in ("frozen","trained")
        policy_badge(x+width-1,y+1,trained=policy_status=="trained")


def model_pill(x,y,width,height=30,size=9.5):
    box(x,y,width,height,ACTION,ACTION,radius=4,lw=.6)
    text(x+width/2,y+height/2,r"$f_\phi$",size=size,color="white")


def chart(x, y, width, height, title="Mean success (%)"):
    a = fig.add_axes([x/W, (H+TOP_CROP-y-height)/H, width/W, height/H])
    a.set_title(title, fontsize=6.4, color=MUTED, pad=3)
    a.tick_params(axis="both", labelsize=6.4, length=2, width=.5, pad=2, colors=MUTED)
    a.spines[["top", "right"]].set_visible(False)
    for spine in a.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(.6)
    return a


# Read the same Netflix source for both the original and refined comparisons.
TASK = "Task: Find Netflix's accepted payment methods."
EXAMPLE_PATH = ROOT / "luna-pre-post-k3-examples-compact.png"
EXAMPLE_VIEWPORT = (32,340,420,510)
FUTURE_VIEWPORTS = [(774,209,1044,325)]
example_pixels = plt.imread(EXAMPLE_PATH)

if render_options.minimal_concept:
    from draw_figure1_conceptual_minimal import draw_conceptual_minimal
    minimal_concept_layout = draw_conceptual_minimal(text, box, line, arrow, tint, C)
elif render_options.swap_teacher_student:
    from draw_figure1_swapped import draw_swapped_training
    swapped_layout = draw_swapped_training(text, box, line, arrow, state_window, tint, ax,
                                           example_pixels, EXAMPLE_VIEWPORT, FUTURE_VIEWPORTS[0], C,
                                           simplified=render_options.simplify_layout, phase_bands=render_options.phase_bands,
                                           panel_headers=not render_options.no_panel_headers,
                                           neutral_diagram=NEUTRAL_DIAGRAM)
elif render_options.minimal_overview:
    from plot_figure1_minimal import draw_minimal_training
    minimal_training_layout = draw_minimal_training(text, box, arrow, tint, C)
elif render_options.redesigned_overview:
    # A compact two-dimensional evidence grid supports a dominant student row.
    text(450,54,"Training",size=11,color=INK,weight="bold")
    for left,right in [(24,383),(517,884)]:
        line([(left,54),(right,54)],BORDER,.75)
    box(24,83,3,53,C["STATE"],"none",radius=1.5,lw=0)
    text(36,109,"Task: Find Netflix's\naccepted payment\nmethods.",size=8.5,color=INK,weight="bold",ha="left",linespacing=1.25)

    # Show the first future state, with dots retaining both continuation axes.
    evidence_group = box(218,80,260,80,tint(SALMON,.94),"none",radius=7,lw=0)
    evidence_group.set_zorder(-1)
    text(348,74,r"Privileged future states $\mathcal{E}_t$",size=7.5,color=TEACHER_INK)
    for x,label in [(263,r"$a_t^{(1)}$"),(350,r"$a_t^{(2)}$"),(394,r"$\cdots$"),(438,r"$a_t^{(K)}$")]:
        text(x,95,label,size=6,color=TEACHER_INK)
    for y,label in [(116.5,r"$s_{t+1}$"),(147,r"$\vdots$")]:
        text(203,y,label,size=6.6 if y!=147 else 8,color=TEACHER_INK)

    def future_cell(x,y,real=False):
        body = state_window(x,y,68,25,lookahead=True)
        if real:
            l,t,r,b = FUTURE_VIEWPORTS[0]
            shot = example_pixels[t:b,l:r]
            h = 66.4*shot.shape[0]/shot.shape[1]
            cy = y+5+(25-5-.8)/2
            page = ax.imshow(shot,extent=(x+.8,x+67.2,cy+h/2,cy-h/2),
                             aspect="auto",interpolation="lanczos",zorder=3)
            page.set_clip_path(body)
        else:
            box(x+5,y+9,54,1.8,tint(SALMON,.05),"none",radius=.3,lw=0)
            box(x+5,y+13,8,8,tint(SALMON,.60),"none",radius=.3,lw=0)
            for dy,length in [(14,40),(18,34),(22,37)]:
                box(x+17,y+dy,length,1,SALMON,"none",radius=0,lw=0)

    for x in [229,316,404]:
        future_cell(x,104,real=x==316)
    text(394,116.5,r"$\cdots$",size=8,color=SALMON)
    for x in [263,350,438]:
        text(x,147,r"$\vdots$",size=6,color=SALMON)
        arrow((x,200),(x,160),SALMON,lw=.75,scale=3.5)
    text(394,147,r"$\ddots$",size=6,color=SALMON)
    line([(478,80),(484,80),(484,160),(478,160)],SALMON,.7)
    arrow((488,92),(543,92),SALMON,lw=.75,scale=3.5)

    # A lighter, smaller teacher leaves the visual emphasis on the student.
    box(547,75,122,34,tint(SALMON,.88),SALMON,radius=4,lw=.45)
    text(608,84,"Teacher",size=7.4,color=TEACHER_INK,weight="bold")
    text(608,100,r"$\mathcal{T}(s_t,\mathcal{A}_t,\mathcal{E}_t)$",size=8.3,color=TEACHER_INK)
    arrow((673,92),(746,92),SALMON,lw=.75,scale=3.5)
    box(750,78,134,28,tint(SALMON,.90),SALMON,radius=4,lw=.45)
    text(817,92,r"$a_t^+=a_t^{(2)}$",size=10,color=TEACHER_INK)

    text(104,179,r"State $s_t$",size=8.6,color=INK)
    body = state_window(16,190,176,78)
    left,top,right,bottom = EXAMPLE_VIEWPORT
    pixels = example_pixels[top:bottom,left:right]
    photo_width = 164
    h = photo_width*pixels.shape[0]/pixels.shape[1]
    cy = 190+78*.17+(78-78*.17-.8)/2
    page = ax.imshow(pixels,extent=(22,186,cy+h/2,cy-h/2),aspect="auto",interpolation="lanczos",zorder=3)
    page.set_clip_path(body)
    source_scale = photo_width/(right-left)
    photo_top = cy-h/2
    for source_rect,label,color in [((136,354,158,396),"2",ACTION),
                                    ((383,376,414,418),"1",PURPLE)]:
        x0,y0,x1,y1 = source_rect
        x,y = 22+(x0-left)*source_scale,photo_top+(y0-top)*source_scale
        w,h = (x1-x0)*source_scale,(y1-y0)*source_scale
        cover = box(x,y,w,h,"white","none",radius=.5,lw=0)
        cover.set_zorder(4)
        marker_cx,marker_cy = (108,226) if label=="2" else (x+w/2,y+h/2)
        marker_width,marker_height = 28,19
        marker = box(marker_cx-marker_width/2,marker_cy-marker_height/2,
                     marker_width,marker_height,color,"none",radius=1,lw=0)
        marker.set_zorder(4)
        text(marker_cx,marker_cy,rf"$a_t^{{({label})}}$",size=6,color="white",zorder=5)
        if label=="2":
            ax.add_patch(FancyArrowPatch((94,226),(75,226),arrowstyle="-|>",
                                        color=color,linewidth=.65,mutation_scale=3,
                                        shrinkA=0,shrinkB=0,zorder=5))
    # The source's old number badge masked part of this heading; restore its text.
    heading_cover = box(36,201,70,13,"white","none",radius=0,lw=0)
    heading_cover.set_zorder(4)
    text(38,207.5,"Payment methods",size=4.3,color="#282828",weight="bold",ha="left",zorder=5)

    text(310,189,r"Actions $\mathcal{A}_t$",size=7,color=ACTION)
    for i,x,width in [("1",223,80),("2",310,80),("K",418,40)]:
        box(x,200,width,58,tint(PURPLE,.68 if i=="2" else .90),ACTION,radius=4,lw=.6)
        cx = x+width/2
        text(cx,214 if i!="K" else 229,rf"$a_t^{{({i})}}$",size=10.5,color=ACTION)
        if i!="K":
            text(cx,244,"Scroll down" if i=="1" else "Click payment\noptions",size=6.4 if i=="1" else 5.8,color=INK)
    text(406,229,r"$\cdots$",size=8,color=ACTION)
    arrow((198,229),(219,229),ACTION,lw=.75,scale=3.5)
    text(206,214,r"$\pi_\theta$",size=7,color=GRAY)
    arrow((466,229),(488,229),ACTION,lw=.75,scale=3.5)
    text(608,151,"Action Decision\nModels",size=15,color=BRIGHT_TITLE,weight="bold",linespacing=1.0)
    box(492,195,232,68,ACTION,ACTION,radius=6,lw=.6)
    text(608,211,"ADM",size=11.5,color="white",weight="bold")
    text(608,241,r"$f_\phi(s_t,\mathcal{A}_t)$",size=14.3,color="white")
    arrow((728,229),(746,229),ACTION,lw=.75,scale=3.5)

    # Label, loss, probabilities and selected action share one rightmost spine.
    box(750,137,134,28,"white",BORDER,radius=4,lw=.6)
    text(817,151,r"$\mathcal{L}_{\mathrm{ADM}}=-\log p_\phi(a_t^+)$",size=8,color=ACTION)
    arrow((817,106),(817,137),SALMON,lw=.75,scale=3)
    arrow((817,213),(817,165),ACTION,lw=.75,scale=3)
    box(750,213,134,32,tint(PURPLE,.96),ACTION,radius=4,lw=.6)
    text(817,229,r"$p_\phi$",size=14,color=ACTION)
    arrow((817,245),(817,252),ACTION,lw=.75,scale=3)
    box(800,252,34,22,tint(PURPLE,.85),ACTION,radius=4,lw=.6)
    text(817,263,r"$\hat a_t$",size=9.5,color=ACTION)
    text(842,250,"argmax",size=5.2,color=ACTION,ha="left")
elif render_options.refined_overview:
    # A shared grid gives each action a future-state browser directly beneath it.
    text(24,11.5,"Training",size=8,color=MUTED,weight="bold",ha="left")
    text(694,34,"Action Decision Models",size=18,color=BRIGHT_TITLE,weight="bold",va="baseline")
    box(24,19,3,18,C["STATE"],"none",radius=1.5,lw=0)
    text(35,33,TASK,size=8.6,color=INK,weight="bold",ha="left",va="baseline")
    text(80,52,r"State $s_t$",size=7.5,color=INK)
    text(310,52,r"Actions $\mathcal{A}_t$",size=7.5,color=ACTION)

    left,top,right,bottom = EXAMPLE_VIEWPORT
    pixels = example_pixels[top:bottom,left:right]
    observation_height = 100*pixels.shape[0]/pixels.shape[1]
    state_window(24,61,112,54)
    ax.imshow(pixels,extent=(30,130,92.5+observation_height/2,92.5-observation_height/2),
              aspect="auto",interpolation="lanczos",zorder=3)
    source_scale = 100/(right-left)
    photo_top = 92.5-observation_height/2
    for source_rect,label,color in [((136,354,158,396),"2",ACTION),
                                    ((383,376,414,418),"1",PURPLE)]:
        x0,y0,x1,y1 = source_rect
        x,y = 30+(x0-left)*source_scale,photo_top+(y0-top)*source_scale
        w,h = (x1-x0)*source_scale,(y1-y0)*source_scale
        cover = box(x,y,w,h,"white","none",radius=.5,lw=0)
        cover.set_zorder(4)
        marker = box(x+.45,y+.45,w-.9,h-.9,color,"none",radius=.4,lw=0)
        marker.set_zorder(4)
        text(x+w/2,y+h/2,label,size=3.7,color="white",weight="bold",zorder=5)

    for i,x,label in [("1",166,"Scroll down"),("2",324,"Click payment\noptions")]:
        box(x,63,130,50,tint(PURPLE,.68 if i=="2" else .90),ACTION,radius=4,lw=.6)
        text(x+22,88,rf"$a_t^{{({i})}}$",size=11,color=ACTION)
        text(x+86,88,label,size=6.6 if i=="2" else 7.1,color=INK)
    arrow((142,88),(160,88),ACTION,lw=.75,scale=3.5)
    text(488,88,r"$\cdots$",size=8,color=ACTION)
    arrow((532,88),(588,88),ACTION,lw=.75,scale=3.5)
    box(592,66,204,44,ACTION,ACTION,radius=5,lw=.6)
    text(694,88,r"ADM $f_\phi(s_t,\mathcal{A}_t)$",size=12.6,color="white",weight="bold")
    arrow((800,88),(824,88),ACTION,lw=.75,scale=3.5)
    box(828,72,56,32,tint(PURPLE,.85),ACTION,radius=4,lw=.6)
    text(856,88,r"$\hat a_t$",size=11,color=ACTION)

    for x in [231,389]:
        line([(x,113),(x,138)],SALMON,.7)
    text(310,123,r"Future states $\mathcal{E}_t$",size=7.2,color=TEACHER_INK)
    state_window(175,138,112,48,lookahead=True)
    box(185,152,84,3,tint(SALMON,.05),"none",radius=.4,lw=0)
    box(185,161,16,17,tint(SALMON,.60),"none",radius=.4,lw=0)
    for y,length in [(162,57),(169,47),(176,52)]:
        box(210,y,length,1.5,SALMON,"none",radius=0,lw=0)
    text(231,194,"Billing",size=7.2,color=TEACHER_INK)

    l,t,r,b = FUTURE_VIEWPORTS[0]
    shot = example_pixels[t:b,l:r]
    body = state_window(333,138,112,48,lookahead=True)
    photo_width = 110.4
    h = photo_width*shot.shape[0]/shot.shape[1]
    cy = 138+48*.17+(48-48*.17-.8)/2
    page = ax.imshow(shot,extent=(333.8,444.2,cy+h/2,cy-h/2),
                     aspect="auto",interpolation="lanczos",zorder=3)
    page.set_clip_path(body)
    text(389,194,"Payment methods",size=7.2,color=TEACHER_INK)
    for x in [310,461]:
        text(x,162,r"$\cdots$",size=9,color=SALMON)
    line([(478,138),(488,138),(488,186),(478,186)],SALMON,.7)
    arrow((492,162),(588,162),SALMON,lw=.75,scale=3.5)

    # Teacher and student share a centerline; supervision is one vertical link.
    box(592,140,204,44,tint(SALMON,.68),SALMON,radius=5,lw=.6)
    text(694,150,"Teacher",size=8.2,color=TEACHER_INK,weight="bold")
    text(694,172,r"$\mathcal{T}(s_t,\mathcal{A}_t,\mathcal{E}_t)$",size=9.5,color=TEACHER_INK)
    arrow((694,136),(694,114),SALMON,lw=.8,scale=3.5,dashed=True)
    text(708,123,r"Label $a_t^{(2)}$",size=7.5,color=TEACHER_INK,ha="left")
else:
    # Two aligned tracks: pre-execution selection above, privileged supervision
    # below. The application columns retain their widths and component sizes.
    if render_options.top_background:
        background = box(10,7,880,185,tint(PURPLE,.91),"none",radius=10,lw=0)
        background.set_zorder(-2)
    if render_options.header_style == "band":
        box(548,12,292,38,tint(PURPLE,.90),"none",radius=6,lw=0)
        box(548,12,4,38,PURPLE,"none",radius=2,lw=0)
        text(694,36,"Action Decision Models",size=13.2,color=PURPLE,weight="bold",va="baseline")
    elif render_options.header_style == "tab":
        box(548,12,292,38,PURPLE,"none",radius=6,lw=0)
        text(694,36,"Action Decision Models",size=13.2,color="white",weight="bold",va="baseline")
    elif render_options.header_style == "badge":
        box(530,11,68,40,PURPLE,"none",radius=5,lw=0)
        text(564,36,"ADM",size=14,color="white",weight="bold",va="baseline")
        text(611,36,"Action Decision Models",size=12.6,color=PURPLE,weight="bold",
             ha="left",va="baseline")
    elif render_options.header_style == "large":
        text(694,41,"Action Decision Models",size=18,color=BRIGHT_TITLE,weight="bold",va="baseline")
    elif render_options.header_style == "underline":
        box(551,41,286,6,tint(PURPLE,.63),"none",radius=3,lw=0)
        text(694,36,"Action Decision Models",size=14.6,color=BRIGHT_TITLE,weight="bold",va="baseline")
    elif render_options.header_style == "integrated":
        pass  # This title is drawn inside the model card below.
    else:
        text(694,33,"Action Decision Models",size=12.6,color=PURPLE,weight="bold",va="baseline")
    TASK_OFFSET = 4 if render_options.phase_overview else 0
    if render_options.phase_overview:
        text(24,11.5,"Training",size=8,color=TEACHER_INK,weight="bold",ha="left")
    box(24,19+TASK_OFFSET,3,18,C["STATE"],"none",radius=1.5,lw=0)
    text(35,33+TASK_OFFSET,TASK,size=8.6,color=INK,weight="bold",ha="left",va="baseline")
    text(80,52,r"State $s_t$" if render_options.phase_overview else r"Current state $s_t$",size=7.5,color=INK)
    text(310,52,r"Actions $\mathcal{A}_t$" if render_options.phase_overview else r"Candidate actions $\mathcal{A}_t$",size=7.5,color=ACTION)
    left,top,right,bottom = EXAMPLE_VIEWPORT
    pixels = example_pixels[top:bottom,left:right]
    observation_height = 100*pixels.shape[0]/pixels.shape[1]
    state_window(24,61,112,54)
    ax.imshow(pixels,extent=(30,130,92.5+observation_height/2,92.5-observation_height/2),
              aspect="auto",interpolation="lanczos",zorder=3)
    # Figure 5's 1/2 badges are annotations, not web content. Relabel those
    # vector overlays to 2/1 so the screenshot agrees with the overview ordering.
    source_scale = 100/(right-left)
    photo_top = 92.5-observation_height/2
    for source_rect,label,color in [((136,354,158,396),"2",ACTION),
                                    ((383,376,414,418),"1",PURPLE)]:
        x0,y0,x1,y1 = source_rect
        x,y = 30+(x0-left)*source_scale,photo_top+(y0-top)*source_scale
        w,h = (x1-x0)*source_scale,(y1-y0)*source_scale
        cover = box(x,y,w,h,"white","none",radius=.5,lw=0)
        cover.set_zorder(4)
        marker = box(x+.45,y+.45,w-.9,h-.9,color,"none",radius=.4,lw=0)
        marker.set_zorder(4)
        text(x+w/2,y+h/2,label,size=3.7,color="white",weight="bold",zorder=5)
    for i,x,label in [("1",166,"Scroll down"),("2",324,"Click payment\noptions")]:
        selected = i=="2"
        box(x,63,130,50,tint(PURPLE,.68 if selected else .90),
            ACTION,radius=4,lw=1.05 if selected else .6)
        text(x+22,88,rf"$a_t^{{({i})}}$",size=11,color=ACTION)
        text(x+86,88,label,size=6.6 if selected else 7.1,color=INK)
    arrow((142,88),(160,88),ACTION,lw=.75,scale=3.5)
    text(488,88,r"$\cdots$",size=8,color=ACTION)
    if render_options.header_style == "integrated":
        arrow((516,88),(540,88),ACTION,lw=.75,scale=3.5)
        box(550,24,264,94,ACTION,ACTION,radius=5,lw=.9)
        box(550,24,264,34,PURPLE,"none",radius=5,lw=0)
        text(682,48,"Action Decision Models",size=12.8,color="white",weight="bold",va="baseline")
        text(682,88,r"ADM $f(s_t,\mathcal{A}_t)$",size=12.6,color="white",weight="bold")
        if not render_options.training_overview:
            arrow((818,88),(842,88),ACTION,lw=.75,scale=3.5)
            box(848,72,36,32,tint(PURPLE,.85),ACTION,radius=4,lw=.6)
            text(866,88,r"$\hat a_t$",size=11,color=ACTION)
    else:
        arrow((532,88),(556,88),ACTION,lw=.75,scale=3.5)
        box(592,58,204,60,ACTION,ACTION,radius=5,lw=.9)
        text(694,88,r"ADM $f(s_t,\mathcal{A}_t)$",size=12.6,color="white",weight="bold")
        if not render_options.training_overview:
            arrow((800,88),(824,88),ACTION,lw=.75,scale=3.5)
            box(828,72,56,32,tint(PURPLE,.85),ACTION,radius=4,lw=.6)
            text(856,88,r"$\hat a_t$",size=11,color=ACTION)

    # One next state per candidate; ellipses retain the remaining lookahead.
    for x in [231,389]:
        line([(x,113),(x,130)],SALMON,.7)
    text(310,121,r"Future states $\mathcal{E}_t$" if render_options.phase_overview else r"Future states $\mathcal{E}$",size=7.2,color=TEACHER_INK)
    for x,viewport in zip([346],FUTURE_VIEWPORTS):
        l,t,r,b = viewport
        shot = example_pixels[t:b,l:r]
        body = state_window(x,130,86,42,lookahead=True)
        # Fill the browser content area, excluding the source image's border.
        # Preserve aspect ratio and clip the small vertical excess to the page body.
        photo_width = 86-1.6
        h = photo_width*shot.shape[0]/shot.shape[1]
        cy = 130+42*.17+(42-42*.17-.8)/2
        page = ax.imshow(shot,extent=(x+.8,x+85.2,cy+h/2,cy-h/2),aspect="auto",interpolation="lanczos",zorder=3)
        page.set_clip_path(body)
        text(x+43,183,"Payment methods" if CLEAR_TRAINING else "Answer page",size=7.2,color=TEACHER_INK)
    state_window(188,130,86,42,lookahead=True)
    box(196,142,64,3,tint(SALMON,.05),"none",radius=.4,lw=0)
    box(196,150,12,15,tint(SALMON,.60),"none",radius=.4,lw=0)
    for dy,length in [(151,44),(157,36),(163,40)]:
        box(215,dy,length,1.5,SALMON,"none",radius=0,lw=0)
    text(231,183,"More billing text" if render_options.training_overview else "Billing",size=7.2,color=TEACHER_INK)
    for x in [299,457]:
        text(x,151,r"$\cdots$",size=9,color=SALMON)

    # A local vertical bracket collects lookahead on the same row as the teacher.
    line([(470,130),(480,130),(480,172),(470,172)],SALMON,.7)
    arrow((484,151),(508,151),SALMON,lw=.75,scale=3.5)
    box(512,127,124,48,tint(SALMON,.68),SALMON,radius=5,lw=.75)
    text(574,139,"Teacher",size=8.2,color=TEACHER_INK,weight="bold")
    text(574,162,r"$\mathcal{T}(s_t,\mathcal{A}_t,\mathcal{E}_t)$" if render_options.phase_overview else r"$\mathcal{T}(s_t,\mathcal{A}_t,\mathcal{E})$",size=9,color=TEACHER_INK)
    if CLEAR_TRAINING:
        line([(640,151),(744,151)],SALMON,.75,dashed=True)
        text(700,167,r"Label $a_t^{(2)}$" if render_options.phase_overview else r"Teacher label $a_t^{(2)}$",size=8 if render_options.phase_overview else 7,color=TEACHER_INK)
        arrow((744,151),(744,122),SALMON,lw=.75,scale=3.5,dashed=True)
        if not render_options.phase_overview:
            text(756,136,"Train ADM",size=7.2,color=TEACHER_INK,ha="left")
    else:
        line([(640,151),(694,151)],SALMON,.75)
        text(666,165,r"$a_t^{(2)}$",size=10,color=TEACHER_INK)
        arrow((694,151),(694,122),SALMON,lw=.75,scale=3.5)
        text(706,136,"Train ADM",size=7.2,color=TEACHER_INK,ha="left")

# The selected no-header comparison uses neutral arrows; other variants keep role colors.
top_arrow_colors = {matplotlib.colors.to_hex(p.get_edgecolor())
                    for p in ax.patches if isinstance(p,FancyArrowPatch)}
assert top_arrow_colors == ({GRAY.lower()} if NEUTRAL_DIAGRAM else {ACTION.lower(),SALMON.lower()}),top_arrow_colors

# Unboxed application columns retain the original reading order and plots.
ORIGINS = [14,310,606]
HEADING_Y, ROW_Y, PLOT_Y, LEGEND_Y = [y+APPLICATION_OFFSET for y in (198,250,294,420)]
if render_options.redesigned_overview:
    app_shift = -112 if render_options.minimal_concept else -108 if render_options.minimal_overview else -28 if render_options.no_panel_headers else 0
    HEADING_Y, ROW_Y, PLOT_Y, LEGEND_Y = [y+app_shift for y in (296,340,382,508)]
    application_heading_y = 278+app_shift-(7 if render_options.simplify_layout else 0)
    if not render_options.no_panel_headers:
        if render_options.phase_bands:
            band = box(24,application_heading_y-13,860,26,"#F3F4F6","none",radius=3,lw=0)
            band.set_zorder(-2)
        text(450,application_heading_y,"B. Applications" if render_options.phase_bands else "Applications",size=9 if render_options.phase_bands else 11,color=INK,weight="bold",**({"clip_on":False} if render_options.phase_bands else {}))
        if not render_options.phase_bands:
            for left,right in [(24,366),(534,884)]:
                line([(left,application_heading_y),(right,application_heading_y)],BORDER,.75)
elif render_options.phase_overview:
    divider_y = 207 if render_options.refined_overview else 195
    line([(24,divider_y),(884,divider_y)],BORDER,.85)
    text(24,220 if render_options.refined_overview else 208,"Applications",size=8,
         color=MUTED if render_options.refined_overview else ACTION,weight="bold",ha="left")
if render_options.redesigned_overview and not render_options.simplify_layout:
    for x in ORIGINS:
        band = box(x,316+app_shift,280,50,"#F7F7FA","none",radius=7,lw=0)
        band.set_zorder(-1)
for x,heading in zip(ORIGINS,["(a) Test-time scaling","(b) Offline actor improvement",
                            "(c) Online actor improvement via RL"]):
    if render_options.phase_bands and not render_options.no_panel_headers:
        heading = heading.replace("(a)","(1)").replace("(b)","(2)").replace("(c)","(3)").replace("Online actor improvement via RL","Online improvement via RL")
    text(x+14,HEADING_Y,heading,size=8.7,
         weight="normal" if render_options.no_panel_headers else "bold",ha="left",va="top")
for x in ([] if render_options.redesigned_overview else [300,596]):
    line([(x,HEADING_Y),(x,H-4)],tint(BORDER,.7 if render_options.refined_overview else .4),
         .4 if render_options.refined_overview else .45)

# Keep the training colors semantic and use one neutral application-flow color.
def application_arrow(start, end, color=PURPLE, **kwargs):
    arrow(start, end, GRAY if render_options.phase_bands else color, **kwargs)

application_arrow_start = len(ax.patches)
# One row shows proposal, candidate set, decision model, and selected action.
x=ORIGINS[0]
pill(x+14,ROW_Y-12,28,r"$\pi_\theta$",actor=True,height=24,size=9 if render_options.phase_bands else 8,policy_status="frozen")
application_arrow((x+46,ROW_Y),(x+56,ROW_Y),GRAY,scale=3)
box(x+60,ROW_Y-16,108,32,tint(PURPLE,.93),ACTION,radius=4,lw=.6)
for dx,label in [(77,r"$a_t^{(1)}$"),(105,r"$a_t^{(2)}$"),
                  (127,r"$\cdots$"),(151,r"$a_t^{(K)}$")]:
    text(x+dx,ROW_Y,label,size=7,color=GRAY if NEUTRAL_DIAGRAM else ACTION)
application_arrow((x+172,ROW_Y),(x+186,ROW_Y),ACTION,scale=3)
model_pill(x+190,ROW_Y-15,34)
application_arrow((x+228,ROW_Y),(x+242,ROW_Y),ACTION,scale=3)
pill(x+246,ROW_Y-12,28,r"$\hat a_t$",height=24,size=9)

# RPO jointly combines preference optimization and chosen-response imitation.
# SFT is shown as a loss component, distinct from the teacher-SFT baseline.
x=ORIGINS[1]
model_pill(x+16,ROW_Y-15,30)
application_arrow((x+50,ROW_Y),(x+58,ROW_Y),ACTION,scale=3)
pill(x+62,ROW_Y-14,74,r"$\hat a_t\succ a_t^{-}$",height=28,size=8)
application_arrow((x+140,ROW_Y),(x+148,ROW_Y),ACTION,scale=3)
if render_options.redesigned_overview:
    box(x+152,ROW_Y-20,82,40,tint(PURPLE,.88),ACTION,radius=4,lw=.6)
    text(x+193,ROW_Y-11,"RPO",size=6.3,color=ACTION,weight="bold")
    component_y = ROW_Y+8
else:
    box(x+152,ROW_Y-14,82,28,tint(PURPLE,.88),ACTION,radius=4,lw=.6)
    text(x+193,ROW_Y-24,"RPO",size=6.3,color=ACTION,weight="bold")
    component_y = ROW_Y
text(x+171,component_y,"DPO",size=6,color=ACTION)
text(x+193,component_y,"+",size=7,color=ACTION)
text(x+215,component_y,"SFT",size=6,color=ACTION)
application_arrow((x+238,ROW_Y),(x+246,ROW_Y),ACTION,scale=3)
pill(x+250,ROW_Y-12,28,r"$\pi_\theta$",actor=True,height=24,size=9 if render_options.phase_bands else 8,policy_status="trained")

x=ORIGINS[2]
model_pill(x+20,ROW_Y-15,34)
application_arrow((x+58,ROW_Y),(x+68,ROW_Y),scale=3)
pill(x+72,ROW_Y-15,34,r"$\hat a_t$",height=30,size=9)
application_arrow((x+110,ROW_Y),(x+120,ROW_Y),scale=3)
pill(x+124,ROW_Y-15,92,"+ ADM reward",height=30,size=6.2)
application_arrow((x+220,ROW_Y),(x+230,ROW_Y),ACTION,scale=3)
pill(x+234,ROW_Y-15,32,r"$\pi_\theta$",actor=True,height=30,size=9 if render_options.phase_bands else 8,policy_status="trained")

if render_options.phase_bands:
    application_arrows = [p for p in ax.patches[application_arrow_start:] if isinstance(p,FancyArrowPatch)]
    assert len(application_arrows) == 9
    assert {matplotlib.colors.to_hex(p.get_edgecolor()) for p in application_arrows} == {GRAY.lower()}

illustration_artists = set(ax.get_children())

inference_data = DATA["inference"]
if render_options.phase_bands:
    inference_data = json.loads((ROOT / "figure1_inference_figure7.json").read_text())
points = inference_data["points"]
styles = {
    "sft1": (LIGHT_GRAY, "s"),
    "sft8b": (GRAY, "s"),
    "lora-gpt55-k3": (C["ADM_4B"], "D"),
    "head-gpt55-k3": (C["ADM_22M"], "o"),
}
if render_options.phase_bands:
    styles.update({"sft1":(LIGHT_GRAY,"o"),"sft8b":(GRAY,"o"),
                   "head-gpt55-k3":("#B69AE8","D"),"head-4m-gpt55":("#B69AE8","D"),
                   "lora-gpt55-k3":("#8246E5","D"),
                   "best-of-q":(GRAY,"^"),"pro-cua-prm":(GRAY,"v"),
                   "webdreamer":(GRAY,"D"),"kev-4b":(GRAY,"o")})
system_parameters = EXTRA["inference_system_size"]["system_parameters_billions"]
system_points = [dict(p,system_parameters_billions=p["system_parameters_billions"] if render_options.phase_bands else system_parameters[p["key"]]) for p in points]
system_frontier = sorted(
    (p for p in system_points if not any(
        q["system_parameters_billions"] <= p["system_parameters_billions"] and
        q["percent"] >= p["percent"] and
        (q["system_parameters_billions"] < p["system_parameters_billions"] or q["percent"] > p["percent"])
        for q in system_points)),key=lambda p:p["system_parameters_billions"])


inference_frontier = inference_data.get("frontier")
if render_options.phase_bands:
    inference_frontier = sorted((p for p in points if not any(
        q["pflop_per_episode"] <= p["pflop_per_episode"] and q["percent"] >= p["percent"] and
        (q["pflop_per_episode"] < p["pflop_per_episode"] or q["percent"] > p["percent"])
        for q in points)),key=lambda p:p["pflop_per_episode"])
    assert "adm-distilled" not in {p["key"] for p in points}
    inference_data["frontier"] = inference_frontier

def inference_panel(a, rows, frontier, x_key):
    a.plot([p[x_key] for p in frontier], [p["percent"] for p in frontier],
           color=C["FRONTIER"], ls="--", lw=.9, zorder=1)
    a.scatter([p[x_key] for p in frontier], [p["percent"] for p in frontier],
              s=35,facecolors="none",edgecolors=C["FRONTIER"],lw=.5,zorder=2)
    for p in rows:
        color,marker = styles[p["key"]]
        a.scatter(p[x_key],p["percent"],s=25 if marker=="D" else 18,
                  marker=marker,facecolors="none" if render_options.phase_bands and p["key"] in ("best-of-q","pro-cua-prm","webdreamer","kev-4b") else color,
                  edgecolors=color if render_options.phase_bands else "white",lw=.8 if render_options.phase_bands else .5,zorder=3)
    a.set(ylim=(35,48),yticks=[36,40,44]) if render_options.phase_bands else a.set(ylim=(38,50),yticks=[40,45])


compact_result_layout = render_options.right_legend and render_options.no_panel_headers
inference_plot_width = 94 if compact_result_layout else 112 if render_options.phase_bands else 100
inference_plot_x = 155 if compact_result_layout else ORIGINS[0]+164 if render_options.phase_bands else ORIGINS[0]+168
offline_plot_x = 370 if compact_result_layout else 410 if render_options.right_legend else ORIGINS[1]+30
offline_plot_width = 208 if compact_result_layout else 168 if render_options.right_legend else 238
inference_legend_x = 262 if compact_result_layout else 301
system_size = chart(ORIGINS[0]+26 if render_options.phase_bands else ORIGINS[0]+30,PLOT_Y,inference_plot_width,104 if render_options.phase_bands else 80)
inference_panel(system_size,system_points,system_frontier,"system_parameters_billions")
system_size.set(xlim=(3.2,19),xticks=[4.44,17.17]) if render_options.phase_bands else system_size.set(xlim=(3.5,9.7),xticks=[4.44,8.88])
system_size.set_xlabel("System size (B)",fontsize=6.1,labelpad=2,color=MUTED)
inference = chart(inference_plot_x,PLOT_Y,inference_plot_width,104 if render_options.phase_bands else 80)
inference_panel(inference,points,inference_frontier,"pflop_per_episode")
inference.set(xlim=(.15,2.85),xticks=[.3,1.5,2.7]) if render_options.phase_bands else inference.set(xlim=(.15,1.62),xticks=[.3,.9,1.5])
inference.set_xlabel("PFLOs / task" if render_options.phase_bands else "PFLOPs / episode",fontsize=6.1,labelpad=2,color=MUTED)
inference.tick_params(axis="y",labelleft=False)
if render_options.phase_bands and not render_options.right_legend:
    # Direct method labels use short leaders so nearby systems remain distinct.
    method_label_positions = {
        "system_parameters_billions": {
            "head-4m-gpt55":(.03,.91,"left"),"lora-gpt55-k3":(.49,.91,"left"),
            "sft1":(.03,.54,"left"),"sft8b":(.48,.62,"left"),
            "kev-4b":(.49,.44,"left"),"best-of-q":(.02,.15,"left"),
            "pro-cua-prm":(.43,.04,"left"),"webdreamer":(.98,.28,"right")},
        "pflop_per_episode": {
            "head-4m-gpt55":(.02,.91,"left"),"lora-gpt55-k3":(.51,.91,"left"),
            "sft1":(.02,.54,"left"),"sft8b":(.02,.67,"left"),
            "kev-4b":(.53,.47,"left"),"best-of-q":(.02,.15,"left"),
            "pro-cua-prm":(.50,.04,"left"),"webdreamer":(.98,.63,"right")},
    }
    for panel,x_key in ((system_size,"system_parameters_billions"),(inference,"pflop_per_episode")):
        for point in system_points:
            label_x,label_y,align = method_label_positions[x_key][point["key"]]
            color,_ = styles[point["key"]]
            panel.annotate("",xy=(point[x_key],point["percent"]),xytext=(label_x,label_y),
                           textcoords="axes fraction",arrowprops={"arrowstyle":"-","color":color,"lw":.4},zorder=2)
            panel.text(label_x,label_y,point["label"],transform=panel.transAxes,
                       ha=align,va="center",fontsize=4.5,color=color,zorder=5,
                       bbox={"facecolor":"white","edgecolor":"none","pad":.25,"alpha":.94})

legend_entries = [(18,"sft1",LEGEND_Y),(65,"sft8b",LEGEND_Y),
                  (112,"head-gpt55-k3",LEGEND_Y),(200,"lora-gpt55-k3",LEGEND_Y)]
if render_options.phase_bands:
    legend_entries = []  # Method names are attached directly to every point.
for dx,key,legend_y in legend_entries:
    color,marker = styles[key]
    x = ORIGINS[0]+dx
    hollow = render_options.phase_bands and key in ("best-of-q","pro-cua-prm","webdreamer","kev-4b")
    ax.scatter(x,legend_y,s=15,marker=marker,facecolors="none" if hollow else color,
               edgecolors=color if hollow else "none",lw=.7)
    label = next(p["label"] for p in points if p["key"]==key)
    text(x+8,legend_y,label,size=4.5 if hollow else 5.6,color=color,ha="left")

# Read the paper's current Table 7 averages rather than the historical OM2W
# snapshot used by earlier overview figures. Teacher SFT is a separate baseline
# from ADM-guided filtered SFT, which remains omitted from this compact view.
OFFLINE_TABLE_PATH = ROOT.parent / "tables/offline-transfer-main.tex"
offline_table_bytes = OFFLINE_TABLE_PATH.read_bytes()
offline_table_lines = offline_table_bytes.decode().splitlines()
offline_rates = {}
for key,prefix,label in [("actor","SFT 4B","Actor"),
                         ("teacher_sft","Teacher actor SFT","Teacher SFT"),
                         ("dpo","DPO","ADM-guided DPO"),
                         ("rpo",r"\textbf{RPO}","ADM-guided RPO")]:
    row = next(line for line in offline_table_lines if line.startswith(prefix) and "&" in line)
    cells = row.split("&")[1:]
    assert len(cells)==4,(prefix,cells)
    rates = [float(re.search(r"\d+(?:\.\d+)?",cell).group()) for cell in cells]
    assert abs(sum(rates[:3])/3-rates[3])<=.1,(prefix,rates)
    offline_rates[key] = {"label":label,"overall":rates[3],
                          "benchmarks":dict(zip(("webvoyager","online_mind2web","deepshop"),rates[:3]))}
OFFLINE_DATA = {"metric":"Equal-weight three-benchmark mean overall success (%)",
                "source":{"path":"tables/offline-transfer-main.tex",
                          "sha256":hashlib.sha256(offline_table_bytes).hexdigest(),
                          "table":"tab:offline-transfer-main","column":"Avg"},
                "rates":offline_rates}

offline = chart(offline_plot_x,PLOT_Y,offline_plot_width,80)
keys = ["actor","teacher_sft","dpo","rpo"]
values = [OFFLINE_DATA["rates"][k]["overall"] for k in keys]
offline.bar(range(4),values,width=.34,color=[GRAY,LIGHT_GRAY,tint(PURPLE,.35),PURPLE])
for i, value in enumerate(values):
    offline.text(i, value+1.8, f"{value:.1f}", ha="center", fontsize=6.8,
                 color=PURPLE if keys[i]=="rpo" else MUTED)
offline.set(xlim=(-.35,3.35),ylim=(0,55),xticks=range(4),yticks=[0,40])
offline.set_xticklabels(["Actor","Teacher\nSFT","DPO","RPO"])
offline.tick_params(axis="x", length=0)
for tick,key in zip(offline.get_xticklabels(),keys):
    tick.set_color(PURPLE if key in ("dpo","rpo") else MUTED)
offline.plot([1.78,1.78,3.22,3.22],[-.30,-.36,-.36,-.30],
             color=PURPLE,lw=.55,transform=offline.get_xaxis_transform(),clip_on=False)
offline.text(2.5,-.575,"ADM-guided",transform=offline.get_xaxis_transform(),
             ha="center",va="center",fontsize=5.8,color=PURPLE)

online = chart(ORIGINS[2]+30,PLOT_Y,100,80,"OM2W success (%)")
rl_styles = [("outcome48", LIGHT_GRAY, "Outcome 48", "--"),
             ("outcome56", GRAY, "Outcome 56", "-"),
             ("gate-b", PURPLE, "+ ADM", "-")]
for key, color, label, ls in rl_styles:
    method = next(m for m in DATA["online"]["methods"] if m["id"]==key)
    raw = method["points"]
    smooth = method["smoothed"]
    online.scatter([p["iteration"] for p in raw], [100*p["successes"]/p["valid"] for p in raw],
                   s=8, color=color, alpha=.22, edgecolors="none")
    online.plot([p["iteration"] for p in smooth], [p["percent"] for p in smooth],
                color=color, ls=ls, lw=1.25, marker="o", ms=2.4, markeredgewidth=0)
online.set(xlim=(18, 62), ylim=(35, 50), xticks=[20, 40, 60], yticks=[40, 50])
online.set_xlabel("Iteration",fontsize=6.1,labelpad=2,color=MUTED)

# Iteration-90 checkpoint cohort, distinct from the learning-curve evaluations.
repetition = chart(ORIGINS[2]+168,PLOT_Y,100,80,"OM2W success (%)")
repetition_rows = EXTRA["repetition"]["data"]["methods"]
for key,color,_,_ in rl_styles:
    row = repetition_rows[key]
    x,y = row["repetition_incidence"],row["valid_success"]
    repetition.errorbar(x["percent"],y["percent"],
                        yerr=[[y["percent"]-y["ci95_percent"][0]],
                              [y["ci95_percent"][1]-y["percent"]]],
                        fmt="o",color=color,markersize=3.2,capsize=1.8,elinewidth=.9,
                        zorder=4 if key=="gate-b" else 3)
repetition.set(xlim=(13,47),ylim=(35,60),xticks=[20,30,40],yticks=[40,50])
repetition.set_xlabel("Repetition (%)",fontsize=6.1,labelpad=2,color=MUTED)
repetition.text(.98,.98,"Iter. 90",transform=repetition.transAxes,
                ha="right",va="top",fontsize=5.3,color=MUTED)
for dx, (_, color, label, ls) in zip([16, 102, 188], rl_styles):
    x = ORIGINS[2]+dx
    line([(x,LEGEND_Y),(x+14,LEGEND_Y)],color,1.1,dashed=ls=="--")
    text(x+21,LEGEND_Y,label,size=5.6,color=color,ha="left")

if render_options.right_legend:
    text(inference_legend_x,PLOT_Y+2,"Single actors",size=5.2,color=INK,weight="bold",ha="left")
    text(inference_legend_x,PLOT_Y+43,"SFT 4B + ADM",size=5.2,color=INK,weight="bold",ha="left")
    for i,point in enumerate(points):
        key = point["key"]
        color,marker = styles[key]
        hollow = key in ("best-of-q","pro-cua-prm","webdreamer","kev-4b")
        legend_y = PLOT_Y + (15 + 12*i if i < 2 else 57 + 12*(i-2))
        ax.scatter(inference_legend_x+5,legend_y,s=15,marker=marker,facecolors="none" if hollow else color,
                   edgecolors=color if hollow else "none",lw=.7)
        label = {"sft1":"SFT 4B","sft8b":"SFT 8B"}.get(key,point["label"])
        text(inference_legend_x+14,legend_y,label,size=5,color=color,ha="left")

# Preserve the complete source-data provenance while omitting every result artist
# from the illustration-only export. Keep component dimensions unchanged.
result_axes = (system_size,inference,offline,online,repetition)
if render_options.no_results:
    for result_axis in result_axes:
        fig.delaxes(result_axis)
    for artist in set(ax.get_children()) - illustration_artists:
        artist.remove()
    H = ROW_Y + 30 - TOP_CROP
    fig.set_size_inches(7.2, 7.2 * H / W)
    ax.set_ylim(H + TOP_CROP, TOP_CROP)
    result_axes = ()
    assert len(fig.axes) == 1

# Catch label collisions and clipped text at the final export dimensions.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
if render_options.phase_bands and not render_options.no_panel_headers:
    for heading, center_y in (("A. Training",54),("B. Applications",application_heading_y)):
        label = next(t for t in ax.texts if t.get_text() == heading)
        bounds = label.get_window_extent(renderer)
        # Require at least four scene units of padding above and below glyphs.
        lower = ax.transData.transform((0,center_y+9))[1]
        upper = ax.transData.transform((0,center_y-9))[1]
        assert bounds.y0 >= lower and bounds.y1 <= upper, heading

labels = list(ax.texts)
for a in result_axes:
    labels += [a.title, a.xaxis.label, *a.texts, *a.get_xticklabels(), *a.get_yticklabels()]
labels = [t for t in labels if t.get_visible() and t.get_text()]
for i, label in enumerate(labels):
    bounds = label.get_window_extent(renderer)
    assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1), label.get_text()
    for other in labels[:i]:
        assert not bounds.overlaps(other.get_window_extent(renderer)), (label.get_text(), other.get_text())

if render_options.preview_only:
    fig.savefig(ROOT / f"{OUTPUT_STEM}.png", dpi=300)
    plt.close(fig)
    print(f"Wrote fig/{OUTPUT_STEM}.png")
    raise SystemExit

metadata = {"CreationDate": None, "ModDate": None, "Creator": "plot_figure1_applications.py"}
fig.savefig(ROOT / f"{OUTPUT_STEM}.pdf", dpi=300, metadata=metadata)
svg = ROOT / f"{OUTPUT_STEM}.svg"
fig.savefig(svg, dpi=300, metadata={"Date": None, "Creator": "plot_figure1_applications.py"})
svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines())+"\n")
fig.savefig(ROOT / f"{OUTPUT_STEM}.png", dpi=300)
plt.close(fig)
result = {"source": "figure1_results.json",
          "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
          "inference": inference_data, "offline": OFFLINE_DATA, "online": DATA["online"],
          "additional_source":EXTRA_SOURCE_PATH.name,
          "additional_source_sha256":hashlib.sha256(EXTRA_SOURCE_PATH.read_bytes()).hexdigest(),
          "inference_system_size":{"points":system_points,"frontier":system_frontier},
          "repetition":{"iteration":90,"threshold":3,"methods":repetition_rows,
                        "uncertainty":EXTRA["repetition"]["data"]["uncertainty"]},
          "inference_illustration":{
              "flow":"Frozen actor -> candidate set -> ADM -> selected action",
              "candidate_examples":["a_t^(1)","a_t^(2)","...","a_t^(K)"],
              "arrows":3,"branch_connector_lines":0},
          "offline_illustration":{
              "flow":"ADM-selected preference pair -> RPO (DPO + SFT) -> actor policy",
              "rpo_reference":"https://arxiv.org/abs/2405.16436v3",
              "reference_title":"Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer",
              "interpretation":"A joint weighted combination of DPO and chosen-response SFT, not sequential training; the diagram shows components without specifying a coefficient.",
              "teacher_sft_baseline":"Separate gray comparison bar; not the SFT regularizer inside RPO."},
          "example": {"source": "figures/lookahead-examples.tex",
                      "task": TASK,
                      "state_id": "13d42f1f406049a59cfd502f38a90b09",
                      "screenshot_source": EXAMPLE_PATH.name,
                      "screenshot_sha256": hashlib.sha256(EXAMPLE_PATH.read_bytes()).hexdigest(),
                      "screenshot_viewport_pixels": list(EXAMPLE_VIEWPORT),
                      "action2_future_viewports_pixels": [list(v) for v in FUTURE_VIEWPORTS],
                      "actions": {"1": "Scroll down", "2": "Click Netflix payment options"},
                      "action_numbering": "Overview displays actions 1 and 2 followed by an ellipsis. Figure 5's displayed click/scroll rows 1/2 are relabeled 2/1; these are not saved candidate indices.",
                      "screenshot_annotation_relabel": {"1": "2", "2": "1"},
                      "future_states": "One next state per action: symbolic billing state for action 1 and a real answer-page screenshot from Figure 5 for action 2. Each is followed by an ellipsis representing the remaining lookahead.",
                      "future_summaries": {"1": "Billing", "2": "Answer page"},
                      "state_descriptions": {"1": ["Billing"], "2": ["Answer page"]},
                      "current_state_display_size": [112,54],
                      "future_state_display_size": [86,42],
                      "future_screenshot_display_width": 84.4,
                      "future_screenshot_layout": "The original screenshot border is cropped away; the page fills and clips to the browser body, preserving its aspect ratio with no inset border.",
                      "future_state_notation": "The future pages use plain-language captions under a Future states E heading; ellipses indicate later states, and the parent action identifies each branch.",
                      "evidence_collection": "Future-state miniatures labeled E feed the teacher T(s_t,A_t,E) through a bracket and one short arrow.",
                      "training_flow": "The teacher's chosen action labels the single right-angle Train ADM connection, without a separate target box. The ADM box contains only its name and f(s_t,A_t).",
                      "teacher_choice": "The teacher T(s_t,A_t,E) chooses a_t^(2), the payment-options action selected by the paper's H=1 and H=4 teachers. The action is shown on the training connection without a prose Chooses label."},
          "visual_semantics": {"selection_connectors": ACTION,
                               "evidence_and_supervision_connectors": SALMON,
                               "box_outlines": BOX_OUTLINES,
                               "box_outline_exceptions": [] if BOX_OUTLINES else ["Green browser frames"],
                               "notation_font":"STIX math; Albert Sans for prose and plot labels",
                               "task_style":"Bold Albert Sans with a slim mint accent, with the full task instruction retained",
                               "header_style":HEADER_DESCRIPTIONS[render_options.header_style],
                               "state_action_set_label_size":10,
                               "application_model_fill": ACTION,
                               "application_model_text":"white",
                               "actor_policy_badges":{"test_time_scaling":"Snowflake: frozen policy",
                                                      "offline_actor_improvement":"Fire: trained policy",
                                                      "online_actor_improvement_via_rl":"Fire: trained policy",
                                                      "rendering":"Emoji-style vector badges at the policy boxes' upper-right corners"},
                               "state_fill": C["STATE"],
                               "state_borders": STATE,
                               "state_icon_style": (
                                   "Browser miniatures with tab controls, address bars, and white page bodies. The mint current-state frame and salmon lookahead frames have role-colored outlines. Symbolic pages use a heading, sidebar, and content."
                                   if BOX_OUTLINES else
                                   "Browser miniatures with tab controls, address bars, and white page bodies. The mint current-state frame retains a thin green outline; salmon privileged lookahead frames stay borderless. Symbolic pages use a heading, sidebar, and content."),
                               "salmon_base": SALMON,
                               "lookahead_state_fill": tint(SALMON,.60),
                               "teacher_fill": tint(SALMON,.68),
                               "teacher_text": TEACHER_INK,
                               "teacher_output_fill": tint(SALMON,.60),
                               "action_style": (
                                   "Light-lavender cards with dark purple text and outlines; selected action 2 uses a moderately deeper lavender fill and heavier outline"
                                   if BOX_OUTLINES else
                                   "Light-lavender cards with dark purple text; selected action 2 uses a moderately deeper lavender fill, with no box outlines"),
                               "actor_style": (
                                   "White policy boxes with gray symbols and outlines" if BOX_OUTLINES else
                                   "Light neutral fills with gray policy symbols and no box outlines"),
                               "model_style": (
                                   "Dark-purple ADM fill and outline with white model notation" if BOX_OUTLINES else
                                   "Dark-purple ADM fill with white model notation and no box outline"),
                               "teacher_output_style": (
                                   "Brighter gallery-salmon fills and outlines with related dark-salmon text" if BOX_OUTLINES else
                                   "Brighter gallery-salmon fills with related dark-salmon text and no box outlines"),
                               "spacing": "A 430-unit canvas tightens vertical gaps without changing font, component, or plot sizes. Application headings sit at 198, schematics share a 250-unit centerline, plots start at 294, and legends sit at 420; plot blocks retain equal 238-unit outer widths.",
                               "top_connectors": "Current state and candidate actions feed the ADM, whose output is the selected action. Future-state branches feed a teacher; one right-angle Train ADM connection transfers its choice to the selector without a separate target box.",
                               "gallery_reference": GALLERY_STYLE,
                               "gallery_palette_blob_sha": "745611bc855a33fa5d99dc558b358b0fe3d0deba",
                               "gallery_typography_blob_sha": "0230bbc22b372c068223e4dd67e58ac8f41449aa",
                               "top_arrow_colors_verified": sorted(top_arrow_colors)},
          "notes": (
              "Fully bordered comparison, exported separately; the current application-first figure is unchanged. Header teacher choices are schematic."
              if BOX_OUTLINES else
              "Separate application-first layout; current Figure 1 is unchanged. Header teacher choices are schematic.")}
if render_options.top_background:
    result["visual_semantics"]["top_background"] = {
        "fill":tint(PURPLE,.91),"bounds":[10,7,880,185],
        "style":"Pale lavender rounded backdrop without an outline; the application panels retain white backgrounds",
    }
if CLEAR_TRAINING:
    result["example"].update({
        "future_summaries":{"1":"More billing text","2":"Payment methods"},
        "state_descriptions":{"1":["More billing text"],"2":["Payment methods"]},
        "training_flow":"The top panel ends at the ADM. A dashed salmon training connection carries the explicitly named Teacher label a_t^(2); deployment action selection appears in application (a).",
        "teacher_choice":"The teacher prefers action 2 because its future state exposes payment-method information; the other branch exposes more billing text. Its choice is labeled on the training connection without a separate output box.",
    })
    result["visual_semantics"]["top_connectors"] = "Solid arrows carry current state, candidates, and future-state inputs. A dashed salmon connection labeled Train ADM carries the Teacher label to the selector. There is no top-panel selected-action output."
    result["visual_semantics"]["top_story"] = "Learning the ADM from future-informed teacher labels; deployment is illustrated in the applications."
    result["notes"] = "Separate training-focused overview variant; the selected paper figure is preserved. Teacher choices are schematic."
if render_options.phase_overview:
    result["example"].update({
        "future_summaries":{"1":"Billing","2":"Payment methods"},
        "state_descriptions":{"1":["Billing"],"2":["Payment methods"]},
        "future_state_notation":"Future states E_t form the privileged evidence for decision t.",
        "evidence_collection":"The teacher T(s_t,A_t,E_t) observes action-conditioned future states E_t, while the ADM f(s_t,A_t) receives pre-execution inputs.",
        "training_flow":"A solid prediction arrow ends at a_hat_t without a redundant Selected action caption. A dashed salmon training connection carries Label a_t^(2) from the teacher to the ADM.",
    })
    result["visual_semantics"]["section_labels"] = ["Training","Applications"]
    result["visual_semantics"]["section_divider"] = {"bounds":[24,195,884,195],"color":BORDER,"line_width":.85,"style":"One full-width horizontal rule separates training from applications without a background fill"}
    result["visual_semantics"]["top_connectors"] = "Solid arrows carry model inputs and its selected-action prediction. A dashed salmon connection carries the teacher's action label without a redundant Train ADM caption. Future evidence is consistently indexed E_t."
    result["visual_semantics"]["top_story"] = "The upper section shows training inputs, prediction, and teacher supervision; the lower section shows uses of the trained ADM."
    result["visual_semantics"]["spacing"] = "Section headings add 22 vertical units; all application components and plots retain their physical sizes and relative spacing."
    result["notes"] = "Separate phase-labeled overview with the ADM prediction restored; the selected paper figure is preserved. Teacher choices are schematic."
if render_options.refined_overview:
    result["example"].update({
        "future_state_display_size":[112,48],
        "future_screenshot_display_width":110.4,
        "evidence_collection":"Action-conditioned future browsers are centered directly below their actions. Their E_t bracket connects horizontally to the teacher T(s_t,A_t,E_t).",
        "training_flow":"The teacher and ADM f_phi(s_t,A_t) share a vertical centerline. A short dashed upward arrow carries Label a_t^(2); the solid prediction arrow ends at a_hat_t without an extra caption.",
    })
    result["visual_semantics"].update({
        "header_style":"The selected oversized vivid-violet title is retained, with its baseline raised to give the ADM more breathing room",
        "state_action_set_label_size":7.5,
        "section_label_style":{"font_size":8,"weight":"bold","color":MUTED},
        "section_divider":{"bounds":[24,207,884,207],"color":BORDER,"line_width":.85,"style":"A single horizontal rule with equal breathing room above and below separates the phases"},
        "top_alignment":{"selection_centerline":88,"future_browser_centerline":162,"teacher_adm_centerline":694,"action_future_centers":[231,389]},
        "action_style":"Light-lavender cards with consistent thin purple outlines; selected action 2 is emphasized through fill alone",
        "top_connectors":"Solid purple prediction arrows; solid salmon future-evidence arrows; one short vertical dashed salmon supervision arrow from teacher to ADM",
        "spacing":"A 464-unit canvas retains all application component, plot, and font sizes. Application headings sit at 232, schematics share a 284-unit centerline, plots start at 328, and legends sit at 454. Application-column separators are lighter than the phase divider.",
    })
    result["notes"] = "Separate refined comparison; earlier figure variants and the selected paper figure are preserved. Teacher choices are schematic; all quantitative results are unchanged."
if render_options.redesigned_overview:
    result["offline_illustration"]["rpo_layout"] = "RPO and its DPO + SFT components share one two-line module, avoiding a floating label"
    result["example"].update({
        "current_state_display_size":[176,78],
        "screenshot_annotation_relabel":{"1":"a_t^(2)","2":"a_t^(1)"},
        "screenshot_annotation_style":"Current-state badges use the same full a_t^(1) and a_t^(2) symbols as the action cards. The a_t^(2) tag sits beside the payment-options link with a short leader so it does not obscure the page heading.",
        "future_state_display_size":[68,25],
        "future_screenshot_display_width":66.4,
        "future_states":"Only s_{t+1} is illustrated for candidates 1, 2, ..., K. Horizontal ellipses extend the candidate dimension and vertical ellipses replace all later future steps. Action 2 uses a real screenshot; the other pages are schematic.",
        "evidence_collection":"Candidates connect upward to a lookahead view with explicit action-column headers and an s_{t+1} row. Vertical dots indicate further lookahead; the privileged evidence E_t feeds the teacher.",
        "training_flow":"The teacher T(s_t,A_t,E_t) outputs the hard label a_t^+=a_t^(2). The student f_phi(s_t,A_t) outputs a K-class p_phi distribution; both label and probabilities feed L_ADM=-log p_phi(a_t^+). Argmax of p_phi produces a_hat_t on a separate prediction branch.",
        "teacher_choice":"The teacher prefers action 2 because its future state exposes payment-method information. The explicit hard-label box a_t^+=a_t^(2) feeds the loss rather than serving as a student input.",
    })
    result["visual_semantics"].update({
        "header_style":"A larger vivid-violet Action Decision Models title occupies two equal-emphasis lines beneath the teacher and directly above the wider student card. Centered phase headings retain their flanking rules.",
        "state_action_set_label_size":{"state":8.6,"actions":7},
        "section_label_style":{"font_size":11,"weight":"bold","color":INK,"center_x":450,"training_y":54,"applications_y":278,"flanking_rules":True},
        "section_divider":{"y":278,"segments":[[24,366],[534,884]],"color":BORDER,"line_width":.75},
        "training_rows":{"teacher_centerline":92,"student_centerline":229,"teacher_center_x":608,"student_center_x":608,"action_future_centers":[263,350,438]},
        "privileged_evidence_group":{"bounds":[218,80,260,80],"fill":tint(SALMON,.94),"outline":False,"columns":["a_t^(1)","a_t^(2)","...","a_t^(K)"],"rows":["s_{t+1}","..."],"ellipsis_dimensions":["candidate action","future time"]},
        "model_emphasis":{"student_bounds":[492,195,232,68],"teacher_bounds":[547,75,122,34],"student_name":"ADM","title_font_size":15,"student_name_font_size":11.5,"student_formula_font_size":14.3,"teacher_formula_font_size":8.3},
        "teacher_fill":tint(SALMON,.88),
        "teacher_output_fill":tint(SALMON,.90),
        "action_style":"Stacked action symbols and concrete descriptions for candidates 1 and 2, followed by an ellipsis and an explicit a_t^(K). Matching thin purple outlines; action 2 has a deeper lavender fill.",
        "top_connectors":"Solid salmon action-to-next-state and evidence-to-teacher arrows; salmon teacher label and purple student distribution converge on the loss; a separate purple argmax path ends at the selected action",
        "student_distribution":{"notation":"p_phi","bounds":[750,213,134,32],"classes":["a_t^(1)","a_t^(2)","...","a_t^(K)"],"visualization":"Compact p_phi node; class bars and repeated action labels are omitted","values":"No numerical probabilities displayed","output":"Vertical argmax branch -> a_hat_t"},
        "rightmost_stack":{"center_x":817,"right_edge":884,"teacher_label_bounds":[750,78,134,28],"loss_bounds":[750,137,134,28],"probability_bounds":[750,213,134,32],"selected_action_bounds":[800,252,34,22]},
        "supervision":{"teacher_target":"a_t^+=a_t^(2)","student_input":"p_phi","loss":"L_ADM=-log p_phi(a_t^+)","interpretation":"Categorical cross-entropy with a hard teacher label; conditioning on s_t,A_t is implicit in this compact notation"},
        "application_structure":{"diagram_bands":[[14,316,280,50],[310,316,280,50],[606,316,280,50]],"fill":"#F7F7FA","vertical_separators":False},
        "spacing":"The preferred 480-unit layout, large current state, prominent student and all application positions are retained. Later future-state rows are replaced with dots; a compact p_phi node replaces the probability chart without changing its connections to the loss or argmax.",
        "viewport":{"top_crop":TOP_CROP,"visible_height":H,"scene_height":H+TOP_CROP},
    })
    result["notes"] = "Separate redesigned comparison; previous variants and the selected paper figure are preserved. Teacher choices are schematic; quantitative results are unchanged."
if render_options.swap_teacher_student and not render_options.minimal_concept:
    result["visual_semantics"]["row_order"] = "Student above teacher"
    result["visual_semantics"]["swapped_layout"] = swapped_layout
    result["visual_semantics"]["action_card_fill"] = tint(PURPLE,.90)
    result["visual_semantics"]["action_style"] = "Candidate action cards use the same pale-lavender background; existing label and outline colors are preserved."
    result["visual_semantics"]["phase_bands"] = {"enabled":render_options.phase_bands,"fill":"#F3F4F6","scope":"Heading strips only; Training and Applications share one neutral color"}
    if render_options.phase_bands:
        result["visual_semantics"]["section_labels"] = ["A. Training","B. Applications"]
        result["visual_semantics"]["phase_heading_font_size"] = 9
        result["visual_semantics"]["phase_heading_font_family"] = "Albert Sans"
        result["visual_semantics"]["selected_action_font_size"] = 10.5
        result["visual_semantics"]["phase_band_height"] = 26
        result["visual_semantics"]["training_policy_and_future_state_font_size"] = 10.5
        result["visual_semantics"]["application_policy_font_size"] = 9
        result["visual_semantics"]["argmax_font_size"] = 7
        result["visual_semantics"]["application_arrow_color"] = GRAY
        result["visual_semantics"]["application_labels"] = ["(1) Test-time scaling","(2) Offline actor improvement","(3) Online improvement via RL"]
    result["visual_semantics"]["top_alignment"] = {"teacher_centerline":swapped_layout["teacher_centerline"],"student_centerline":swapped_layout["student_centerline"],"teacher_center_x":608,"student_center_x":608,"action_future_centers":[263,350,438]}
    result["visual_semantics"]["training_rows"] = dict(result["visual_semantics"]["top_alignment"])
    result["visual_semantics"]["header_style"] = "Student-first preview with a single-line vivid-violet title above the student; teacher and future evidence are below."
    result["visual_semantics"]["privileged_evidence_group"]["bounds"] = [218,190,260,80]
    result["visual_semantics"]["model_emphasis"].update({"student_bounds":[492,93,232,68],"teacher_bounds":swapped_layout["teacher_bounds"],"teacher_formula_font_size":swapped_layout["teacher_formula_font_size"],"title_font_size":15})
    result["visual_semantics"]["student_distribution"]["bounds"] = [750,111,134,32]
    result["visual_semantics"]["rightmost_stack"].update({"teacher_label_bounds":swapped_layout["teacher_label_bounds"],"loss_bounds":swapped_layout["loss_bounds"],"probability_bounds":[750,111,134,32],"selected_action_bounds":[800,75,34,22]})
    result["visual_semantics"]["spacing"] = "The larger teacher and hard-label boxes align with the future evidence at y=230, filling the lower-right space. A larger loss block sits between the label and student probabilities. The ADM and all application positions, fonts and plots remain unchanged."
    result["example"]["evidence_collection"] = "Candidates connect downward to future states s_{t+1}; the teacher receives this evidence."
    result["example"]["screenshot_annotation_relabel"] = {"1":"2","2":"1"}
    result["example"]["screenshot_annotation_style"] = "Plain 1 and 2 badges are retained in the current-state screenshot: 1 corresponds to Scroll down and 2 to Click payment options."
    result["notes"] = "Separate student-first preview; selected paper Figure 1 and quantitative results are unchanged."
if render_options.simplify_layout and not render_options.minimal_concept:
    result["example"]["teacher_choice"] = "The teacher's hard target is shown directly as action a_t^(2), without repeating a_t^+=a_t^(2)."
    result["example"]["training_flow"] = "The teacher T(s_t,A_t,E_t) outputs action 2 as a hard label. The student f_phi(s_t,A_t) outputs a miniature K-class probability chart with action labels 1, 2, ..., K and no p notation. A thin neutral comparison line links the chart and hard label, with CE loss annotated beside it; argmax branches upward to a_hat_t. Chart heights are illustrative, not measured probabilities."
    result["example"]["screenshot_action_colors"] = swapped_layout["action_colors"]
    result["visual_semantics"]["section_divider"]["y"] = 271
    result["visual_semantics"]["section_label_style"].update({"training_y":54,"applications_y":271})
    result["visual_semantics"]["model_emphasis"].update({"student_bounds":[492,109,232,68],"title_y":84})
    result["visual_semantics"]["state_action_set_label_size"] = {"state":8.6,"actions":8.6}
    result["visual_semantics"]["state_action_header_baseline"] = swapped_layout["state_action_header_baseline"]
    result["visual_semantics"]["component_label_font_size"] = {"state":8.6,"actions":8.6,"privileged_lookahead":8.6,"teacher":8.6,"loss":8.6}
    result["visual_semantics"]["student_distribution"].update({"notation":None,"bounds":swapped_layout["distribution_bounds"],"label_center":None,"visualization":"Miniature probability bars labeled 1, 2, ..., K, without a p symbol or enclosing box","values":"Schematic bar heights, not measured probabilities","schematic_bars":swapped_layout["distribution_bars"]})
    result["visual_semantics"]["rightmost_stack"].update({"teacher_label_bounds":swapped_layout["teacher_label_bounds"],"teacher_label_center":[817,221],"probability_bounds":swapped_layout["distribution_bounds"],"selected_action_bounds":swapped_layout["selected_action_bounds"],"selected_action_font_size":swapped_layout["selected_action_font_size"],"loss_bounds":None,"loss_label_position":swapped_layout["loss_label_position"]})
    result["visual_semantics"]["privileged_evidence_group"].update({"bounds":[218,186,260,62],"fill":None,"label":"Privileged lookahead E_t","visible_column_headers":False,"column_identity":"Reused from the aligned action labels and their downward arrows"})
    result["visual_semantics"]["supervision"].update({"displayed_target":"a_t^(2)","displayed_student_output":"Miniature K-class probability chart","displayed_loss":"CE loss","loss_style":"Unboxed neutral annotation beside one thin comparison line; no arrowheads","comparison_line":swapped_layout["loss_comparison_line"],"formal_loss":"Unchanged categorical cross-entropy objective"})
    result["visual_semantics"]["application_structure"].update({"diagram_bands":None,"fill":None})
    result["visual_semantics"]["action_style"] = "The action symbols a_t^(1), a_t^(2), ..., a_t^(K) share y=100 above the two concrete action cards. Action 1's label and 1.2-point card outline use the screenshot marker's light purple; action 2 uses its matching dark purple. Card fills are tints of these same colors; a_t^(K) is a plain continuation symbol."
    result["visual_semantics"]["spacing"] = "Same 480-unit height and application plot sizes. State and Actions share baseline y=75 and an 8.6-point font. Training stays at y=54 and the model title at y=84, preserving their breathing room. The loss is a neutral, unboxed annotation beside a fine comparison line; the selected action remains prominent above the miniature probability chart. All application diagrams and plots retain their original positions and type sizes."
    result["notes"] = "Separate simplified student-first comparison; the current swapped figure is preserved. All application workflows and numerical results are retained."
if render_options.minimal_overview:
    result["example"] = {"kind":"Minimal conceptual training diagram",**minimal_training_layout}
    result["visual_semantics"] = {
        "fonts":{"prose":"Albert Sans","math":"STIX"},
        "colors":{"student":ACTION,"teacher_and_evidence":SALMON},
        "training":"Compact shared inputs, privileged evidence, teacher label, student probabilities, cross-entropy and argmax",
        "applications":"All three detailed application workflows and all five quantitative plots are reused from the full figure without changing sizes, fonts or data",
        "canvas":{"width":W,"height":H,"top_crop":TOP_CROP},
        "application_positions":{"headings":HEADING_Y,"diagrams":ROW_Y,"plots":PLOT_Y,"legends":LEGEND_Y},
        "omitted":["Netflix task example","screenshots","detailed lookahead grid","full loss equation"],
    }
    result["notes"] = "Compact training alternative retaining the complete application detail; the full overview is unchanged."
    minimal_training_layout["applications"] = {
        "workflows":["test-time scaling","offline actor improvement via RPO (DPO + SFT)","online actor improvement via ADM reward"],
        "plots":["system size","test-time compute","offline success","RL training","repetition robustness"],
        "result_provenance":f"{OUTPUT_STEM}_results.json",
    }
    (ROOT / f"{OUTPUT_STEM}_layout.json").write_text(json.dumps(minimal_training_layout,indent=2)+"\n")
if render_options.right_legend:
    result["visual_semantics"]["right_legend"] = {"inference_method_labels":"Immediately right of the inference plots", "groups":["Single actors", "SFT 4B + ADM"],"legend_x":inference_legend_x,"inference_plot_width":inference_plot_width,"offline_panel_x":offline_plot_x,"offline_panel_width":offline_plot_width,"online_panel_width":100,"rl_legend":"Below online plots"}
if render_options.no_results:
    result["visual_semantics"]["quantitative_results_displayed"] = False
    result["visual_semantics"]["viewport"] = {"top_crop":TOP_CROP,"visible_height":H,"scene_height":H+TOP_CROP}
    result["visual_semantics"]["spacing"] = "Training and all three application illustrations retain their positions and sizes; the canvas ends below the application workflows with compact bottom padding."
    result["notes"] = "Separate illustration-only version. Quantitative plots and legends are omitted; numerical source data is retained here for provenance. The selected paper figure and results-bearing preview are unchanged."
if render_options.no_panel_headers:
    result["visual_semantics"]["section_labels"] = []
    result["visual_semantics"]["application_labels"] = ["(a) Test-time scaling","(b) Offline actor improvement","(c) Online actor improvement via RL"]
    result["visual_semantics"]["phase_bands"] = {"enabled":False,"fill":None,"scope":"Phase headings and their decorations omitted"}
    result["visual_semantics"]["section_label_style"]["visible"] = False
    result["visual_semantics"]["section_divider"]["visible"] = False
    result["visual_semantics"]["panel_headers"] = {"visible":False,"scope":"Training and Applications phase headings and their bands or flanking rules","application_headers_visible":True,"retained":"The (a), (b), (c) application titles, Action Decision Models title, component labels, plot titles and legends"}
    result["visual_semantics"]["viewport"] = {"top_crop":TOP_CROP,"visible_height":H,"scene_height":H+TOP_CROP}
    result["visual_semantics"]["text_weights"] = {"task_instruction":"normal","application_titles":"normal"}
    result["visual_semantics"]["spacing"] = "Training and Applications phase headings and their decorations are omitted; the (a), (b), (c) application titles are retained in regular weight, as is the task instruction. Applications move upward by 28 scene units and 16 units of unused top margin are trimmed. In the right-legend version, the two inference plots narrow from 112 to 94 units and their legend moves left, widening the middle offline plot from 168 to 208 units. Heights, font sizes, other diagram dimensions and numerical results are unchanged."
    result["notes"] = "Separate comparison without phase headers, retaining all three application titles; the selected paper figure and fully headed variants are preserved. Numerical results are unchanged."
if NEUTRAL_DIAGRAM:
    result["visual_semantics"].update({
        "selection_connectors":GRAY,
        "evidence_and_supervision_connectors":GRAY,
        "top_connectors":"Gray horizontal arrows align current state/actions with the ADM and the salmon lookahead box with the teacher. Their f_phi(s_t,A_t) and T(s_t,A_t,E_t) formulas retain the distinct conditioning. No shared-input frames or branching connectors are displayed; the purple screenshot scroll arrow is preserved.",
        "state_icon_style":"Browser frames and symbolic page contents stay green. State/candidate-action notation and continuation dots use policy gray; lookahead and teacher notation use dark salmon; action outputs keep their purple or salmon colors.",
        "input_notation_color":GRAY,
        "privileged_notation_color":TEACHER_INK,
        "output_notation_colors":{"selected_action":ACTION,"teacher_action":TEACHER_INK,"model_on_dark_fill":"white"},
        "lookahead_state_fill":tint(C["STATE"],.65),
        "future_state_color":STATE,
        "screenshot_scroll_arrow":{"marker":"1","color":PURPLE,"preserved":True},
        "teacher_output_fill":tint(SALMON,.90),
        "teacher_output_style":"The hard teacher action sits in a light salmon rounded box, matching the single-column conceptual figure.",
        "input_layout":{"shared_input_boxes_displayed":False,"lookahead_box_displayed":True,"extra_group_captions_displayed":False,"labels":["State s_t","Actions A_t","Privileged lookahead E_t"],"student_inputs":["s_t","A_t"],"teacher_inputs":["s_t","A_t","E_t"],"to_student":[[460,143],[492,143]],"to_teacher":[[478,221],[527,221]],"branch_connector_lines":0},
        "action_style":"Candidate notation is policy gray; card outlines retain their purple screenshot-marker colors, with pale-lavender fills.",
        "task_style":"Full three-line task instruction with the mint accent below the current state; task context is shared by the ADM and teacher.",
        "task_caption":{"position":[36,219],"font_size":8.5,"lines":3,"weight":"normal","shared_context":True},
    })
    result["visual_semantics"]["privileged_evidence_group"].update({"bounds":[218,192,260,56],"fill":tint(SALMON,.96),"outline":SALMON,"line_width":.5,"frame_displayed":True,"heading_inside":False})
    result["visual_semantics"]["model_emphasis"].update({"student_bounds":[492,109,232,68],"student_center_x":608,"title_center_x":608,"title_color":ACTION})
    for key in ("top_alignment","training_rows"):
        result["visual_semantics"][key]["student_center_x"] = 608
if render_options.conceptual_overview:
    result["figure_purpose"] = "Conceptual Figure 1: lookahead-supervised ADM training and three deployment applications"
    result["reproduction_command"] = "python fig/plot_figure1_applications.py --conceptual-overview"
    result["visual_semantics"]["spacing"] = "A compact 288-unit-high conceptual overview keeps the training/model diagram and the three labeled application workflows at their existing dimensions and font sizes. Quantitative plots, axes and method legends are omitted; the canvas ends beneath the workflows."
    result["notes"] = "New conceptual Figure 1, exported separately. The existing figure with quantitative results and the selected paper figure are preserved. Source numerical data remains in this manifest for provenance but is not displayed."
if render_options.minimal_concept:
    result["figure_purpose"] = "Minimal ADM concept and three application workflows"
    result["reproduction_command"] = "python fig/plot_figure1_applications.py --minimal-concept"
    result["example"] = {"kind":"Abstract concept without a task example",**minimal_concept_layout}
    result["visual_semantics"] = {
        "fonts":{"prose":"Albert Sans","math":"STIX"},
        "colors":{"adm":ACTION,"teacher_and_lookahead":SALMON,"policy":GRAY},
        "training":minimal_concept_layout,
        "applications":{"labels":["(a) Test-time scaling","(b) Offline actor improvement","(c) Online actor improvement via RL"],
                        "heading_y":HEADING_Y,"workflow_y":ROW_Y,
                        "style":"Same workflows, dimensions, font sizes, policy badges and regular-weight titles as the detailed conceptual Figure 1"},
        "quantitative_results_displayed":False,
        "viewport":{"top_crop":TOP_CROP,"visible_height":H,"scene_height":H+TOP_CROP},
    }
    result["notes"] = "Separate minimal conceptual alternative. The teacher receives the shared inputs plus future evidence and supplies action labels to train the ADM. The ADM uses the shared inputs to select an action. All existing figures are preserved."
(ROOT / f"{OUTPUT_STEM}_results.json").write_text(json.dumps(result, indent=2)+"\n")
print(f"Wrote fig/{OUTPUT_STEM}.{{pdf,svg,png}} and {OUTPUT_STEM}_results.json")
