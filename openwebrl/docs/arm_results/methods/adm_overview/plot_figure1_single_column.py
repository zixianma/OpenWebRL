"""Single-column ADM concept with symbolic states, actions and three applications.

Run with the paper's plotting environment. Exports PDF, SVG, PNG and layout JSON.
The 3.28125-inch width matches the CVPR template's actual column width.
"""

import json
from pathlib import Path
import sys
_style_root = next((parent for parent in Path(__file__).resolve().parents
                    if (parent / 'paper_plot_style.py').is_file()), None)
if _style_root is not None:
    sys.path.insert(0, str(_style_root))
    import paper_plot_style

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parent
COLORS = json.loads((ROOT / 'figure1_sources.json').read_text())['colors']
for font in (ROOT / 'fonts/albert-sans').glob('*.ttf'):
    font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':'Albert Sans','font.sans-serif':['Albert Sans'],
                     'mathtext.fontset':'stix','mathtext.default':'it',
                     'pdf.fonttype':42,'svg.fonttype':'path',
                     'svg.hashsalt':'adm-single-column-concept'})
INK, BORDER = COLORS['INK'], COLORS['BORDER']
PURPLE, ADM = COLORS['ADM_DEFAULT'], COLORS['ACTION_INK']
SALMON, TEACHER = COLORS['LOOKAHEAD_TEACHER'], '#944735'
STATE, POLICY = COLORS['STATE_BORDER'], COLORS['BASELINE_DARK']
W, H = 236.25, 206  # PDF points: (6.875in - 0.3125in) / 2.
fig = plt.figure(figsize=(W/72,H/72),facecolor='white')
ax = fig.add_axes([0,0,1,1])
ax.set(xlim=(0,W),ylim=(H,0))
ax.axis('off')


def tint(color,white):
    rgb=[int(color.lstrip('#')[i:i+2],16) for i in (0,2,4)]
    return '#'+''.join(f'{round(v*(1-white)+255*white):02x}' for v in rgb)


def text(x,y,label,size=7.5,color=INK,ha='center',**kwargs):
    return ax.text(x,y,label,fontsize=size,color=color,ha=ha,va='center',**kwargs)


def box(x,y,w,h,fill='white',edge=BORDER,lw=.6,radius=3):
    patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                        facecolor=fill,edgecolor=edge,linewidth=lw)
    ax.add_patch(patch)
    return patch


def line(points,color=BORDER,lw=.6,zorder=2):
    ax.plot(*zip(*points),color=color,lw=lw,solid_capstyle='round',zorder=zorder)


def arrow(start,end,color=POLICY,lw=.7):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=4,
                                linewidth=lw,color=color,shrinkA=0,shrinkB=0))


def browser(x,y,w,h,color,label=None):
    """Use the same symbolic browser for current and privileged future states."""
    chrome=max(3,min(4.5,h*.20))
    box(x,y,w,h,tint(color,.82),color,lw=.6,radius=2)
    box(x+.6,y+chrome,w-1.2,h-chrome-.6,'white','none',lw=0,radius=1)
    line([(x+.6,y+chrome),(x+w-.6,y+chrome)],color,.35)
    for dx in (.08,.16,.24):
        ax.add_patch(Circle((x+w*dx,y+chrome*.5),.38,facecolor=color,edgecolor='none'))
    box(x+w*.35,y+chrome*.28,w*.58,chrome*.45,'white','none',lw=0,radius=.5)
    if label is not None:
        text(x+w/2,y+chrome+(h-chrome-.6)/2,label,size=7.1,color=POLICY)
        return
    box(x+w*.10,y+chrome+2,w*.16,h-chrome-4,tint(color,.75),'none',lw=0,radius=.4)
    for fraction,length in ((.18,.58),(.48,.48),(.77,.54)):
        yy=y+chrome+2+(h-chrome-4)*fraction
        line([(x+w*.34,yy),(x+w*(.34+length),yy)],tint(color,.25),.65)


def policy_badge(x,y,trained=False):
    ax.add_patch(Circle((x,y),3.1,facecolor='white',edgecolor='none',zorder=5))
    if trained:
        flame=np.array([(0,-3),(.8,-1.2),(1.8,-2),(2.5,.4),(1.7,2.3),
                        (0,2.8),(-2,1.6),(-2.1,-.5),(-.9,-1.7),(-1,0)])
        ax.add_patch(Polygon(flame+np.array([x,y]),facecolor='#FF9D00',
                             edgecolor='none',zorder=6))
        core=np.array([(0,-.1),(1,1.1),(.6,2.1),(-.6,2.1),(-1,1.1)])
        ax.add_patch(Polygon(core+np.array([x,y]),facecolor='#FFE078',
                             edgecolor='none',zorder=7))
    else:
        for angle in np.arange(0,360,60):
            u=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
            n=np.array([-u[1],u[0]])
            center=np.array([x,y])
            line([center,center+u*2.5],'#73BCEA',.55,zorder=6)
            for sign in (-1,1):
                joint=center+u*1.35
                line([joint,joint+u*.65+sign*n*.65],'#73BCEA',.45,zorder=6)


def pill(x,cy,w,label,actor=False,trained=None,size=9,height=18,label_color=None):
    color=POLICY if actor else ADM
    box(x,cy-height/2,w,height,'white' if actor else tint(PURPLE,.90),color,radius=2.5)
    text(x+w/2,cy,label,size=size,color=label_color or color)
    if trained is not None:
        policy_badge(x+w-.5,cy-height/2,trained)


def model(x,cy,width=28,height=18):
    box(x,cy-height/2,width,height,ADM,ADM,radius=2.5)
    text(x+width/2,cy,r'$f_\phi$',size=9.5,color='white')


text(W/2,10,'Action Decision Models',size=11.2,color=ADM,weight='bold')
# Current state and individual actions stay separate from privileged lookahead.
text(23,25,r'State $s_t$',size=7.3,color=POLICY)
text(71,25,r'Actions $\mathcal{A}_t$',size=7.3,color=POLICY)
browser(11,35,24,20,STATE)
box(39,36,26,19,tint(PURPLE,.95),PURPLE,lw=.6,radius=2)
box(68,36,26,19,tint(PURPLE,.90),ADM,lw=.7,radius=2)
text(52,45.5,r'$a_1$  Scroll',size=6,color=PURPLE)
text(81,45.5,r'$a_2$  Click',size=6,color=ADM)
text(100,45.5,r'$\cdots$',size=7,color=POLICY)

# Two aligned model rows, with only eight points between their cards.
box(119,32,50,26,ADM,ADM,lw=.55,radius=2)
text(144,39.5,'ADM',size=7.7,color='white',weight='bold')
text(144,51.5,r'$f_\phi(s_t,\mathcal{A}_t)$',size=8.6,color='white')
arrow((105,45),(119,45),POLICY,lw=.6)
arrow((169,45),(177,45),POLICY,lw=.65)

# Selection stays on the forward path; the loss branches from probabilities.
line([(177,49),(202,49)],BORDER,.45)
for x,height,color in [(178,7,PURPLE),(186,16,ADM),(198,4,tint(PURPLE,.25))]:
    box(x,49-height,4,height,color,'none',lw=0,radius=.4)
for x,label,color in [(180,r'$a_1$',PURPLE),(188,r'$a_2$',ADM),(200,r'$a_K$',POLICY)]:
    text(x,55.5,label,size=5.3,color=color)
text(194,43,r'$\cdots$',size=4.6,color=POLICY)
box(215,37,17,16,tint(PURPLE,.90),ADM,lw=.55,radius=2)
text(223.5,45,r'$\hat a_t$',size=10.5,color=ADM)
arrow((207,45),(215,45),POLICY,lw=.55)
text(220,32,'argmax',size=5.3,color=POLICY)

# Each candidate owns a vertical future-state branch; dots extend future time.
text(21,77,'Lookahead\n'+r'$\mathcal{E}_t$',size=6.3,color=TEACHER,
     linespacing=1.0)
box(39,66,66,24,tint(SALMON,.96),SALMON,lw=.5,radius=2)
browser(40.5,71,23,13,STATE,label=r'$s_{t+1}$')
browser(69.5,71,23,13,STATE,label=r'$s_{t+1}$')
text(100,77.5,r'$\cdots$',size=7,color=POLICY)
for x in (52,81):
    arrow((x,55),(x,70),POLICY,lw=.65)
    text(x,87,r'$\vdots$',size=5.2,color=POLICY)
box(119,66,50,24,tint(SALMON,.94),SALMON,lw=.5,radius=2)
text(144,72.1,'Teacher',size=7.1,color=TEACHER)
text(144,83.5,r'$\mathcal{T}(s_t,\mathcal{A}_t,\mathcal{E}_t)$',size=7.7,color=TEACHER)
arrow((105,78),(119,78),POLICY,lw=.65)

# Cross-entropy compares the probability chart with a hard teacher label.
arrow((169,78),(178,78),POLICY,lw=.65)
box(178,72,38,12,tint(SALMON,.90),SALMON,lw=.5,radius=1.5)
text(197,78,r'$a_2$  Click',size=6.2,color=TEACHER)
arrow((197,59.5),(197,61),POLICY,lw=.55)
arrow((197,72),(197,69),POLICY,lw=.55)
text(197,65,'CE loss',size=6.6,color=POLICY)

# Three compact application rows retain the actual policy-improvement mechanisms.
APP_X, APP_W = [8,51,120,202], [28,54,64,26]
APP_ARROWS = [(APP_X[i]+APP_W[i]+4,APP_X[i+1]-3) for i in range(3)]


def application_arrows(cy):
    for left,right in APP_ARROWS:
        arrow((left,cy),(right,cy))


for y,label in [(102,'(a) Test-time scaling'),(137,'(b) Offline actor improvement'),
                (172,'(c) Online actor improvement via RL')]:
    text(8,y,label,size=7.5,ha='left')

cy=120
pill(APP_X[0],cy,APP_W[0],r'$\pi_\theta$',actor=True,trained=False)
pill(APP_X[1],cy,APP_W[1],r'$a_1\;a_2\;\cdots\;a_K$',size=8,label_color=ADM)
model(APP_X[2],cy,width=APP_W[2],height=22)
pill(APP_X[3],cy,APP_W[3],r'$\hat a_t$')
application_arrows(cy)

cy=155
model(APP_X[0],cy,width=APP_W[0])
pill(APP_X[1],cy,APP_W[1],r'$\hat a_t\succ a_t^-$',size=8)
box(APP_X[2],cy-11,APP_W[2],22,tint(PURPLE,.90),ADM,radius=2.5)
text(152,cy-4.5,'RPO',size=6.8,color=ADM,weight='bold')
text(152,cy+5.2,'DPO + SFT',size=6.3,color=ADM)
pill(APP_X[3],cy,APP_W[3],r'$\pi_\theta$',actor=True,trained=True)
application_arrows(cy)

cy=190
model(APP_X[0],cy,width=APP_W[0])
pill(APP_X[1],cy,APP_W[1],r'$\hat a_t$')
pill(APP_X[2],cy,APP_W[2],'+ ADM reward',size=7.5,height=22)
pill(APP_X[3],cy,APP_W[3],r'$\pi_\theta$',actor=True,trained=True)
application_arrows(cy)

# Check actual single-column typography, not a reduced preview of a wide drawing.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for i,label in enumerate(ax.texts):
    bounds=label.get_window_extent(renderer)
    assert fig.bbox.contains(*bounds.p0) and fig.bbox.contains(*bounds.p1),label.get_text()
    for earlier in list(ax.texts)[:i]:
        assert not bounds.overlaps(earlier.get_window_extent(renderer)),(label.get_text(),earlier.get_text())

stem=ROOT/'figure1_single_column'
fig.savefig(str(stem)+'.pdf',metadata={'CreationDate':None,'ModDate':None,
                                      'Creator':'plot_figure1_single_column.py'})
fig.savefig(str(stem)+'.svg',metadata={'Date':None,'Creator':'plot_figure1_single_column.py'})
svg=Path(str(stem)+'.svg')
svg.write_text('\n'.join(row.rstrip() for row in svg.read_text().splitlines())+'\n')
fig.savefig(str(stem)+'.png',dpi=400)
plt.close(fig)
layout={
    'reproduction_command':'python fig/plot_figure1_single_column.py',
    'physical_size_points':[W,H],'physical_size_inches':[W/72,H/72],
    'column_width_source':'cvpr.sty: textwidth 6.875in, columnsep 0.3125in',
    'fonts':{'prose':'Albert Sans','math':'STIX','application_titles':7.5},
    'symbols':{'state':'Green symbolic browser icons, with s_t at the current state and s_(t+1) inside each future state','actions':'Individually boxed a_1 Scroll and a_2 Click examples followed by an ellipsis for additional actions, A_t; a_1 is purple and a_2 is dark purple',
               'lookahead':'Green future browsers beneath the candidates inside a pale salmon lookahead box; vertical arrows show action-to-future transitions and dots extend the time/action dimensions, E_t','adm':'Dark-purple model boxes matching the title, with white f_phi text',
               'policy':'Gray pi_theta; snowflake means frozen and flame means trained'},
    'teacher_inputs':['s_t','A_t','E_t'],'student_inputs':['s_t','A_t'],
    'model_notation':'f_phi(s_t,A_t)','teacher_notation':'T(s_t,A_t,E_t)',
    'example':{'task_displayed':False,'state':'Symbolic browser',
               'candidate_actions':['a_1: Scroll','a_2: Click'],'teacher_choice':'a_2: Click',
               'state_icons':True,'action_icons':False,'actual_screenshots':False},
    'top_panel_style':'The state and individually boxed action examples feed the ADM; a pale salmon lookahead box feeds the teacher. Distinct model input formulas and straight gray arrows are retained without nested input frames or extra group captions.',
    'row_gap_points':8,
    'top_arrow_color':POLICY,
    'input_notation_color':POLICY,
    'candidate_action_colors':{'a_1':PURPLE,'a_2':ADM},
    'privileged_notation_color':TEACHER,
    'output_notation_colors':{'selected_action':ADM,'teacher_action':TEACHER,'model_on_dark_fill':'white'},
    'title_color':ADM,
    'top_nodes':{'shared_context_area':[8,22,97,38],'adm':[119,32,50,26],
                 'student_probabilities':[177,33,25,27],'lookahead':[39,66,66,24],
                 'chosen_action':[215,37,17,16],
                 'teacher':[119,66,50,24],'teacher_label':[178,72,38,12],
                 'loss_center':[197,65]},
    'input_style':{'group_boxes':False,'extra_group_captions':False,'lookahead_box':True,'lookahead_outline':SALMON},
    'box_simplification':{'action_example_boxes':True,'application_action_symbol_boxes':True,'input_group_boxes':False,'lookahead_box':True,'model_boxes_preserved':True},
    'input_connectors':{'context_to_adm':[[105,45],[119,45]],
                         'lookahead_to_teacher':[[105,78],[119,78]],
                         'branch_connector_lines':0},
    'output_connectors':{'adm_to_probabilities':[[169,45],[177,45]],
                          'teacher_to_target':[[169,78],[178,78]]},
    'action_set_ellipsis':[100,45.5],
    'lookahead_branch_ellipsis':[100,77.5],
    'action_branches':[
        {'action':'a_1: Scroll','action_bounds':[39,36,26,19],'future_bounds':[40.5,71,23,13],
         'future_label':'s_(t+1)','future_color':STATE,'connector':[[52,55],[52,70]]},
        {'action':'a_2: Click','action_bounds':[68,36,26,19],'future_bounds':[69.5,71,23,13],
         'future_label':'s_(t+1)','future_color':STATE,'connector':[[81,55],[81,70]]}],
    'student_output':{'kind':'K-class probabilities','classes':['a_1','a_2','...','a_K'],
                      'ellipsis_position':[194,43],
                      'bar_heights':[7,16,None,4],'values':'Illustrative, not measured probabilities',
                      'highest_probability_action':'a_2: Click',
                      'argmax':{'notation':'a_hat_t','connector':[[207,45],[215,45]],
                                'clearance_from_probability_chart':5,
                                'path':'Probability chart -> argmax -> chosen action'}},
    'supervision':{'target':'a_2: Click','kind':'Hard candidate label','label_bounds':[178,72,38,12],
                   'path':'Student probabilities -> CE loss <- teacher label',
                   'loss_inputs':['K-class student probabilities','Hard teacher action label'],
                   'caption_displayed':False},
    'application_rows':{'test_time_scaling':120,'offline_improvement':155,'online_RL':190},
    'application_grid':{'column_left_edges':APP_X,'column_widths':APP_W,'column_heights':[18,18,22,18],'arrow_spans':APP_ARROWS,'boxed_action_outputs':True,'candidate_action_color':ADM,
                        'row_roles':{'test_time_scaling':['frozen_policy','candidate_actions','ADM','selected_action'],
                                     'offline_improvement':['ADM','preference_pair','RPO','updated_policy'],
                                     'online_RL':['ADM','selected_action','ADM_reward','updated_policy']}},
    'offline_mechanism':'RPO = DPO + SFT','online_mechanism':'Actor update with ADM-guided reward',
    'quantitative_results_displayed':False,
    'notes':'Separate single-column symbolic alternative; all previous figures are preserved.'}
(ROOT/'figure1_single_column_layout.json').write_text(json.dumps(layout,indent=2)+'\n')
print('Wrote fig/figure1_single_column.{pdf,svg,png} and layout JSON')
