"""Student-first preview for the preferred Figure 1 training diagram."""


def draw_swapped_training(text, box, line, arrow, state_window, tint, ax,
                          pixels, state_viewport, future_viewport, colors, simplified=False,
                          phase_bands=False, panel_headers=True, neutral_diagram=False):
    ink, border = colors['INK'], colors['BORDER']
    purple, action = colors['ADM_DEFAULT'], colors['ACTION_INK']
    salmon, teacher_ink = colors['LOOKAHEAD_TEACHER'], '#944735'
    input_ink = colors['BASELINE_DARK'] if neutral_diagram else action
    state_ink = colors['STATE_BORDER'] if neutral_diagram else salmon
    state_label_ink = input_ink if neutral_diagram else teacher_ink
    evidence_line = colors['BASELINE_DARK'] if neutral_diagram else salmon
    training_y = 54
    model_shift = 16 if simplified else 0
    action_colors = {'1':purple,'2':action,'K':action}
    if panel_headers:
        if phase_bands:
            band = box(24,training_y-13,860,26,'#F3F4F6','none',radius=3,lw=0)
            band.set_zorder(-2)
        text(450,training_y,'A. Training' if phase_bands else 'Training',size=9 if phase_bands else 11,color=ink,weight='bold',**({'clip_on':False} if phase_bands else {}))
        if not phase_bands:
            for left,right in [(24,383),(517,884)]:
                line([(left,training_y),(right,training_y)],border,.75)

    # Current state, proposals and student occupy the upper row.
    text(104,81 if phase_bands else 75 if simplified else 77,r'State $s_t$',size=8.6,color=input_ink if neutral_diagram else ink,
         va='baseline' if simplified else 'center')
    body = state_window(16,88,176,78)
    left,top,right,bottom = state_viewport
    shot = pixels[top:bottom,left:right]
    scale = 164/(right-left)
    height = 164*shot.shape[0]/shot.shape[1]
    cy = 88+78*.17+(78-78*.17-.8)/2
    page = ax.imshow(shot,extent=(22,186,cy+height/2,cy-height/2),
                     aspect='auto',interpolation='lanczos',zorder=3)
    page.set_clip_path(body)
    photo_top = cy-height/2
    for rect,label,color in [((136,354,158,396),'2',action),
                             ((383,376,414,418),'1',purple)]:
        x0,y0,x1,y1 = rect
        x,y = 22+(x0-left)*scale,photo_top+(y0-top)*scale
        w,h = (x1-x0)*scale,(y1-y0)*scale
        cover = box(x,y,w,h,'white','none',radius=.5,lw=0)
        cover.set_zorder(4)
        marker = box(x+.45,y+.45,w-.9,h-.9,color,'none',radius=.4,lw=0)
        marker.set_zorder(4)
        text(x+w/2,y+h/2,label,size=5.2,color='white',weight='bold',zorder=5)
    text(310,81 if phase_bands else 75 if simplified else 87,r'Actions $\mathcal{A}_t$',
         size=8.6 if simplified else 7,color=input_ink,
         va='baseline' if simplified else 'center')
    for i,x,width in [('1',223,80),('2',310,80),('K',418,40)]:
        label_color = action_colors[i] if simplified else action
        if not (simplified and i=='K'):
            fill = tint(purple,.90)
            box(x,118 if simplified else 98,width,38 if simplified else 58,
                fill,label_color,radius=4,lw=1.2 if simplified else .6)
        cx = x+width/2
        text(cx,100 if simplified else 112 if i!='K' else 127,
             rf'$a_t^{{({i})}}$',size=10.5,color=input_ink if neutral_diagram else label_color)
        if i!='K':
            text(cx,137 if simplified else 142,'Scroll down' if i=='1' else 'Click payment\noptions',
                 size=6.4 if i=='1' else 5.8,color=ink)
    text(406,100 if simplified else 127,r'$\cdots$',size=8,color=input_ink)
    arrow((198,137 if simplified else 127),(219,137 if simplified else 127),action,lw=.75,scale=3.5)
    text(206,122 if simplified else 112,r'$\pi_\theta$',size=10.5 if phase_bands else 7,color=colors['BASELINE_DARK'])
    if neutral_diagram:
        arrow((460,143),(492,143),action,lw=.75,scale=3.5)
    else:
        arrow((466,127+model_shift),(488,127+model_shift),action,lw=.75,scale=3.5)
    student_x = 608
    text(student_x,84 if simplified else 77,'Action Decision Models',size=15,color=action if neutral_diagram else '#A05AFF',weight='bold')
    box(492,93+model_shift,232,68,action,action,radius=6,lw=.6)
    text(student_x,109+model_shift,'ADM',size=11.5,color='white',weight='bold')
    text(student_x,139+model_shift,r'$f_\phi(s_t,\mathcal{A}_t)$',size=14.3,color='white')
    arrow((728,127+model_shift),(768 if simplified else 746,127+model_shift),action,lw=.75,scale=3.5)
    if simplified:
        # Schematic K-class probabilities: show the first, second and final action.
        line([(778,145),(866,145)],border,.5)
        for x,height,color in [(781,8,purple),(811,24,action),(851,6,tint(purple,.2))]:
            box(x,145-height,12,height,color,'none',radius=.8,lw=0)
        for x,label,color in [(787,'1',purple),(817,'2',action),(857,'K',action)]:
            text(x,155,label,size=5.5,color=input_ink if neutral_diagram else color)
        text(837,139,r'$\cdots$',size=6,color=input_ink)
    else:
        box(750,111,134,32,tint(purple,.96),action,radius=4,lw=.6)
        text(817,127,r'$p_\phi$',size=14,color=action)
    if simplified:
        box(798,72,38,28,tint(purple,.86),action,radius=4,lw=.65) if phase_bands else box(792,68,50,36,tint(purple,.86),action,radius=5,lw=.65)
    else:
        box(800,75,34,22,tint(purple,.85),action,radius=4,lw=.6)
    text(817,86,r'$\hat a_t$',size=10.5 if phase_bands else 14.5 if simplified else 9.5,color=action)
    arrow((817,118 if simplified else 111),(817,100 if phase_bands else 104 if simplified else 97),action,lw=.75,scale=3)
    text(850 if simplified else 842,113 if simplified else 103,
         'argmax',size=7 if phase_bands else 5.2,color=input_ink,ha='left')

    # Future evidence flows down from each proposal to the teacher row.
    box(24,193,3,53,colors['STATE'],'none',radius=1.5,lw=0)
    text(36,219,"Task: Find Netflix's\naccepted payment\nmethods.",size=8.5,
         color=ink,weight='bold' if panel_headers else 'normal',ha='left',linespacing=1.25)
    if neutral_diagram:
        group = box(218,192,260,56,tint(salmon,.96),salmon,radius=5,lw=.5)
        group.set_zorder(-1)
    elif not simplified:
        group = box(218,190,260,80,tint(salmon,.94),'none',radius=7,lw=0)
        group.set_zorder(-1)
    text(348,180 if simplified else 184,r'Privileged lookahead $\mathcal{E}_t$' if simplified else
         r'Privileged future states $\mathcal{E}_t$',size=8.6 if simplified else 7.5,color=teacher_ink)
    if not simplified:
        for x,label in [(263,r'$a_t^{(1)}$'),(350,r'$a_t^{(2)}$'),
                        (394,r'$\cdots$'),(438,r'$a_t^{(K)}$')]:
            text(x,205,label,size=6,color=teacher_ink)
    evidence_shift = -14 if simplified else 0
    text(198,226.5+evidence_shift,r'$s_{t+1}$',size=10.5 if phase_bands else 6.6,color=state_label_ink)
    text(203,257+evidence_shift-(4 if simplified else 0),r'$\vdots$',size=8,color=state_label_ink)
    for x in [229,316,404]:
        cell_y = 214+evidence_shift
        body = state_window(x,cell_y,68,25,lookahead=True)
        if x==316:
            l,t,r,b = future_viewport
            shot = pixels[t:b,l:r]
            h = 66.4*shot.shape[0]/shot.shape[1]
            cy = cell_y+5+(25-5-.8)/2
            page = ax.imshow(shot,extent=(x+.8,x+67.2,cy+h/2,cy-h/2),
                             aspect='auto',interpolation='lanczos',zorder=3)
            page.set_clip_path(body)
        else:
            box(x+5,223+evidence_shift,54,1.8,tint(state_ink,.05),'none',radius=.3,lw=0)
            box(x+5,227+evidence_shift,8,8,tint(state_ink,.60),'none',radius=.3,lw=0)
            for dy,length in [(14,40),(18,34),(22,37)]:
                box(x+17,cell_y+dy,length,1,state_ink,'none',radius=0,lw=0)
    text(394,226.5+evidence_shift,r'$\cdots$',size=8,color=input_ink if neutral_diagram else state_ink)
    for x in [263,350,438]:
        arrow((x,120 if simplified and x==438 else 156),(x,170 if simplified else 174),salmon,lw=.75,scale=3.5)
        text(x,257+evidence_shift-(4 if simplified else 0),r'$\vdots$',size=6,color=input_ink if neutral_diagram else state_ink)
    text(394,257+evidence_shift-(4 if simplified else 0),r'$\ddots$',size=6,color=input_ink if neutral_diagram else state_ink)
    bracket_top = 186 if simplified else 190
    bracket_bottom = 248 if simplified else 266
    if not neutral_diagram:
        line([(478,bracket_top),(484,bracket_top),(484,bracket_bottom),(478,bracket_bottom)],evidence_line,.7)
    if simplified:
        arrow((478 if neutral_diagram else 488,221),(527 if neutral_diagram else 523,221),salmon,lw=.75,scale=3.5)
        box(527,196,162,50,tint(salmon,.88),salmon,radius=5,lw=.45)
        text(608,208,'Teacher',size=8.6,color=teacher_ink,weight='bold')
        text(608,232,r'$\mathcal{T}(s_t,\mathcal{A}_t,\mathcal{E}_t)$',size=9.5,color=teacher_ink)
        teacher_label_bounds = [789,205,56,32] if neutral_diagram else None
        if teacher_label_bounds:
            box(*teacher_label_bounds,tint(salmon,.90),salmon,radius=4,lw=.5)
        arrow((693,221),(785 if neutral_diagram else 794,221),salmon,lw=.75,scale=3.5)
        text(817,221,r'$a_t^{(2)}$',size=11,color=teacher_ink)
        # A quiet comparison link conveys the objective without another flow node.
        line([(817,163),(817,205)],border,.55)
        text(830,181,'CE loss',size=8.6,color=colors['MUTED'],ha='left')
        return {'student_centerline':143,'teacher_centerline':221,'model_name':'ADM',
                'teacher_bounds':[527,196,162,50],'teacher_label_bounds':teacher_label_bounds,
                'loss_bounds':None,'loss_label_position':[830,181],
                'loss_comparison_line':[[817,163],[817,205]],'teacher_formula_font_size':9.5,
                'screenshot_markers':{'1':'Scroll down','2':'Click payment options'},
                'action_colors':action_colors,
                'state_action_header_baseline':81 if phase_bands else 75,
                'action_label_y':100,'action_card_bounds':[[223,118,80,38],[310,118,80,38]],
                'action_card_line_width':1.2,'component_label_font_size':8.6,
                'distribution_bounds':[778,121,88,39],
                'distribution_bars':{'classes':['1','2','...','K'],'heights':[8,24,None,6],
                                     'interpretation':'Schematic probabilities, not measured values'},
                'selected_action_bounds':[798,72,38,28] if phase_bands else [792,68,50,36],'selected_action_font_size':10.5 if phase_bands else 14.5,
                'future_states':'s_{t+1}; vertical dots indicate later states; action headers reused',
                'flow':'A fine neutral line compares the miniature probability chart with the hard teacher label, with an unboxed CE loss annotation beside it. Argmax branches upward separately. No p symbol is displayed.'}
    # Match the evidence row's center and use the lower-right space for supervision.
    arrow((488,230),(523,230),salmon,lw=.75,scale=3.5)
    box(527,205,162,50,tint(salmon,.88),salmon,radius=5,lw=.45)
    text(608,217,'Teacher',size=8.4,color=teacher_ink,weight='bold')
    text(608,241,r'$\mathcal{T}(s_t,\mathcal{A}_t,\mathcal{E}_t)$',size=9.5,color=teacher_ink)
    arrow((693,230),(746,230),salmon,lw=.75,scale=3.5)
    box(750,212,134,36,tint(salmon,.90),salmon,radius=4,lw=.45)
    text(817,230,r'$a_t^+=a_t^{(2)}$',size=11,color=teacher_ink)
    box(750,168,134,32,'white',border,radius=4,lw=.6)
    text(817,184,r'$\mathcal{L}_{\mathrm{ADM}}=-\log p_\phi(a_t^+)$',size=8.5,color=action)
    arrow((817,143),(817,168),action,lw=.75,scale=3)
    arrow((817,212),(817,200),salmon,lw=.75,scale=3)
    return {'student_centerline':127,'teacher_centerline':230,'model_name':'ADM',
            'teacher_bounds':[527,205,162,50],'teacher_label_bounds':[750,212,134,36],
            'loss_bounds':[750,168,134,32],'teacher_formula_font_size':9.5,
            'screenshot_markers':{'1':'Scroll down','2':'Click payment options'},
            'future_states':'s_{t+1}; vertical dots indicate later states',
            'flow':'Student probabilities and teacher label converge on the loss; argmax branches upward separately.'}
