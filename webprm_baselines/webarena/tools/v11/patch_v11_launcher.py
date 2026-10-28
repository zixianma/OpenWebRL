# v11 (2026-10-27): copy of v10-tools/patch_v10_launcher.py (unchanged there) for two methods per package, interleaved:
# cohort = site tasks x (no_search, piotr_selectionarm) in task-major order; one admission block serves both methods while
# the Piotr server runs (no_search never calls it), so each task's two cells run back to back on fresh copies of the same
# site base. v10 header follows.
# v10 (2026-10-27, user: "ah I see. yes do A. so then we can use gpt4o + our ARM right? no need for openwebrl-sft"):
# launcher changes on top of the v9 launcher (exec'd by build_v10.py after patch_v9_launcher.py with `t` = launcher text).
#   - one method (piotr_selectionarm) and one local model (Piotr SelectionARM, sglang, processor view v3, compatible
#     RoPE config) per package; the v9 no_search cells of the same 128 tasks are the paired control
#   - native_policy is one choice again (no n=20); local route selector_piotr (65,536 input / 128 output tokens, as the
#     earlier CMS / OM2W Piotr campaigns)
V11_PAIRS = [
    ("CRITICS = ('arbiter7b', 'shepherd8b')\nPREFIX = dict(zip(CRITICS, ('arbiter', 'shepherd')))\n",
     "CRITICS = ('piotr_selectionarm',)  # v10: Piotr SelectionARM only\nPREFIX = dict(zip(CRITICS, ('piotr',)))\n"),
    ("            or cohort['selectors']!=['no_search','arbiter7b','shepherd8b']):\n        raise ValueError('Only a fixed v9 Table 5 site cohort x three fixed methods is admitted')",
     "            or cohort['selectors']!=['no_search','piotr_selectionarm']):\n        raise ValueError('Only a fixed v11 Table 5 site cohort x (no_search, Piotr SelectionARM) is admitted')"),
    ("    if set(proposal['models'])!=set(CRITICS):raise ValueError('Only cachedArbiter/Shepherd critics are admitted')",
     "    if set(proposal['models'])!=set(CRITICS):raise ValueError('Only the cached Piotr SelectionARM is admitted')"),
    ("    rows=[dict(model='gpt-4o-2024-08-06',purpose='native_policy',local=False,max_input_tokens=120000,max_output_tokens=2048,max_choices=20),  # v9 n=20",
     "    rows=[dict(model='gpt-4o-2024-08-06',purpose='native_policy',local=False,max_input_tokens=120000,max_output_tokens=2048),  # v10: one choice"),
    ("""    rows.extend([dict(model=models['arbiter7b']['served_model'],purpose='native_reward',critic='arbiter7b',local=True,max_input_tokens=120000,max_output_tokens=2048),
          dict(model=models['shepherd8b']['served_model'],purpose='selector_checklist',critic='shepherd8b',local=True,max_input_tokens=120000,max_output_tokens=1024),
          dict(model=models['shepherd8b']['served_model'],purpose='selector_pointwise',critic='shepherd8b',local=True,max_input_tokens=120000,max_output_tokens=2048)])""",
     """    rows.append(dict(model=models['piotr_selectionarm']['served_model'],purpose='selector_piotr',critic='piotr_selectionarm',local=True,max_input_tokens=65536,max_output_tokens=128))  # v10"""),
    ("    if selector=='shepherd8b':rows.extend([[models[selector]['served_model'],p] for p in ('selector_checklist','selector_pointwise')])\n",
     "    if selector=='shepherd8b':rows.extend([[models[selector]['served_model'],p] for p in ('selector_checklist','selector_pointwise')])\n"
     "    if selector=='piotr_selectionarm':rows.append([models[selector]['served_model'],'selector_piotr'])  # v10\n"),
    ("        argv=model_command(name,model,self.ports[name])\n        put(self.output/(name+'-command.json'),dict(argv=argv,gpu_index=0,scientific_identity_unchanged=True))",
     "        compatible=None\n"
     "        if name=='piotr_selectionarm':  # v10: published RoPE fields in the sglang-compatible form (OM2W / CMS Piotr campaigns)\n"
     "            compatible=self.output/'piotr-compatible-config.json'\n"
     "            if not compatible.exists():put(compatible,piotr_compatible_config(model))\n"
     "        argv=model_command(name,model,self.ports[name],compatible_config=compatible)\n"
     "        put(self.output/(name+'-command.json'),dict(argv=argv,gpu_index=0,scientific_identity_unchanged=True))"),
    ("        purpose='native_reward' if name=='arbiter7b' else 'selector_pointwise'\n        payload=dict(model=model['served_model'],messages=[dict(role='user',content='Reply OK.')],temperature=0.,top_p=1.,max_tokens=8 if name=='arbiter7b' else 2048)",
     "        purpose='native_reward' if name=='arbiter7b' else 'selector_piotr' if name=='piotr_selectionarm' else 'selector_pointwise'\n"
     "        payload=dict(model=model['served_model'],messages=[dict(role='user',content='Reply OK.')],temperature=0.,top_p=1.,max_tokens=8 if name in ('arbiter7b','piotr_selectionarm') else 2048)"),
    ("            for selector in ('no_search','arbiter7b','shepherd8b'):\n",
     "            for selector in ('piotr_selectionarm',):  # v11: one interleaved block (no_search + ARM per task) under the ARM server\n"),
    ("                block=[c for c in cells if c['selector']==selector and store.state(c)=='pending']\n",
     "                block=[c for c in cells if store.state(c)=='pending']  # v11: both methods\n"),
    ("                    can_start=lambda cell,active:cell['selector']==selector and self.can_start(cell,active),\n",
     "                    can_start=lambda cell,active:self.can_start(cell,active),  # v11: both methods, cohort (task-major) order\n"),
    ("    expected=[(task,method) for method in cohort['selectors'] for task in ids]\n",
     "    expected=[(task,method) for task in ids for method in cohort['selectors']]  # v11: task-major interleaving\n"),
    ("                or block['selector'] not in ('no_search', 'arbiter7b', 'shepherd8b')",
     "                or block['selector'] not in ('no_search', 'piotr_selectionarm')"),
    ("fresh.get('identity') != 'native_table5_v9'", "fresh.get('identity') != 'native_table5_v11'"),
]
for a, b in V11_PAIRS:
    assert t.count(a) == 1, ('v11 launcher anchor', a[:120])
    t = t.replace(a, b)
