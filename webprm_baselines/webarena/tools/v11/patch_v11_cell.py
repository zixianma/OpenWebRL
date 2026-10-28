# v11 (2026-10-27): copy of v10-tools/patch_v10_cell.py (unchanged there) with the 1280x720 actor viewport the user chose
# for both methods (no_search rerun + piotr_selectionarm) and the v11 Piotr helper (the ARM sees the actor's own screenshot).
# v10 header follows.
# v10 (2026-10-27, user: "ah I see. yes do A. so then we can use gpt4o + our ARM right? no need for openwebrl-sft"):
# cell changes on top of the v9 cell (exec'd by build_v10.py after patch_v9_cell.py with NEW = package root).
#   - fourth method piotr_selectionarm: released WebArbiter 5-BLOCK policy prompt unchanged (released default decoding),
#     released filter/dedup/rerank, terminal detector short-circuits (as no_search), Piotr SelectionARM selects
#     (openwebrl/webprm_native_piotr_v10.py); arbiter7b / shepherd8b / no_search code paths unchanged
#   - the browser hook grounds every executed action from its own pre-action frame for Piotr's history
cp = NEW/'source/scripts/run_native_webarena_cell.py'
ct = cp.read_text()
V11_CELL_PAIRS = [
    ("METHODS = ('no_search','arbiter7b','shepherd8b')\n",
     "METHODS = ('no_search','arbiter7b','shepherd8b','piotr_selectionarm')  # v10: Piotr SelectionARM\n"),
    ("    ENVELOPE_NO_SEARCH_PROMPT)\n",
     "    ENVELOPE_NO_SEARCH_PROMPT)\nfrom openwebrl.webprm_native_piotr_v11 import NativePiotrSelector, RECEIPT as V11_RECEIPT\n"),
    ("            if method == 'no_search' or not action:\n                return action\n",
     "            if method in ('no_search', 'piotr_selectionarm') or not action:  # v10: Piotr keeps the released override\n                return action\n"),
    ("adaptation='shared_released_policy_matched_control' if method == 'no_search' else 'v8_stop_competes_with_shepherd_pool'))",
     "adaptation='shared_released_policy_matched_control' if method == 'no_search' else\n"
     "                    'v10_piotr_selectionarm_released_five_block_pool' if method == 'piotr_selectionarm' else 'v8_stop_competes_with_shepherd_pool'))"),
    ("v9=V9_RECEIPT if method!='no_search' else None)\n",
     "v9=V9_RECEIPT if method in ('arbiter7b','shepherd8b') else None,v11=V11_RECEIPT if method=='piotr_selectionarm' else None)\n"),
    ("                if value['purpose']=='native_policy':\n                    value['upstream_unadapted_payload']=deepcopy(value['payload'])\n",
     "                if value['purpose']=='native_policy' and method=='piotr_selectionarm':  # v10: released 5-BLOCK prompt\n"
     "                    value['upstream_unadapted_payload']=deepcopy(value['payload'])\n"
     "                    value['policy_adaptation']='v10_released_five_block_prompt_unchanged'\n"
     "                elif value['purpose']=='native_policy':\n                    value['upstream_unadapted_payload']=deepcopy(value['payload'])\n"),
    ("                if purpose=='native_policy' and method!='no_search':\n                    if 'text' not in v9_observation:",
     "                if purpose=='native_policy' and method in ('arbiter7b','shepherd8b'):\n                    if 'text' not in v9_observation:"),
    ("        shepherd=ViewerSelector(method,config,folder/'requests-unused.jsonl',request=request) if method=='shepherd8b' else None\n",
     "        shepherd=ViewerSelector(method,config,folder/'requests-unused.jsonl',request=request) if method=='shepherd8b' else None\n"
     "        piotr=NativePiotrSelector(config,folder/'requests-piotr.jsonl',request=request) if method=='piotr_selectionarm' else None\n"),
    ("                return 0,[]\n            proposal=dict(candidates=deepcopy(candidates),terminal=False,",
     "                return 0,[]\n"
     "            if method=='piotr_selectionarm':  # v10: the frame is the browser's current (pre-action) capture\n"
     "                return await asyncio.to_thread(piotr.select,candidates,dict(intent=context.intent,current_url=inputs['current_url'],\n"
     "                    observation_text=inputs['observation_text']),deepcopy(browser.observations[-1]),inputs['current_step'])\n"
     "            proposal=dict(candidates=deepcopy(candidates),terminal=False,"),
    ("            return await v8_adapter(native=native,agent=agent,context=context,inputs=inputs)\n",
     "            decision=await v8_adapter(native=native,agent=agent,context=context,inputs=inputs)\n"
     "            if piotr:piotr.note_decision(decision)\n"
     "            return decision\n"),
    ("                executed=v8_bracket(action_string)\n",
     "                if piotr:piotr.note_execution(action_string,deepcopy(self.observations[-1]))  # v10: pre-action grounding\n"
     "                executed=v8_bracket(action_string)\n"),
    ("terminal=episode.get('terminal'),raw_effective_verdicts_preserved=True,",
     "terminal=episode.get('terminal'),raw_effective_verdicts_preserved=True,\n"
     "            v10_piotr_ungroundable_history_actions=piotr.ungroundable if piotr else None,"),
    ("'disclosed_single_action_control' if method=='no_search' else 'WebShepherd_critic_best_of_20_top5_frequency')",
     "'disclosed_single_action_control' if method=='no_search' else\n"
     "                'v10_piotr_selectionarm_released_five_block_pool' if method=='piotr_selectionarm' else 'WebShepherd_critic_best_of_20_top5_frequency')"),
]
V11_CELL_PAIRS.append(("V8_VIEWPORT = {'width': 1024, 'height': 2490}  # released service default (run_management.py:172)\n",
    "V8_VIEWPORT = {'width': 1280, 'height': 720}  # v11: user-chosen 1280x720 (WebArena browser_env default, envs.py:73; was the released service's 1024x2490, run_management.py:172)\n"))
for a, b in V11_CELL_PAIRS:
    assert ct.count(a) == 1, ('v11 cell anchor', a[:120])
    ct = ct.replace(a, b)
cp.write_text(ct)
