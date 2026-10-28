# v9 (2026-10-26, user: "ah yes can we follow that config for webarbiter/webshepherd (read the papers again to
# confirm). also let's do B"): cell-worker science changes on top of the v8 worker. exec'd by build_v9.py with
# NEW = package root, V9 = dict(task_ids=..., api_usd=...).
#   1. Candidate protocol for arbiter7b / shepherd8b, from the papers (WebShepherd App. F.2: "we sample 20 output
#      sequences using nucleus sampling with a temperature of 1.0"; the top-n most frequent actions are the
#      candidates; WebArbiter Sec. 5.2.1: "Following WebShepherd ... Best-of-N ... N=5"): one policy request with
#      n=20, temperature 1.0, top_p 1.0 and the released single-action prompt envelope (same as no_search); every
#      sample is parsed to one BLOCK and checked with the released validate/filter_candidates; the 5 most frequent
#      distinct valid actions (ties: first sampled) become the BLOCKS the released sampler parses, with confidence =
#      sample share, so the released confidence reranking keeps frequency order. The full 20-sample record is saved.
#      WebShepherd ties already go to the lower index (= more frequent). no_search is unchanged.
#   2. Four Table 5 sites: RunContext domain / released domain knowledge follow the task's site (was 'reddit').
#   3. Fixed v9 task set and per-package API cap.
cp = NEW/'source/scripts/run_native_webarena_cell.py'
ct = cp.read_text()


def _v9_patch(text, pairs):
    for a, b in pairs:
        assert text.count(a) == 1, ('v9 cell anchor', a)
        text = text.replace(a, b)
    return text


V9_HELPERS = '''

# ---- v9 (2026-10-26): paper Best-of-N candidate protocol; four Table 5 sites. ----
V9_SAMPLES = 20  # WebShepherd App. F.2 "we sample 20 output sequences"
V9_TOP_N = 5  # WebArbiter Sec. 5.2.1 "N=5 candidate actions"; WebShepherd "top-n most frequent actions"
V9_SAMPLING = dict(n=V9_SAMPLES, temperature=1.0, top_p=1.0)  # "nucleus sampling with a temperature of 1.0"
V9_RECEIPT = dict(samples=V9_SAMPLES, top_n=V9_TOP_N, sampling=V9_SAMPLING, prompt='disclosed_single_action_envelope_v2',
    ranking='frequency_desc_then_first_sampled', validity='released_filter_candidates',
    confidence='sample_share_of_valid', methods=['arbiter7b', 'shepherd8b'], no_search='unchanged_v8')


def v9_aggregate(response, observation_text, sampling, choice_text):
    """Turn one n=20 policy response into the released sampler's BLOCKS text plus a full record.

    choice_text(single_choice_response) -> text validates one choice exactly like a one-choice reply.
    """
    from collections import Counter
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != V9_SAMPLES:
        raise EvaluationNotReady('v9 policy reply must carry exactly 20 samples')
    usage = response.get('usage')
    samples, valid = [], []
    for index, choice in enumerate(choices):
        row = dict(index=index, finish_reason=choice.get('finish_reason') if isinstance(choice, dict) else None)
        try:
            text = strip_json_fence(choice_text(dict(choices=[choice], usage=usage)))
            parsed = json.loads(text)
            blocks = parsed.get('BLOCKS') if isinstance(parsed, dict) else None
            if not isinstance(blocks, list) or len(blocks) != 1 or not isinstance(blocks[0], dict):
                raise ValueError('not exactly one BLOCK')
            thought = (blocks[0].get('thought') or '').strip() if isinstance(blocks[0].get('thought'), str) else ''
            action = (blocks[0].get('action') or '').strip() if isinstance(blocks[0].get('action'), str) else ''
            row.update(thought=thought, action=action)
            block = sampling.BlockInfo(thought=thought, action=action, confidence=None)
            if thought and action and sampling.filter_candidates([block], observation_text):
                row['status'] = 'valid'
                valid.append(row)
            else:
                row['status'] = 'rejected_by_released_filter'
        except Exception as error:
            row.update(status='unparsed', error_type=type(error).__name__)
        samples.append(row)
    counts = Counter(row['action'] for row in valid)
    first = {}
    for row in valid:
        first.setdefault(row['action'], row)
    ranked = sorted(counts, key=lambda action: (-counts[action], first[action]['index']))[:V9_TOP_N]
    blocks = [dict(thought=first[action]['thought'], action=action, confidence=counts[action]/len(valid),
                   frequency=counts[action]) for action in ranked]
    record = dict(protocol=V9_RECEIPT, samples=samples, valid_samples=len(valid), distinct_valid_actions=len(counts),
                  candidates=[dict(action=b['action'], frequency=b['frequency'], first_sample=first[b['action']]['index'])
                              for b in blocks])
    return json.dumps(dict(BLOCKS=blocks)), record
'''
anchor = "\n\ndef _run_episode(task_record, arm, config):"
assert ct.count(anchor) == 1
ct = ct.replace(anchor, V9_HELPERS + anchor)

ct = _v9_patch(ct, [
    ("PROTOCOL = 'released_coordinator_original_browser_reddit19_v1'",
     "PROTOCOL = 'released_coordinator_original_browser_table5_v9'"),
    ("TASK_IDS = (5,14,97,98,131,132,133,134,135,136,137,138,139,140,141,151,152,153,154)",
     "TASK_IDS = " + repr(tuple(V9['task_ids'])) + "  # v9: Table 5 Shopping/CMS/Reddit/GitLab single-site tasks"),
    ("config.get('api_usd_cap')!=50.0:  # v7h block cap", f"config.get('api_usd_cap')!={float(V9['api_usd'])}:  # v9 package cap"),
    ("raise EvaluationNotReady('Exact positive native57 authorization required')",
     "raise EvaluationNotReady('Exact positive v9 authorization required')"),
    ("intent=task.actor_view['intent'],domain='reddit',domain_knowledge=native.history.load_domain_knowledge('reddit') or '',",
     "intent=task.actor_view['intent'],domain=v9_domain,domain_knowledge=native.history.load_domain_knowledge(v9_domain) or '',"),
    ("        Spec=native.tools['engine'].ModelSpec\n",
     "        Spec=native.tools['engine'].ModelSpec\n"
     "        v9_domain=task.native_config()['sites'][0]  # v9: the task's own Table 5 site\n"
     "        if len(task.native_config()['sites'])!=1:raise EvaluationNotReady('v9 admits single-site tasks only')\n"
     "        v9_observation={}\n"),
    ("max_attempts=512,selector=method,no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT if method=='no_search' else None,v8=V8_RECEIPT)",
     "max_attempts=512,selector=method,no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT if method=='no_search' else None,v8=V8_RECEIPT,\n"
     "            v9=V9_RECEIPT if method!='no_search' else None)"),
    # request: arbiter7b / shepherd8b policy = single-action envelope sampled 20 times at T=1, top_p=1.
    ("                if method=='no_search' and value['purpose']=='native_policy':\n"
     "                    value['upstream_unadapted_payload']=deepcopy(value['payload'])\n"
     "                    value['payload']['messages'][0]['content']=single_action_system_prompt(\n"
     "                        value['payload']['messages'][0]['content'],version=ENVELOPE_NO_SEARCH_PROMPT)\n"
     "                    value['policy_adaptation']='disclosed_single_action_envelope_v2'\n",
     "                if value['purpose']=='native_policy':\n"
     "                    value['upstream_unadapted_payload']=deepcopy(value['payload'])\n"
     "                    value['payload']['messages'][0]['content']=single_action_system_prompt(\n"
     "                        value['payload']['messages'][0]['content'],version=ENVELOPE_NO_SEARCH_PROMPT)\n"
     "                    value['policy_adaptation']='disclosed_single_action_envelope_v2'\n"
     "                    if method!='no_search':\n"
     "                        value['payload'].update(V9_SAMPLING)\n"
     "                        value['policy_adaptation']='v9_best_of_20_top5_frequency'\n"),
    # response: aggregate the 20 samples (no_search keeps its exact one-BLOCK check).
    ("            def response_text(self,response):\n"
     "                text,usage=super().response_text(response)\n"
     "                purpose=response.get('_webprm_broker',{}).get('purpose')\n",
     "            def response_text(self,response):\n"
     "                purpose=response.get('_webprm_broker',{}).get('purpose')\n"
     "                if purpose=='native_policy' and method!='no_search':\n"
     "                    if 'text' not in v9_observation:raise EvaluationNotReady('v9 policy reply without its observation')\n"
     "                    single=super().response_text\n"
     "                    def choice_text(value):return single(value)[0]\n"
     "                    usage=deepcopy(response.get('usage'))\n"
     "                    total=dict(choices=[dict(finish_reason='stop',message=dict(role='assistant',content='{}'))],usage=usage)\n"
     "                    single(total)  # usage schema/reconciliation exactly as a one-choice reply\n"
     "                    text,record=v9_aggregate(response,v9_observation['text'],native.tools['sampling'],choice_text)\n"
     "                    v9_observation['sequence']=v9_observation.get('sequence',0)+1\n"
     "                    write(folder/f\"v9-samples-{v9_observation['sequence']:03d}.json\",dict(record,step=v9_observation.get('step'),\n"
     "                        request_id=response.get('_webprm_broker',{}).get('request_id')))\n"
     "                    return text,usage\n"
     "                text,usage=super().response_text(response)\n"),
    # the decision adapter records the observation the sampler sees (same inputs the released coordinator gets).
    ("        decision_adapter=v8_decision_adapter(method,select_control,\n"
     "            lambda step,value:write(folder/f'selector-{step:03d}.json',value))\n",
     "        v8_adapter=v8_decision_adapter(method,select_control,\n"
     "            lambda step,value:write(folder/f'selector-{step:03d}.json',value))\n"
     "        async def decision_adapter(*,native,agent,context,inputs):\n"
     "            v9_observation.update(text=inputs['observation_text'],step=inputs['current_step'])\n"
     "            return await v8_adapter(native=native,agent=agent,context=context,inputs=inputs)\n"),
    ("raise EvaluationNotReady('Cell outside fixed native19×3 cohort')", "raise EvaluationNotReady('Cell outside fixed v9 cohort')"),
    ("'disclosed_single_action_control' if method=='no_search' else 'WebShepherd_critic_shared_five_candidate_policy')",
     "'disclosed_single_action_control' if method=='no_search' else 'WebShepherd_critic_best_of_20_top5_frequency')"),
])
cp.write_text(ct)
