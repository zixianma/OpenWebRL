# v9 (2026-10-26, user: "follow that config for webarbiter/webshepherd ... also let's do B"): launcher changes on
# top of the built v8 launcher. exec'd by build_v9.py with `t` = launcher text and V9 = build parameters.
#   - one Table 5 site per package (reddit, or a validated third-party Magento site: shopping / shopping_admin);
#     the fixed cohort is that site's released single-site tasks x three methods
#   - native_policy route admits n=20 (paper Best-of-N sampling); every other route stays one choice
#   - per-package time / API / request / worker caps
#   - a cell invalid only because the retired gpt-4-1106-preview fuzzy judge is unavailable does not count toward
#     the 4-consecutive-invalid admission halt (17 of 128 tasks need it; it is re-scored offline as a sensitivity)
V9_PAIRS = [
    # cohort: the package's site, every released single-site task of it except DROPPED_TASK_IDS
    ("""    if (digest(cohort)!=proposal['cohort_sha256'] or len(cohort['cells'])!=54
            or cohort['actors']!=['gpt-4o-2024-08-06']
            or cohort['selectors']!=['no_search','arbiter7b','shepherd8b']):
        raise ValueError('Only19 nativeReddit tasks x three fixed methods are admitted')
    raw=read(Path(template['arbiter_source'])/TASK_FILE)
    ids=sorted(t['task_id'] for t in raw if t.get('sites')==['reddit'] and t['task_id'] not in DROPPED_TASK_IDS)
    expected=[(task,method) for method in cohort['selectors'] for task in ids]
    if (len(ids)!=18 or [(c['task_id'],c['selector']) for c in cohort['cells']]!=expected
            or len({c['cell_id'] for c in cohort['cells']})!=54
            or any(c['domain']!='reddit' or c['actor']!='gpt-4o-2024-08-06'
                or c['task_config_sha256']!=cohort_task_digest(raw[c['task_id']]) for c in cohort['cells'])):
        raise ValueError('Native57 exact task/method order or task hashes differ')""",
     """    site=v9_site(proposal)
    if (digest(cohort)!=proposal['cohort_sha256'] or cohort.get('protocol')!=V9_COHORT_PROTOCOL
            or cohort['actors']!=['gpt-4o-2024-08-06']
            or cohort['selectors']!=['no_search','arbiter7b','shepherd8b']):
        raise ValueError('Only a fixed v9 Table 5 site cohort x three fixed methods is admitted')
    raw=read(Path(template['arbiter_source'])/TASK_FILE)
    ids=sorted(t['task_id'] for t in raw if t.get('sites')==[site] and t['task_id'] not in DROPPED_TASK_IDS)
    expected=[(task,method) for method in cohort['selectors'] for task in ids]
    if (len(ids)!=V9_SITE_TASKS[site] or [(c['task_id'],c['selector']) for c in cohort['cells']]!=expected
            or len({c['cell_id'] for c in cohort['cells']})!=len(expected)
            or any(c['domain']!=site or c['actor']!='gpt-4o-2024-08-06' or raw[c['task_id']]['task_id']!=c['task_id']
                or c['task_config_sha256']!=cohort_task_digest(raw[c['task_id']]) for c in cohort['cells'])):
        raise ValueError('v9 exact task/method order or task hashes differ')"""),
    ("    if resources!=dict(gpus=1,gpu_type='H200',cpus=8,memory_gib=240,total_seconds=14280):",
     "    if resources!=dict(gpus=1,gpu_type='H200',cpus=8,memory_gib=240,total_seconds=V7_TOTAL_SECONDS):"),
    ("""    exact=dict(api_usd=50.0,paid_requests=6000,local_requests=6000,total_requests=12000,
        max_workers=1,scratch_bytes=1024*2**30,""",
     """    exact=dict(api_usd=V9_API_USD,paid_requests=V9_PAID_REQUESTS,local_requests=V9_LOCAL_REQUESTS,
        total_requests=V9_PAID_REQUESTS+V9_LOCAL_REQUESTS,max_workers=V9_MAX_WORKERS,scratch_bytes=1024*2**30,"""),
    # domain: original Reddit (unchanged checks) or exactly one validated Magento site
    ("""    if set(proposal['domains'])!={'reddit'}:raise ValueError('Original cachedReddit domain only')
    reddit=proposal['domains']['reddit']
    if reddit['endpoint']!='http://127.0.0.1' or reddit['state_apparent_bytes']!=50*2**30:
        raise ValueError('OriginalReddit endpoint/50GiB state bound required')
    for name in ('download_receipt','raw_verification','startup_review'):
        if str(reddit[name]) not in source:raise ValueError('OriginalReddit hosting evidence not frozen')
    for name in ('prepare_reddit_original.py','run_reddit_instance.py','start_reddit_services.sh'):
        if str(RUNTIME/name) not in source:raise ValueError('OriginalReddit hosting primitive not frozen')""",
     """    if set(proposal['domains'])!={site}:raise ValueError('A v9 package hosts exactly its one Table 5 site')
    if site in MAGENTO_DOMAINS:
        validate_magento_domain(site,proposal['domains'][site],source)
    else:
        reddit=proposal['domains']['reddit']
        if reddit['endpoint']!='http://127.0.0.1' or reddit['state_apparent_bytes']!=50*2**30:
            raise ValueError('OriginalReddit endpoint/50GiB state bound required')
        for name in ('download_receipt','raw_verification','startup_review'):
            if str(reddit[name]) not in source:raise ValueError('OriginalReddit hosting evidence not frozen')
        for name in ('prepare_reddit_original.py','run_reddit_instance.py','start_reddit_services.sh'):
            if str(RUNTIME/name) not in source:raise ValueError('OriginalReddit hosting primitive not frozen')"""),
    ("    rows=[dict(model='gpt-4o-2024-08-06',purpose='native_policy',local=False,max_input_tokens=120000,max_output_tokens=2048),",
     "    rows=[dict(model='gpt-4o-2024-08-06',purpose='native_policy',local=False,max_input_tokens=120000,max_output_tokens=2048,max_choices=20),  # v9 n=20"),
    ("limits=self.proposal['limits'];base_bytes=self.proposal['domains']['reddit']['state_apparent_bytes']",
     "limits=self.proposal['limits'];base_bytes=self.proposal['domains'][self.site_domain]['state_apparent_bytes']"),
    ("routes[(route['model'],route['purpose'])]=dict(local=route['local'],endpoint=endpoint,max_input_tokens=route['max_input_tokens'],max_output_tokens=route['max_output_tokens'])",
     "routes[(route['model'],route['purpose'])]=dict(local=route['local'],endpoint=endpoint,max_input_tokens=route['max_input_tokens'],max_output_tokens=route['max_output_tokens'],max_choices=route.get('max_choices',1))"),
    ("            self.consecutive_invalid=0 if result.get('valid') else getattr(self,'consecutive_invalid',0)+1\n",
     "            self.consecutive_invalid=(0 if result.get('valid') else getattr(self,'consecutive_invalid',0)\n"
     "                if v9_fuzzy_judge_invalid(read(result_path)) else getattr(self,'consecutive_invalid',0)+1)\n"),
    ("                    run_one=self.run_one,release=self.release,max_workers=1,",
     "                    run_one=self.run_one,release=self.release,max_workers=self.proposal['limits']['max_workers'],"),
    ("V7_TOTAL_SECONDS = 14280", f"V7_TOTAL_SECONDS = {int(V9['total_seconds'])}  # v9 package"),
    ("fresh.get('identity') != 'native_reddit54_v8_0806'", "fresh.get('identity') != 'native_table5_v9'"),
]
for a, b in V9_PAIRS:
    assert t.count(a) == 1, ('v9 launcher anchor', a[:120])
    t = t.replace(a, b)
V9_HELPERS = f'''

# ---- v9 (2026-10-26): four Table 5 sites, paper Best-of-N sampling. ----
V9_COHORT_PROTOCOL = 'released_coordinator_original_browser_table5_v9'
V9_SITE_TASKS = {{'shopping': 45, 'shopping_admin': 35, 'reddit': 18, 'gitlab': 30}}
V9_API_USD = {float(V9['api_usd'])}
V9_PAID_REQUESTS = {int(V9['paid_requests'])}
V9_LOCAL_REQUESTS = {int(V9['local_requests'])}
V9_MAX_WORKERS = {int(V9['max_workers'])}
V9_FUZZY_ERROR = 'Native fuzzy transport/accounting failed; no retry'


def v9_site(proposal):
    domains = list(proposal.get('domains', {{}}))
    if len(domains) != 1 or domains[0] not in HOSTED_DOMAINS or domains[0] not in V9_SITE_TASKS:
        raise ValueError('A v9 package hosts exactly one implemented Table 5 site')
    return domains[0]


def v9_fuzzy_judge_invalid(result):
    """Invalid only because the retired gpt-4-1106-preview fuzzy judge could not be called."""
    return (result.get('valid') is False and result.get('error_type') == 'EvaluationNotReady'
            and result.get('error') == V9_FUZZY_ERROR)
'''
anchor = "\nif __name__=='__main__':main()"
assert t.count(anchor) == 1
t = t.replace(anchor, V9_HELPERS + anchor)
