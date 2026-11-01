"""Prospective transport/timing injection; archived runtime files stay unchanged.

Use the same GPT-high worker for Luna single high and N5 high. The caller owns
paired task admission, all campaign caps and explicit shared Terra binding.
There is no paid CLI and importing/installing makes no provider calls.
"""
from pathlib import Path
import threading
from .cloud_browser import CloudBrowser, Timeline, need, write_private


class TimedClient:
    def __init__(self, client, timeline, role):
        self.client, self.timeline, self.role = client, timeline, role
        self.responses = self.Endpoint(client.responses, timeline, role, 'responses') if hasattr(client, 'responses') else None
        if hasattr(client, 'chat'):
            self.chat = self.Chat(client.chat, timeline, role)

    class Chat:
        def __init__(self, chat, timeline, role):
            self.completions = TimedClient.Endpoint(chat.completions, timeline, role, 'chat.completions')

    class Endpoint:
        def __init__(self, endpoint, timeline, role, name):
            self.endpoint, self.timeline, self.role, self.name = endpoint, timeline, role, name
            if hasattr(endpoint, 'input_tokens'):
                self.input_tokens = TimedClient.Endpoint(endpoint.input_tokens, timeline, role, 'input_tokens')

        def __getattr__(self, name):
            value = getattr(self.endpoint, name)
            if name not in ('create','count'):
                return value
            def call(*args, **kwargs):
                with self.timeline.span(self.role+'_http', endpoint=self.name, token_counter=self.name=='input_tokens'):
                    return value(*args, **kwargs)
            return call

    def __getattr__(self, name):
        return getattr(self.client, name)


class Registry:
    def __init__(self, worker, environments, *, http_factory, admission,
                 metadata_factory, on_halt, max_proxy_usd, on_usage=None,
                 judge_function=None, judge_budget_type=None, cloud_options=None):
        self.worker, self.environments = worker, environments
        self.judge_budget_type = judge_budget_type
        self.original = dict(OwnedBrowser=worker.OwnedBrowser, PairedModel=worker.PairedModel,
                             BrowserCodeWorkspace=worker.BrowserCodeWorkspace, judge=worker.judge)
        self.sessions = {}
        self.timelines = {}
        self.lock = threading.RLock()
        owner = self
        options = dict(cloud_options or {})
        need(not({'http','admission','metadata','on_halt','max_proxy_usd','on_usage','timeline'} & set(options)), 'Cloud identity/approval options cannot be overridden')

        def timeline(directory):
            directory = Path(directory).resolve()
            with owner.lock:
                if directory not in owner.timelines:
                    owner.timelines[directory] = Timeline(directory)
                return owner.timelines[directory]

        class Cloud(CloudBrowser):
            def __init__(self, directory):
                directory = Path(directory).resolve()
                need(directory not in owner.sessions, 'One cloud create owner per attempt directory')
                super().__init__(directory, http_factory(), lambda payload:admission(directory,payload),
                    metadata=metadata_factory(directory), on_halt=on_halt, max_proxy_usd=max_proxy_usd,
                    on_usage=(lambda view:on_usage(directory,view)) if on_usage else None,
                    timeline=timeline(directory), **options)
                owner.sessions[directory] = self

        class Model(self.original['PairedModel']):
            def __init__(self, *args, **kwargs):
                args = list(args)
                directory = args[1] if len(args)>1 else kwargs['directory']
                self.cloud_timeline = timeline(directory)
                self.cloud_timeline.phase = 'actor'
                if len(args)>2:
                    args[2] = TimedClient(args[2], self.cloud_timeline, 'actor')
                else:
                    kwargs['client'] = TimedClient(kwargs['client'], self.cloud_timeline, 'actor')
                super().__init__(*args, **kwargs)

            def query(self, *args, **kwargs):
                with self.cloud_timeline.span('model_decision', decision_turn=getattr(self,'turn',None)):
                    return super().query(*args, **kwargs)

        class Workspace(self.original['BrowserCodeWorkspace']):
            def __init__(self, *args, **kwargs):
                workspace = kwargs.get('output_dir', args[0] if args else None)
                need(workspace is not None, 'Workspace output directory required')
                self.cloud_timeline = timeline(Path(workspace).parent)
                super().__init__(*args, **kwargs)

            def start(self, *args, **kwargs):
                with self.cloud_timeline.span('browser_bridge_start'):
                    return super().start(*args, **kwargs)

            def execute(self, *args, **kwargs):
                with self.cloud_timeline.span('browser_execute', phase=self.cloud_timeline.phase):
                    return super().execute(*args, **kwargs)

            def capture(self, *args, **kwargs):
                with self.cloud_timeline.span('terminal_capture'):
                    return super().capture(*args, **kwargs)

            def close(self):
                with self.cloud_timeline.span('browser_bridge_close'):
                    return super().close()

        source_judge = judge_function or self.original['judge']
        def judge(client, budget, task, actions, image, directory):
            directory = Path(directory).resolve()
            need(directory in owner.sessions, 'Judge lacks owned cloud session')
            owner.sessions[directory].require_stopped()
            clock = timeline(directory)
            with clock.span('terra_total'):
                return source_judge(TimedClient(client, clock, 'terra'), budget, task, actions, image, directory)

        worker.OwnedBrowser, worker.PairedModel, worker.BrowserCodeWorkspace, worker.judge = Cloud, Model, Workspace, judge

    def close_all(self):
        errors = []
        for directory, session in list(self.sessions.items()):
            if session.session_id is None and not session.create_dispatched:
                continue
            if session.session_id is None:
                # A failed/ambiguous create must be reconciled, never silently
                # considered a confirmed stopped browser or zero-dollar event.
                errors.append(dict(directory=str(directory), reason='unresolved_cloud_create'))
                continue
            try:
                session.close()
                session.require_stopped()
            except BaseException as exc:
                errors.append(dict(directory=str(directory), error_type=type(exc).__name__))
            write_private(directory/'timing-summary.json', self.timelines[directory].summary())
        need(not errors, 'Unresolved cloud ownership; inspect private session receipts')
        return dict(verified=True, sessions=len(self.sessions))

    def restore(self):
        for name, value in self.original.items():
            setattr(self.worker, name, value)


def install(worker, environments, **kwargs):
    return Registry(worker, environments, **kwargs)


def configure(runtime):
    """Prepare a NEW generated GPT-Luna-single runtime for the cloud study.

    No provider calls. Archive every replaced generated file, leave repository
    snapshots unchanged, bind exact Terra functions and refresh source manifests.
    This grants no paid authorization and changes no output/decision ceiling.
    """
    import copy
    import hashlib
    import json
    import shutil
    from . import rejudge
    runtime = Path(runtime).resolve()
    need(not runtime.is_relative_to(Path('/gpfs/home/zixianma')), 'Home is read-only')
    need(not (runtime/'cloud-experiment.json').exists(), 'Cloud experiment already configured')
    need(not (runtime/'episodes').exists() and not (runtime/'budget').exists(), 'Only unused generated runtimes may be configured')
    read = lambda p:json.loads(Path(p).read_text())
    digest = rejudge.sha
    manifest = read(runtime/'source-manifest.json')
    need(all(digest(path)==pin for path,pin in manifest['source_hashes'].items()), 'Original generated runtime source changed')
    plan, config = read(runtime/'campaign-plan.json'), read(runtime/'runtime-config.json')
    need(len(plan['arms'])==1 and plan['arms'][0]['arm']=='luna_single_high'
         and plan['arms'][0]['model']=='gpt-6-luna'
         and plan['arms'][0]['output_tokens']==12288 and plan['protocol']['max_decisions']==30,
         'Expected unused high1x GPT-Luna-single runtime')
    judge, budget, sources = rejudge.science()
    index = read(rejudge.ROOT/'source-provenance.json')['snapshots']['browsercode_luna_effort']['files']
    entries = {row['file']:row for row in index}
    replacements = {'common_judge.py':rejudge.ROOT/entries['common_judge.py']['published_path'],
                    'claude_budget.py':rejudge.ROOT/entries['claude_budget.py']['published_path'],
                    'request_budget.py':rejudge.ROOT/entries['request_budget.py']['published_path']}
    names = ['campaign-plan.json','runtime-config.json','source-manifest.json','execution-source-manifest.json',
             'portable-build.json','episode_metrics.py',*replacements]
    archive = runtime/'cloud-originals'
    archive.mkdir(exist_ok=False)
    original = {}
    for name in names:
        if (runtime/name).is_file():
            shutil.copy2(runtime/name, archive/name)
            original[name] = digest(archive/name)
    single = copy.deepcopy(plan['arms'][0])
    n5 = dict(single, arm='parallel5_select_high', proposal_count=5,
              reasoning_effort='high', selector_reasoning_effort='high', selector_output_tokens=12288)
    plan['arms'] = [single, n5]
    plan['arm_protocols']['parallel5_select_high'] = dict(actor_model='gpt-6-luna', selector_model='gpt-6-luna', candidates=5)
    protocol = copy.deepcopy(judge.TERRA_PROTOCOL)
    plan['protocol'].update(judge_model=protocol['model'], judge_protocol=protocol,
                            browser_backend='browser-use-v4', proxy_country='us')
    config['judge_protocol'] = protocol
    config['pricing'].update(judge_input=protocol['pricing']['input'], judge_cached=protocol['pricing']['cached_input'], judge_output=protocol['pricing']['output'])
    config['cloud_protocol'] = dict(browser_backend='browser-use-v4', proxy_country='us',
             max_provider_minutes=20, viewport=[1280,1800], paid_authorization=False)
    for name, source in replacements.items():
        shutil.copyfile(source, runtime/name)
    metrics = (runtime/'episode_metrics.py').read_text()
    old = "cost=api_cost(u,'judge' if role=='judge' else 'actor',pricing)['upper_usd']"
    need(metrics.count(old)==1, 'Frozen metric accounting adaptation anchor changed')
    metrics = metrics.replace(old, "cost=d['cost']['upper_usd'] if role=='judge' and 'cost'in d else api_cost(u,'judge' if role=='judge' else 'actor',pricing)['upper_usd']")
    (runtime/'episode_metrics.py').write_text(metrics)
    write_private(runtime/'campaign-plan.json', plan)
    write_private(runtime/'runtime-config.json', config)
    build = read(runtime/'portable-build.json')
    build['prospective_cloud_adapter'] = dict(source_sha256=digest(__file__), original_file_sha256=original,
                    judge_sources=sources, archive='cloud-originals', paid_authorization=False)
    write_private(runtime/'portable-build.json', build)
    receipt = dict(schema='prospective-browsercode-cloud-study-v1', browser_backend='browser-use-v4',
          proxy_country='us', arms=['luna_single_high','parallel5_select_high'], actor_output_tokens=12288,
          max_decisions=30, actor_reasoning_effort='high', judge_protocol=protocol,
          archived_original_file_sha256=original, terra_source_sha256=sources,
          adapter_sha256=digest(__file__),
          adapter_source_sha256={name:digest(Path(__file__).with_name(name)) for name in ('common.py','cloud_browser.py','cloud_browsercode.py')},
          paid_authorization=False)
    write_private(runtime/'cloud-experiment.json', receipt, exclusive=True)
    for path in [runtime/name for name in names if name not in ('source-manifest.json','execution-source-manifest.json')]+[runtime/'cloud-experiment.json']+list(archive.iterdir()):
        if path.is_file():
            manifest['source_hashes'][str(path)] = digest(path)
    manifest['plan_sha256'] = digest(runtime/'campaign-plan.json')
    write_private(runtime/'source-manifest.json', manifest)
    write_private(runtime/'execution-source-manifest.json', manifest)
    return receipt
