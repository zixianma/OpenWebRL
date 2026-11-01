"""Two-worker paired BrowserCode pilot. Default preparation makes no paid calls.

Install beside common.py/cloud_browsercode.py. A caller supplies a prepared
cloud-study runtime, frozen ordered proposal and an explicit local approval.
Every provider receipt remains private under --output. Recovery is not automatic.
"""
import argparse
import contextlib
from datetime import datetime
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ARMS = ('luna_single_high', 'parallel5_select_high')
CAPS = {'luna_single_high': ('1', 1200), 'parallel5_select_high': ('3', 7200)}


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def money(value):
    result = Decimal(str(value))
    need(result.is_finite() and result >= 0, 'Finite nonnegative provider fee required')
    return result


class Control:
    """Locked, durable all-attempt admission; reservations are never refunded."""
    def __init__(self, root, *, clock=time.time):
        self.root, self.clock = Path(root).resolve(), clock
        need(not self.root.is_relative_to(Path('/gpfs/home/zixianma')),'Home is read-only')

    @contextlib.contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'control.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read(self.root / 'control.json')
            yield state
            write(self.root / 'control.json', state)

    def initialize(self, deadline):
        need(type(deadline) in (int, float) and math.isfinite(deadline) and deadline > self.clock(), 'Future finite deadline required')
        need(not (self.root / 'control.json').exists(), 'Existing campaign needs separate recovery review')
        write(self.root / 'control.json', dict(deadline=deadline, creates={}, sessions={}, arrivals={}, finished={}, released_pair_phases={}, abort=None))

    @staticmethod
    def total(state):
        values=[v.get('admission_maximum_observed',{}) for v in state['sessions'].values()]
        return sum((money(v['browserCost'])+money(v['proxyCost']) for v in values if 'browserCost' in v and 'proxyCost' in v),Decimal(0))

    def snapshot(self):
        with self.locked() as state:
            return json.loads(json.dumps(state))

    def halt(self, reason):
        with self.locked() as state:
            if state['abort'] is None:
                state['abort'] = {'reason': str(reason), 'epoch': self.clock()}

    def check(self, state, *, new_pair=False):
        need(state['abort'] is None, 'Shared pilot halted; no new paid admission')
        need(self.clock() < state['deadline'] - 660, 'Pilot all-attempt wall guard')
        if new_pair:
            need(self.clock() < state['deadline'] - 1860, 'Insufficient owned time for another pair')
            need(self.total(state) < 8, 'Operational $8 next-pair admission threshold reached')
            need(all(v['known'] for v in state['sessions'].values()), 'Provider fee unknown')

    def pair(self, arm, index, *, after=False, sleep=time.sleep):
        need(arm in ARMS and type(index) is int and 0 <= index < 20, 'Exact pair/arm identity required')
        field = 'finished' if after else 'arrivals'
        key = str(index)
        with self.locked() as state:
            self.check(state)
            if not after:
                need(self.clock() < state['deadline']-1860,'Insufficient owned time for another pair')
                if index > 0:
                    need(state.get('released_pair_phases',{}).get('finished:'+str(index-1)) is True,'Previous phase must finish before next admission')
            values = state[field].setdefault(key, [])
            need(arm not in values, 'Duplicate pair transition')
            values.append(arm)
        while True:
            with self.locked() as state:
                self.check(state)
                phase = field+':'+key
                if state.get('released_pair_phases',{}).get(phase) is True:
                    return
                if set(state[field][key]) == set(ARMS):
                    self.require_settled(state)
                    if not after:
                        self.check(state,new_pair=True)
                    # Check the previous phase exactly once while holding this
                    # lock. A fast released peer may now admit the next create.
                    state.setdefault('released_pair_phases',{})[phase] = True
                    return
            sleep(0.1)

    @staticmethod
    def require_settled(state):
        need(len(state['creates']) == len(state['sessions']), 'An admitted create lacks known provider identity/usage')
        need(all(v['known'] and v.get('status') == 'stopped' and v.get('reconciled_epoch') is not None for v in state['sessions'].values()), 'Every previous owned session needs stopped known late fees before pair release')

    def reconciled(self, directory, stopped_epoch, observed_epoch):
        need(all(type(v) in (int,float) and math.isfinite(v) for v in (stopped_epoch,observed_epoch)), 'Finite observed reconciliation times required')
        need(observed_epoch-stopped_epoch >= 90 and observed_epoch <= self.clock()+1, 'At least90 seconds after actual stopped receipt required')
        with self.locked() as state:
            entry = state['sessions'][str(Path(directory).resolve())]
            need(entry['known'] and entry['status'] == 'stopped', 'Late unknown/nonstopped usage cannot release pair')
            entry.update(reconciled_epoch=observed_epoch, first_stopped_receipt_epoch=stopped_epoch)

    def admission(self, directory, payload):
        key = str(Path(directory).resolve())
        with self.locked() as state:
            self.check(state)
            need(self.total(state) < 8, 'Operational cloud admission threshold reached')
            need(key not in state['creates'] and len(state['creates']) < 42, 'No create retry or extra create allowance')
            state['creates'][key] = dict(epoch=self.clock(), payload_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
        return dict(approved=True, proxy_overshoot_authorized=True, creation_retries=0, traffic_guard='best_effort_not_hard')

    def usage(self, directory, view):
        # Never throw from a provider usage callback: cleanup must remain possible.
        try:
            key = str(Path(directory).resolve())
            with self.locked() as state:
                need(key in state['creates'], 'Usage lacks durable create admission')
                known = all(view.get(k) is not None for k in ('browserCost', 'proxyCost', 'proxyUsedMb'))
                previous = state['sessions'].get(key, {})
                current = dict(previous, id=view['id'], known=known, status=view.get('status'), epoch=self.clock())
                current['latest_provider_usage']={field:view.get(field) for field in ('browserCost','proxyCost','proxyUsedMb')}
                current['admission_fee_basis']='maximum_observed_reserve_not_actual_spend'
                if known:
                    maxima=current.setdefault('admission_maximum_observed',{})
                    for field in ('browserCost', 'proxyCost', 'proxyUsedMb'):
                        maxima[field] = str(max(money(view[field]),money(maxima.get(field,0))))
                state['sessions'][key] = current
                if not known and state['abort'] is None:
                    state['abort'] = dict(reason='provider_usage_unknown', epoch=self.clock())
                if self.total(state) >= 10 and state['abort'] is None:
                    state['abort'] = dict(reason='observed_cloud_10_usd_stop_not_hard_cap', epoch=self.clock())
                return state['abort'] is None
        except BaseException:
            self.halt('usage_callback_evidence_error')
            return False


def guard_budget_type(cls, control):
    for method in ('reserve','reserve_when_available','update'):
        if not hasattr(cls,method):
            continue
        original = getattr(cls,method)
        def guarded(self,*args,_original=original,_method=method,**kwargs):
            reservation = _method != 'update' or (args[0] if args else kwargs.get('reserve')) is not None
            if reservation:
                with control.locked() as state:
                    control.check(state)
            return _original(self,*args,**kwargs)
        setattr(cls,method,guarded)


def allocation_deadline(raw, now, job_id=None):
    fields = dict(token.split('=',1) for token in raw.split() if '=' in token)
    need(type(now) in (int,float) and math.isfinite(now),'Actual finite host time required')
    need(fields.get('JobState') == 'RUNNING','Owned allocation must actually be running')
    if job_id is not None:
        need(fields.get('JobId') == job_id,'Returned Slurm job identity differs')
    need(fields.get('NumCPUs') == '2' and fields.get('NumNodes') == '1', 'Approved2CPU/1node allocation required')
    need(fields.get('Restarts','0') == '0', 'A requeue needs separate remaining-time review')
    for name in ('ReqTRES','AllocTRES'):
        tres = dict(x.split('=',1) for x in fields.get(name,'').split(',') if '=' in x)
        need(tres.get('cpu') == '2' and tres.get('mem') in ('16G','16384M'), 'Actual2CPU/16GiB request/allocation required')
        need(all(Decimal(v)==0 for k,v in tres.items() if 'gpu' in k), 'GPU allocation forbidden')
    def epoch(name):
        value = fields[name]
        need(value not in ('Unknown','N/A'), 'Actual allocation start/end required')
        return datetime.fromisoformat(value).timestamp()
    start,end = epoch('StartTime'),epoch('EndTime')
    need(start <= now < end and 0 < end-start <= 14400, 'Original four-hour allocation calendar required')
    return min(end,start+14400)


def prepare(runtime, proposal, output):
    runtime, proposal, output = (Path(p).resolve() for p in (runtime, proposal, output))
    need(not output.is_relative_to(Path('/gpfs/home/zixianma')), 'Home is read-only')
    need(not output.exists() and not output.is_relative_to(runtime), 'New output outside source runtime required')
    document = read(proposal)
    need(document['status'] == 'prepared_not_authorized', 'Proposal is preparation, not an execution approval')
    pairs = document['pairs']
    need(len(pairs) == 20 and [p['pair'] for p in pairs] == list(range(20)), 'Exactly twenty ordered pairs required')
    ids = [p['task_id'] for p in pairs]
    need(len(set(ids)) == 20 and all(set(p['worker_arms']) == set(ARMS) for p in pairs), 'Unique paired tasks and both arms required')
    need(all(type(i) is str and re.fullmatch(r'[A-Za-z0-9_. -]+',i) and i not in ('.','..') for i in ids),'Safe exact task filenames required')
    roster = {t['task_id']: t for t in read(runtime / 'tasks.json')}
    need(all(i in roster for i in ids), 'Task absent from prepared runtime')
    protocol = document['protocol']
    need((protocol['max_decisions'], protocol['max_output_tokens'], protocol['actor_loop_wall_seconds']) == (30, 12288, 900), 'Pilot scientific protocol changed')
    need(protocol['judge'] == dict(model='gpt-5.6-terra', reasoning_effort='medium', max_output_tokens=4096, seed=42), 'Terra protocol changed')
    judge=read(runtime/'runtime-config.json')['judge_protocol']
    need({k:judge.get(k) for k in ('model','reasoning_effort','max_completion_tokens','seed')}==dict(model='gpt-5.6-terra',reasoning_effort='medium',max_completion_tokens=4096,seed=42),'Actual runtime Terra protocol differs')
    limits = document['proposed_limits']
    need(limits['browser_creates_all_attempts'] == 42 and limits['model_http_calls_including_input_counters_all_attempts'] == 8400 and limits['terra_calls_all_attempts'] == 40, 'Physical caps changed')
    need(limits['usd'] == dict(browser='10', n5='3', single='1', terra='2'), 'Requested dollar split changed')
    need(limits['resources'] == dict(cpus=2,gpus=0,memory_gib=16,nodes=1,seconds_all_attempts=14400), 'Resource request changed')
    need(limits['concurrent_browsers'] == 2 and limits['concurrent_generation_http'] == 6, 'Concurrency changed')
    # Importing public preparation helpers makes no browser/SDK/provider call.
    from . import common
    outputs = {}
    for arm in ARMS:
        amount, calls = CAPS[arm]
        args = common.parser('gpt').parse_args(['--runtime',str(runtime),'--arm',arm,'--output',str(output/arm),'--limit','20','--model-usd',amount,'--judge-usd','1','--model-calls',str(calls),'--judge-calls','20','--browser-starts','21','--wandb-mode','disabled'])
        plan, _ = common.prepare(args, 'gpt')
        need(plan['model']=='gpt-6-luna' and plan['candidate_effort']=='high' and plan['max_output_tokens']==12288 and plan['max_decisions']==30,'Exact Luna high12288/30 actor required')
        need(plan['candidate_count']==(1 if arm==ARMS[0] else 5),'Exact single versus five concurrent candidates required')
        if arm==ARMS[1]:
            need(plan['selector_effort']=='high' and plan['selector_max_output_tokens']==12288,'Exact high12288 selector required')
        plan.update(selected_task_ids=ids, task_selection_count=20, task_count=20, pilot_proposal_sha256=digest(proposal))
        outputs[arm] = dict(args=vars(args), plan=plan)
    sources = {str(Path(__file__).with_name(name).resolve()):digest(Path(__file__).with_name(name)) for name in ('stealth_pilot.py','common.py','cloud_browser.py','cloud_browsercode.py')}
    return dict(schema='browsercode-stealth-paired-pilot-execution-plan-v1', runtime=str(runtime), proposal=str(proposal), output=str(output), proposal_sha256=digest(proposal), runtime_manifest_sha256=digest(runtime/'source-manifest.json'), source_sha256=sources, task_ids=ids, arms=outputs, creates_all_attempts=42, model_calls_all_attempts=8400, terra_calls_all_attempts=40, seconds_all_attempts=14400, approved=False)


def approval_check(plan, approval):
    need(approval.get('approved') is True and approval.get('proxy_overshoot_authorized') is True, 'Exact explicit paid/overshoot approval required')
    need(approval.get('plan_sha256') == hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest(), 'Approval must bind the exact prepared plan')
    need(approval.get('openai_payload_egress_approved') is True and approval.get('browser_use_payload_egress_approved') is True, 'Both exact provider payload destinations need approval')


def _worker(arm, plan, env_file, public_control_root):
    for path,expected in plan['source_sha256'].items():
        need(digest(path)==expected,'Executed adapter/owner source changed')
    from . import common, cloud_browsercode
    from .cloud_browser import V4HTTP
    from dotenv import dotenv_values
    control = Control(public_control_root)
    spec = plan['arms'][arm]
    args = argparse.Namespace(**spec['args'])
    for name in ('runtime','output'):
        setattr(args, name, Path(getattr(args,name)))
    args.execute, args.env_file = True, Path(env_file) if env_file else None
    keys = dict(os.environ)
    if env_file:
        keys.update({k:v for k,v in dotenv_values(env_file).items() if v})
    need(keys.get('BROWSER_USE_API_KEY'), 'Browser Use key required after approval')
    tasks_by_id = {t['task_id']:t for t in read(Path(plan['runtime'])/'tasks.json')}
    tasks = [tasks_by_id[i] for i in plan['task_ids']]
    registry_holder = []
    def setup(worker, environments):
        import importlib
        budget = importlib.import_module('request_budget')
        # Guard every physical API admission; settlement remains unchanged.
        for cls in {worker.ReservationBudget, budget.ReservationBudget}:
            guard_budget_type(cls,control)
        registry = cloud_browsercode.install(worker,environments,
            http_factory=lambda:V4HTTP(keys['BROWSER_USE_API_KEY']),
            admission=control.admission,
            metadata_factory=lambda directory:dict(campaign=Path(plan['output']).name, arm=arm, attempt=hashlib.sha256(str(directory).encode()).hexdigest()[:20]),
            on_halt=control.halt,max_proxy_usd='10',on_usage=control.usage,
            judge_budget_type=budget.ReservationBudget)
        original = worker.episode
        original_preflight = worker.preflight
        def settled(directory):
            session = registry.sessions[Path(directory).resolve()]
            session.require_stopped()
            receipt = session.directory/'cloud-private/first-close.json'
            need(receipt.is_file(),'Original stopped receipt missing; preserve unknown')
            stopped = receipt.stat().st_mtime
            remaining = stopped+90-time.time()
            if remaining > 0:
                time.sleep(remaining)
            observed = session.reconcile_usage()
            control.reconciled(directory,stopped,observed['observed_epoch'])
        def preflight(root,browser_budget):
            result = original_preflight(root,browser_budget)
            for directory in list(registry.sessions):
                settled(directory)
            return result
        worker.preflight = preflight
        def paired(root,task,cell,*positional,**kwargs):
            index = plan['task_ids'].index(task['task_id'])
            for pool in positional[2].values():
                if (pool.root/'halt.json').exists():
                    control.halt('model_or_terra_pool_halted')
            control.pair(arm,index)
            original_error = None
            try:
                # This is the logical assignment; fixed process arm is separate metadata.
                kwargs.pop('slot',None)
                values = list(positional)
                if len(values)>3:
                    values[3] = (index + ARMS.index(arm)) % 2
                return original(root,task,cell,*values,**kwargs)
            except BaseException as error:
                original_error = error
                control.halt('episode_interrupted_'+arm)
                raise
            finally:
                if any((pool.root/'halt.json').exists() for pool in positional[2].values()):
                    control.halt('model_or_terra_pool_halted')
                directory=Path(root)/'episodes'/(task['task_id']+'-'+cell['harness']+'-'+cell['policy'])
                try:
                    settled(directory)
                except BaseException as error:
                    write(Path(root)/'pair-late-usage-error.json',dict(task_id=task['task_id'],error_type=type(error).__name__,original_episode_error_type=type(original_error).__name__ if original_error else None,raw_evidence_preserved=True))
                    control.halt('pair_late_usage_incomplete')
                    if original_error is None:
                        raise
                if original_error is None:
                    control.pair(arm,index,after=True)
        worker.episode = paired
        registry_holder.append(registry)
        return registry
    code = 1
    try:
        executed = dict(spec['plan'],execute=True)
        code = common.execute(args,executed,tasks,worker_setup=setup)
    except BaseException:
        control.halt('worker_failed_'+arm)
        raise
    finally:
        # close_all is also called by common. Late billing is a separate query,
        # never presented as a final invoice or used to add browser lifetime.
        for registry in registry_holder:
            errors, reconciled = [], []
            try:
                registry.close_all()
            except BaseException as error:
                errors.append(dict(component='cloud_close',error_type=type(error).__name__))
            for session in registry.sessions.values():
                if session.session_id is None and not session.create_dispatched:
                    continue
                try:
                    session.require_stopped()
                    receipt=session.directory/'cloud-private/first-close.json'
                    need(receipt.is_file(),'Original stopped receipt missing')
                    remaining=receipt.stat().st_mtime+90-time.time()
                    if remaining>0:
                        time.sleep(remaining)
                    reconciled.append(session.reconcile_usage())
                except BaseException as error:
                    errors.append(dict(directory=str(session.directory),component='late_provider_usage',error_type=type(error).__name__))
            if errors:
                control.halt('provider_final_reconciliation_incomplete')
                code = 1
            write(Path(plan['output'])/arm/'postcampaign-provider-usage.json',dict(observations=reconciled,errors=errors,minimum_post_stop_seconds=90,complete=not errors,final_invoice_asserted=False))
    return code


def _child_entry(*args):
    raise SystemExit(_worker(*args))


def run(plan, env_file, approval=None):
    approval_check(plan,approval or {})
    need(os.environ.get('SLURM_JOB_ID','').isdigit(),'This pilot requires the exact approved Slurm allocation; no implicit local mode')
    for path,expected in plan['source_sha256'].items():
        need(digest(path)==expected,'Executed adapter/owner source changed')
    fresh=prepare(plan['runtime'],plan['proposal'],plan['output'])
    need(json.dumps(fresh,sort_keys=True,default=str)==json.dumps(plan,sort_keys=True,default=str),'Prepared source/runtime/proposal/caps changed')
    deadline = time.time()+14400
    allocation = None
    if os.environ.get('SLURM_JOB_ID'):
        job = os.environ['SLURM_JOB_ID']
        need(job.isdigit(),'Exact owned Slurm primary ID required')
        command=['scontrol','show','job','-o',job]
        raw=subprocess.run(command,check=True,text=True,capture_output=True).stdout
        deadline=allocation_deadline(raw,time.time(),job)
        allocation=dict(job_id=job,argv=command,raw=raw,deadline=deadline)
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'execution-plan.json',plan)
    write(root/'execution-approval.json',approval)
    if allocation:
        write(root/'allocation.json',allocation)
    control = Control(root/'shared-control')
    control.initialize(deadline)
    ctx = multiprocessing.get_context('spawn')
    workers = [ctx.Process(target=_child_entry,args=(arm,plan,env_file,str(control.root)),name='owned-pilot-'+arm) for arm in ARMS]
    signalled = set()
    parent_stop = []
    signal.signal(signal.SIGTERM,lambda *_:parent_stop.append(True))
    try:
        for worker in workers:
            worker.start()
        while any(worker.is_alive() for worker in workers):
            if any(worker.exitcode not in (None,0) for worker in workers):
                control.halt('owned_worker_failed')
            if parent_stop or time.time() >= control.snapshot()['deadline'] - 660:
                control.halt('owner_wall_deadline')
                for worker in workers:
                    if worker.is_alive() and worker.pid not in signalled:
                        os.kill(worker.pid,signal.SIGTERM)
                        signalled.add(worker.pid)
            for worker in workers:
                worker.join(timeout=0.25)
    finally:
        if any(worker.is_alive() for worker in workers):
            control.halt('owner_draining')
            for worker in workers:
                if worker.is_alive() and worker.pid not in signalled:
                    os.kill(worker.pid,signal.SIGTERM)
                    signalled.add(worker.pid)
        for worker in workers:
            if worker.pid is not None:
                worker.join()
    result = dict(worker_exitcodes={arm:w.exitcode for arm,w in zip(ARMS,workers)},control=control.snapshot(),all_children_awaited=True,final_invoice_asserted=False)
    write(root/'owner-summary.json',result)
    return 0 if all(w.exitcode==0 for w in workers) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','proposal','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    plan = prepare(args.runtime,args.proposal,args.output)
    if not args.execute:
        print(json.dumps(plan,indent=2,default=str))
        return 0
    need(args.approval is not None,'Exact separate approval receipt required')
    approval_check(plan,read(args.approval))
    return run(plan,str(args.env_file) if args.env_file else None,read(args.approval))


if __name__ == '__main__':
    raise SystemExit(main())
