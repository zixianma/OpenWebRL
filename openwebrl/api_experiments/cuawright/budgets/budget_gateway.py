"""Shard-local physical HTTP meter. Private journals preserve every request/response.

Input reservations bound text by UTF-8 bytes plus serialization overhead, vision by
the previously frozen 32768-token image allowance, and encrypted reasoning by originating output usage.
No remote/file images, server-side conversation references, streaming, or paid tools.
"""
import contextlib
import base64
import fcntl
import hashlib
import http.server
import json
import io
import math
import os
import pathlib
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

PRICE_SOURCE = "https://developers.openai.com/api/docs/guides/images-vision"


def _bytes(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def _save(path, data):
    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _events(folder):
    path = folder / "events.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _append(folder, event):
    with open(folder / "events.jsonl", "ab") as f:
        f.write(_bytes(event) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def _folder(root, shard):
    p = pathlib.Path(root) / ("shard%d" % int(shard)) / ".host" / "meter"
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextlib.contextmanager
def _lock(folder):
    with open(folder / "ledger.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _summary(events):
    ends = {e["id"]: e for e in events if e["event"] == "finish"}
    out = {"model": {"calls": 0, "usd": 0., "unknown": 0},
           "judge": {"calls": 0, "usd": 0., "unknown": 0}, "browser_starts": 0,
           "reservation_exceeded": sum(bool(e.get("reservation_exceeded")) for e in events)}
    for e in events:
        if e["event"] == "browser_start":
            out["browser_starts"] += 1
        if e["event"] != "reserve":
            continue
        t = out[e["pool"]]
        end = ends.get(e["id"], {})
        t["calls"] += 1
        t["usd"] += end.get("charge_usd", e["reserve_usd"])
        t["unknown"] += int(not end.get("usage_known", False))
    return out


def totals(root, shard):
    folder = _folder(root, shard)
    with _lock(folder):
        return _summary(_events(folder))


def _text_transport_escape_excess(payload):
    """Exclude JSON-wire escapes only for decoded Responses textual content.

    Keep two quote bytes plus every structural byte, tools/schema/arguments and
    framing allowance. Do not recursively interpret actor-authored schema data
    as Responses message parts. This changes accounting, never the request.
    """
    def excess(value):
        if not isinstance(value, str):
            return 0
        extra = len(_bytes(value)) - 2 - len(value.encode("utf-8"))
        if extra < 0:
            raise ValueError("Invalid decoded text byte accounting")
        return extra
    items = payload.get("input")
    if not isinstance(items, list):
        return 0
    result = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call_output":
            result += excess(item.get("output"))
        elif item.get("role") in ("user", "assistant", "system", "developer"):
            parts = item.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and part.get("type") in ("input_text", "output_text"):
                    result += excess(part.get("text"))
    return result


def _bound(payload, pool, encrypted, counted_input_upper=None, model="gpt-6-luna", rates=None):
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    model = model if pool == "model" else "gpt-4o-2024-08-06"
    if payload.get("model") != model:
        raise ValueError("Exact approved model required")
    if payload.get("stream") or payload.get("previous_response_id") or payload.get("conversation"):
        raise ValueError("Unmeterable streaming or server-side input reference")
    if payload.get("background") or payload.get("service_tier", "default") not in ("default", "auto"):
        raise ValueError("Unsupported billing mode")
    if any(t.get("type") != "function" for t in payload.get("tools", [])):
        raise ValueError("Only local function tools approved")
    if pool == "judge":
        output = payload.get("max_tokens", payload.get("max_completion_tokens"))
        if output != 512 or payload.get("temperature") != 0:
            raise ValueError("Official judge requires 512 tokens and temperature zero")
    else:
        output = payload.get("max_output_tokens")
        if not isinstance(output, int) or isinstance(output, bool) or not 1 <= output <= 32000:
            raise ValueError("Approved output bound is 32000")
    bonus = 0

    def scrub(x):
        nonlocal bonus
        if isinstance(x, list):
            return [scrub(v) for v in x]
        if not isinstance(x, dict):
            return x
        if x.get("type") in ("input_file", "input_audio", "input_video", "item_reference", "file_search", "web_search", "web_search_preview", "code_interpreter", "computer_use_preview"):
            raise ValueError("Unapproved input or paid hosted tool")
        if x.get("type") in ("input_image", "image_url"):
            image = x.get("image_url")
            detail = x.get("detail", "auto")
            if isinstance(image, dict):
                detail = image.get("detail", detail)
                image = image.get("url")
            if not isinstance(image, str) or not re.fullmatch(r"data:image/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/=\s]+", image):
                raise ValueError("Only inline image data is bounded")
            if detail not in ("low", "high", "original", "auto"):
                raise ValueError("Unknown image detail")
            # Same conservative image allowance as the frozen Luna BrowserCode campaign.
            if pool == "model":
                bonus += 32768
            elif detail == "low":
                bonus += 85
            else:
                from PIL import Image
                data = base64.b64decode(image.split(",", 1)[1])
                with Image.open(io.BytesIO(data)) as im:
                    width, height = im.size
                scale = min(1., 2048. / max(width, height))
                width, height = math.ceil(width * scale), math.ceil(height * scale)
                scale = 768. / min(width, height)
                # Ceil each dimension conservatively at both resize stages.
                bonus += 85 + 170 * math.ceil(math.ceil(width * scale) / 512) * math.ceil(math.ceil(height * scale) / 512)
            return {"type": x["type"], "image_url": "IMAGE"}
        out = {}
        for k, v in x.items():
            if k == "encrypted_content" and v:
                digest = hashlib.sha256(v.encode()).hexdigest()
                if digest not in encrypted:
                    raise ValueError("Encrypted reasoning lineage has no known originating usage")
                bonus += encrypted[digest]
                out[k] = "ENCRYPTED"
            else:
                out[k] = scrub(v)
        return out

    compact = scrub(payload)
    # UTF-8 bytes dominate byte-BPE text tokens; generous message/tool framing.
    upper = len(_bytes(compact)) - _text_transport_escape_excess(compact) + 4096 + bonus
    if counted_input_upper is not None:
        if pool != "model" or type(counted_input_upper) is not int or counted_input_upper < 4096:
            raise ValueError("Invalid native input count bound")
        upper = counted_input_upper
    if pool == "model" and upper > 272000:
        raise ValueError("Input exceeds frozen Luna price tier")
    if pool == "judge" and upper + output > 128000:
        raise ValueError("Conservative input plus output exceeds judge context bound")
    # Treat every noncached input token as cache-write for admission. This
    # dominates ordinary input and avoids an unbounded missing-write receipt.
    rates = (rates["cache_write_input"], rates["output"]) if pool == "model" and rates is not None else (.125, .5) if pool == "model" else (2.5, 10.)
    return (upper * rates[0] + output * rates[1]) / 1e6, upper


def _input_count_payload(payload):
    # Same frozen native counting fields used by the prior Luna BrowserCode
    # campaign. Refuse an unknown potentially input-affecting field rather than
    # count a request different from the generation input.
    fields = ("model", "input", "tools", "text", "reasoning", "tool_choice", "parallel_tool_calls")
    harmless = {"max_output_tokens", "include", "store", "service_tier"}
    if set(payload) - set(fields) - harmless:
        raise ValueError("Unrecognized field for native input counting")
    return {key: payload[key] for key in fields if key in payload}


def _charge(body, pool, rates=None):
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        inp = usage["input_tokens" if pool == "model" else "prompt_tokens"]
        out = usage["output_tokens" if pool == "model" else "completion_tokens"]
        details = usage.get("input_tokens_details" if pool == "model" else "prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (inp, out, cached)) or cached > inp:
            return None
        if pool == "model":
            if inp > 272000:
                return None
            writes = details.get("cache_write_tokens")
            if writes is None:
                writes = inp - cached
            if (not isinstance(writes, int) or isinstance(writes, bool)
                    or writes < 0 or writes + cached > inp):
                return None
            rates = rates or {"input":.1,"cached_input":.01,"cache_write_input":.125,"output":.5}
            return ((inp - cached - writes) * rates["input"] + cached * rates["cached_input"]
                    + writes * rates["cache_write_input"] + out * rates["output"]) / 1e6
        return ((inp - cached) * 2.5 + cached * 1.25 + out * 10.) / 1e6
    except (KeyError, TypeError):
        return None


def _encrypted_bounds(body):
    output = (body.get("usage") or {}).get("output_tokens")
    result = {}
    if isinstance(output, int) and not isinstance(output, bool) and output >= 0:
        def visit(x):
            if isinstance(x, list):
                for v in x:
                    visit(v)
            elif isinstance(x, dict):
                if x.get("encrypted_content"):
                    result[hashlib.sha256(x["encrypted_content"].encode()).hexdigest()] = output
                for v in x.values():
                    visit(v)
        visit(body.get("output", []))
    return result


def _forward(endpoint, payload, key, timeout):
    request = urllib.request.Request("https://api.openai.com/v1/" + endpoint,
                                     data=payload, headers={"Authorization": "Bearer " + key,
                                     "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()



def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _group_holds(events):
    groups = {e['id']: e for e in events if e['event'] == 'group_open'}
    closed = {e['group_id'] for e in events if e['event'] == 'group_finish'}
    holds = {}
    for ident, group in groups.items():
        if ident in closed:
            continue
        used = [e for e in events if e['event'] == 'reserve' and e.get('group_id') == ident]
        holds[ident] = {'calls': group['held_model_calls'] - len(used),
                        'usd': group['held_usd'] - sum(e['reserve_usd'] for e in used)}
        if holds[ident]['calls'] < 0 or holds[ident]['usd'] < -1e-10:
            raise ValueError('Atomic group exceeded its admission credits')
    return holds


def effective_browser_cap(root, shard):
    plan = json.loads((pathlib.Path(root) / 'plan.json').read_text())
    return next(s for s in plan['shards'] if s['shard'] == int(shard))['browser_starts_all_attempts']


def _halted(root, shard, stage=None):
    path = pathlib.Path(root) / ('shard%d' % int(shard)) / 'operational-halt.json'
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get('mode') != 'cleared'
    except (OSError, ValueError):
        return True


class ShardGateway:
    """One durable shard ledger. HTTP/group caller identity is enforced outside actor."""
    def __init__(self, root, shard, deadline, key, group_secret):
        import secrets
        self.root = pathlib.Path(root).resolve(); self.shard = int(shard)
        self.deadline = deadline; self.key = key
        self.folder = _folder(self.root, shard)
        self.plan = json.loads((self.root / 'plan.json').read_text())
        self.cap = next(s for s in self.plan['shards'] if s['shard'] == self.shard)
        self.allowed = set(self.cap['task_ids']); self.model = self.plan['model']
        expected = {'gpt-6-luna': {'input':.1,'cached_input':.01,'cache_write_input':.125,'output':.5},
                    'gpt-6-sol': {'input':2.,'cached_input':.2,'cache_write_input':2.5,'output':10.}}
        self.rates = json.loads((self.root / 'pricing-snapshot.json').read_text())['usd_per_million_tokens']
        if self.model not in expected or self.rates != expected[self.model]:
            raise ValueError('Exact arm model and frozen token rates required')
        if self.cap['model_max_inflight'] != 8:
            raise ValueError('Approved shared model inflight is exactly eight')
        self.inflight = threading.BoundedSemaphore(8)
        self.group_secret = group_secret or secrets.token_hex(32)
        self.group_lock = threading.RLock()
        self.private = self.folder / 'private-groups'; self.private.mkdir(exist_ok=True)
        with _lock(self.folder):
            limits = self.folder / 'limits.json'
            value = {'cap':self.cap,'model':self.model,'rates':self.rates}
            if limits.exists():
                if json.loads(limits.read_text()) != value:
                    raise ValueError('Frozen shard model/pricing/caps changed')
            else:
                _save(limits, _bytes(value))

    def _events(self):
        return _events(self.folder)

    def _halt(self, reason, ident):
        p = self.root / ('shard%d' % self.shard) / 'operational-halt.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and json.loads(p.read_text()).get('mode') != 'cleared':
            return
        temp = p.with_name('halt-' + uuid.uuid4().hex + '.partial')
        _save(temp, _bytes({'mode':'blocked','reason':reason,'physical_id':ident,'time':time.time()}))
        os.replace(temp,p)

    def _check(self, events, stage):
        if time.time() >= self.deadline or _halted(self.root,self.shard,stage):
            raise ValueError('Shard halted or deadline exhausted')
        state = _summary(events)
        if state['reservation_exceeded']:
            raise ValueError('Reservation violation requires independent audit')
        return state

    def _credits(self, events, pool, cost, group_id=None):
        state = self._check(events,pool); holds = _group_holds(events)
        usdkey = 'model_usd_all_attempts' if pool == 'model' else 'external_judge_usd_all_attempts'
        callkey = 'model_calls_all_attempts' if pool == 'model' else 'external_judge_calls_all_attempts'
        if group_id:
            if group_id not in holds or holds[group_id]['calls'] < 1 or holds[group_id]['usd'] + 1e-10 < cost:
                raise ValueError('Atomic group credits unavailable')
            # Spending a held credit exchanges hold for a physical reservation.
            extra_calls = 0
            # Include any sub-ulp negative remainder conservatively in the strict cap check.
            extra_usd = max(0.,cost-holds[group_id]['usd'])
        else:
            extra_calls, extra_usd = 1, cost
        held_calls = sum(h['calls'] for h in holds.values()) if pool == 'model' else 0
        held_usd = sum(h['usd'] for h in holds.values()) if pool == 'model' else 0
        if (state[pool]['calls'] + held_calls + extra_calls > self.cap[callkey]
                or state[pool]['usd'] + held_usd + extra_usd > self.cap[usdkey]):
            raise ValueError('Shard budget exhausted including atomic group holds')

    def _encrypted(self, events):
        encrypted = {}
        for event in events:
            encrypted.update(event.get('encrypted_bounds',{}))
        return encrypted

    def _bound(self, payload, pool, encrypted, counted=None):
        return _bound(payload,pool,encrypted,counted,model=self.model,rates=self.rates)

    def _physical(self, case, stage, endpoint, payload, reserve, upper, group_id=None,
                  role=None, counter=False, generation_raw=None, counter_id=None, counted=None):
        pool = 'judge' if stage == 'judge' else 'model'; raw = _bytes(payload)
        # Do not reserve eight semaphore slots per group: only actual HTTP uses a slot.
        semaphore = self.inflight if pool == 'model' else contextlib.nullcontext()
        acquired = False
        if pool == 'model':
            acquired = self.inflight.acquire(timeout=max(0,self.deadline-time.time()))
            if not acquired: raise ValueError('Model inflight deadline exhausted before dispatch')
        ident = uuid.uuid4().hex
        try:
            with _lock(self.folder):
                events = self._events(); self._credits(events,pool,reserve,group_id)
                _save(self.folder/(ident+'.request.json'),raw)
                event = {'event':'reserve','id':ident,'case':case,'stage':'input_token_count' if counter else stage,
                         'pool':pool,'time':time.time(),'reserve_usd':reserve,'input_token_upper':upper,
                         'request_sha256':hashlib.sha256(raw).hexdigest(),'generation':not counter,
                         'admission_version':'native-input-count-v1' if counter or counter_id else 'decoded-text-byte-v1',
                         'model':self.model if pool=='model' else 'gpt-4o-2024-08-06'}
                if group_id: event.update(group_id=group_id,role=role)
                if counter:
                    _save(self.folder/(ident+'.generation-request.json'),generation_raw)
                    event.update(endpoint='responses/input_tokens',generation_request_sha256=hashlib.sha256(generation_raw).hexdigest(),generation_request_file=ident+'.generation-request.json')
                if counter_id: event.update(input_token_count_id=counter_id,native_input_tokens=counted,input_token_count_request_sha256=hashlib.sha256((self.folder/(counter_id+'.request.json')).read_bytes()).hexdigest())
                _append(self.folder,event)
            try:
                status,result = _forward(endpoint,raw,self.key,max(1,min(600,self.deadline-time.time())))
            except Exception as exc:
                status,result=502,_bytes({'error':{'message':'Transport failed','type':type(exc).__name__}})
            _save(self.folder/(ident+'.response.json'),result)
            try: body=json.loads(result)
            except (ValueError,UnicodeError): body={}
            if not isinstance(body,dict): body={}
            value=body.get('input_tokens') if counter else None
            count_known = counter and status==200 and body.get('object')=='response.input_tokens' and type(value)is int and value>=0
            charge = 0. if count_known else None if counter else _charge(body,pool,self.rates)
            known = count_known if counter else charge is not None
            finish={'event':'finish','id':ident,'time':time.time(),'status':status,'usage_known':known,
                    'charge_usd':reserve if charge is None else charge,'reservation_exceeded':charge is not None and charge>reserve+1e-10,
                    'response_sha256':hashlib.sha256(result).hexdigest(),'usage':None if counter else body.get('usage'),
                    'encrypted_bounds':{} if counter or not known else _encrypted_bounds(body)}
            if counter: finish.update(stage='input_token_count',generation=False,input_tokens_counted=value if count_known else None,
                                      generation_request_sha256=hashlib.sha256(generation_raw).hexdigest(),request_sha256=hashlib.sha256(raw).hexdigest())
            with _lock(self.folder):
                _append(self.folder,finish)
                if not known: self._halt('unknown_physical_usage',ident)
                if finish['reservation_exceeded']:self._halt('reservation_bound_exceeded',ident)
            return {'status':status,'body':body,'raw':result,'physical_request_id':ident,
                    'usage_known':known,'input_tokens_counted':value if count_known else None}
        finally:
            if acquired:self.inflight.release()

    def create(self, case, payload, stage='model', group_id=None, role=None):
        if case not in self.allowed:raise ValueError('Unregistered case')
        pool='judge' if stage=='judge' else 'model'
        with _lock(self.folder):
            events=self._events();self._check(events,stage);encrypted=self._encrypted(events)
        # Orchestration requests retain exact high/20k; native helper overrides remain intact.
        native = pool=='model' and len(payload.get('tools',[]))==1 and payload['tools'][0].get('name')=='run_command'
        if (native or group_id) and (payload.get('reasoning',{}).get('effort')!='high' or payload.get('max_output_tokens')!=20000):
            raise ValueError('Actor/compaction/selector require exact high20000')
        count_id=None;count_value=None
        try:reserve,upper=self._bound(payload,pool,encrypted)
        except ValueError as exc:
            if pool!='model' or str(exc)!='Input exceeds frozen Luna price tier':raise
            raw=_bytes(payload);counter=self._physical(case,stage,'responses/input_tokens',_input_count_payload(payload),0.,None,group_id,role,True,raw)
            count_id=counter['physical_request_id'];count_value=counter['input_tokens_counted']
            if not counter['usage_known']:raise ValueError('Native input count unknown; shard halted')
            reserve,upper=self._bound(payload,pool,encrypted,count_value+4096)
        result=self._physical(case,stage,'chat/completions' if pool=='judge' else 'responses',payload,reserve,upper,
                              group_id,role,counter_id=count_id,counted=count_value)
        result['input_counter_id']=count_id
        return result

    def group_open(self, data):
        case=data['case'];decision=data['decision'];payload=data['candidate_payload']
        if case not in self.allowed or not isinstance(decision,(str,int)) or isinstance(decision,bool):raise ValueError('Invalid group identity')
        count=self.plan.get('candidate_count',3)
        if type(count)is not int or count not in (3,5) or self.plan.get('arm_id')in('sol_high_single','luna_high_single'):raise ValueError('Single arm or unsupported count cannot open proposal groups')
        if data.get('candidate_count',count)!=count:raise ValueError('Requested group candidate count differs from frozen plan')
        generations=count+1;physical_credits=2*generations
        if payload.get('model')!=self.model or payload.get('reasoning',{}).get('effort')!='high' or payload.get('max_output_tokens')!=20000:
            raise ValueError('Group model/effort/output differs')
        # Validate all original input types/lineage even if real counter is needed later.
        with _lock(self.folder):
            events=self._events();state=self._check(events,'model');encrypted=self._encrypted(events)
            try:self._bound(payload,'model',encrypted)
            except ValueError as exc:
                if str(exc)!='Input exceeds frozen Luna price tier':raise
                _input_count_payload(payload)
            holds=_group_holds(events);usd=generations*(272000*self.rates['cache_write_input']+20000*self.rates['output'])/1e6
            if (state['model']['calls']+sum(x['calls']for x in holds.values())+physical_credits>self.cap['model_calls_all_attempts']
                    or state['model']['usd']+sum(x['usd']for x in holds.values())+usd>self.cap['model_usd_all_attempts']):
                raise ValueError('Whole proposal group cannot fit remaining shard budget')
            if any(e['event']=='group_open'and e['case']==case and e['decision']==decision for e in events):raise ValueError('Decision group cannot be retried unchanged')
            ident=uuid.uuid4().hex;raw=_bytes(payload);_save(self.private/(ident+'.candidate.json'),raw)
            event={'event':'group_open','id':ident,'case':case,'decision':decision,'time':time.time(),'held_model_calls':physical_credits,'candidate_count':count,
                   'reserved_generation_calls':generations,'held_usd':usd,'candidate_request_sha256':hashlib.sha256(raw).hexdigest(),
                   'shared_model_inflight_limit':8,'deadline_epoch':self.deadline,'model':self.model}
            _append(self.folder,event)
        return {'group_id':ident,'held_model_calls':physical_credits,'candidate_count':count,'held_usd':usd,'reserved_generation_calls':generations,
                'shared_model_inflight_limit':8,'deadline_epoch':self.deadline,'admitted':True,
                'auxiliary_calls_share_all_attempt_caps':True}

    def _group(self, ident, data=None):
        events=self._events();rows=[e for e in events if e['event']=='group_open'and e['id']==ident]
        if len(rows)!=1:raise ValueError('Unknown group')
        group=rows[0]
        if data is not None:
            if data.get('case',group['case'])!=group['case'] or data.get('decision',group['decision'])!=group['decision']:
                raise ValueError('Group case/decision mismatch')
        return group,events

    def group_create(self,data):
        ident=data['group_id'];role=data['role'];payload=data['payload']
        if role=='candidate':
            index=data['candidate_index']
            if type(index)is not int or index not in range(self.plan.get('candidate_count',3)):raise ValueError('Invalid candidate index')
            role='candidate_'+str(index)
        with self.group_lock,_lock(self.folder):
            group,events=self._group(ident,data);self._check(events,'model')
            if any(e['event']=='group_finish'and e['group_id']==ident for e in events):raise ValueError('Group already finished')
            count=self.plan.get('candidate_count',3)
            if type(count)is not int or count not in (3,5) or group.get('candidate_count',3)!=count:raise ValueError('Group count changed after admission')
            if role not in tuple('candidate_'+str(i)for i in range(count))+('selector',):raise ValueError('Unapproved group role')
            if any(e['event']=='group_create_started'and e['group_id']==ident and e['role']==role for e in events):raise ValueError('Group role already dispatched; no unchanged retry')
            if role.startswith('candidate_'):
                if _bytes(payload)!=(self.private/(ident+'.candidate.json')).read_bytes():raise ValueError('Candidate payload differs from common predecision')
            else:
                done={e.get('role') for e in events if e['event']=='group_create_finished'and e['group_id']==ident and e['known_success']}
                if not {'candidate_'+str(i)for i in range(count)}<=done:raise ValueError('Selector requires three settled known candidates'if count==3 else'Selector requires five settled known candidates')
            _append(self.folder,{'event':'group_create_started','group_id':ident,'role':role,'time':time.time()})
        try:
            value=self.create(group['case'],payload,'model',ident,role)
            result={k:v for k,v in value.items()if k!='raw'}
            success=value['status']==200 and value['usage_known']
        except BaseException as exc:
            with _lock(self.folder):_append(self.folder,{'event':'group_create_finished','group_id':ident,'role':role,'time':time.time(),'known_success':False,'error_type':type(exc).__name__})
            raise
        with _lock(self.folder):_append(self.folder,{'event':'group_create_finished','group_id':ident,'role':role,'time':time.time(),'known_success':success,'physical_request_id':value['physical_request_id'],'input_counter_id':value['input_counter_id']})
        return result

    def group_finish(self,data):
        ident=data['group_id'];selected=data.get('selected_candidate_index');error=data.get('error_type')
        with self.group_lock,_lock(self.folder):
            group,events=self._group(ident,data)
            if any(e['event']=='group_finish'and e['group_id']==ident for e in events):raise ValueError('Group finish is immutable')
            started=[e for e in events if e['event']=='group_create_started'and e['group_id']==ident]
            ended=[e for e in events if e['event']=='group_create_finished'and e['group_id']==ident]
            if {e['role']for e in started}!={e['role']for e in ended}:raise ValueError('Drain every dispatched sibling before settlement')
            if error is None:
                count=group.get('candidate_count',3)
                if type(count)is not int or count not in (3,5) or type(selected)is not int or selected not in range(count) or len(ended)!=count+1 or not all(e['known_success']for e in ended):raise ValueError('Successful decision requires all known generations and selected index')
            elif not isinstance(error,str)or not error or selected is not None:raise ValueError('Failed decision must have error_type and no selected action')
            physical=[e for e in events if e['event']=='reserve'and e.get('group_id')==ident]
            settled={e['id']for e in events if e['event']=='finish'}
            if any(e['id']not in settled for e in physical):raise ValueError('Physical receipt settlement pending; retain group holds')
            hold=_group_holds(events)[ident]
            event={'event':'group_finish','group_id':ident,'case':group['case'],'decision':group['decision'],'time':time.time(),
                   'selected_candidate_index':selected,'error_type':error,'released_unused_calls':hold['calls'],'released_unused_usd':hold['usd'],
                   'actual_physical_ids':[e['id']for e in events if e['event']=='reserve'and e.get('group_id')==ident],
                   'all_dispatched_siblings_settled':True,'unknown_reservations_retained':True}
            _append(self.folder,event)
        return dict(event,verified=True)

    def group_record(self,data):
        ident=data['group_id'];kind=data['kind'];identity=data.get('identity');record=data.get('record',data.get('data'))
        if '/'in kind:
            kind,embedded=kind.split('/',1)
            if identity is not None and identity!=embedded:raise ValueError('Private record identity mismatch')
            identity=embedded
        if kind not in ('physical','decisions')or not isinstance(identity,str)or not re.fullmatch(r'[A-Za-z0-9_.-]{1,200}',identity):raise ValueError('Invalid immutable record identity')
        with self.group_lock,_lock(self.folder):
            group,events=self._group(ident,data)
            raw=_canonical(record);path=self.private/(ident+'.'+kind+'.'+identity+'.json')
            if len(raw)>150000000:raise ValueError('Record too large')
            _save(path,raw);digest=hashlib.sha256(raw).hexdigest()
            _append(self.folder,{'event':'group_private_record','group_id':ident,'case':group['case'],'decision':group['decision'],'kind':kind,'identity':identity,'sha256':digest,'time':time.time()})
        return {'stored':True,'record_sha256':digest}


def start_gateway(root, shard, deadline, key, group_secret=None):
    import hmac
    service=ShardGateway(root,shard,deadline,key,group_secret)
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,value):
            data=value if isinstance(value,bytes)else _bytes(value);self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<=length<=150000000:raise ValueError('Invalid body length')
                raw=self.rfile.read(length);data=json.loads(raw or b'{}')
                if self.path.startswith('/group/'):
                    if not hmac.compare_digest(self.headers.get('X-Campaign-Group-Secret',''),service.group_secret):return self.send(403,{'error':{'message':'Private group authorization required'}})
                    operation=self.path.removeprefix('/group/')
                    if operation not in ('open','create','finish','record'):raise ValueError('Unknown private group operation')
                    return self.send(200,getattr(service,'group_'+operation)(data))
                match=re.fullmatch(r'/(model|helper|judge|browser)/([A-Za-z0-9_-]+)/(responses|chat/completions|reserve|failure)',self.path)
                if not match or match[2]not in service.allowed:raise ValueError('Unregistered route or case')
                stage,case,endpoint=match.groups()
                if stage=='browser':
                    if endpoint not in ('reserve','failure'):raise ValueError('Browser route mismatch')
                    with _lock(service.folder):
                        events=service._events()
                        if endpoint=='failure':
                            if not any(e['event']=='browser_start'and e['id']==data.get('reserved_browser_id')and e['case']==case for e in events):raise ValueError('Browser failure unbound')
                            service._halt('browser_startup_failure',data['reserved_browser_id']);return self.send(200,{'recorded':True,'halted':True})
                        state=service._check(events,'browser')
                        if state['browser_starts']>=service.cap['browser_starts_all_attempts']:raise ValueError('Browser starts exhausted')
                        ident=uuid.uuid4().hex;_append(service.folder,{'event':'browser_start','id':ident,'case':case,'time':time.time()})
                    return self.send(200,{'reserved':True,'id':ident})
                if endpoint!=('chat/completions'if stage=='judge'else'responses'):raise ValueError('Route mismatch')
                value=service.create(case,data,stage)
                return self.send(value['status'],value['raw'])
            except (ValueError,TypeError,KeyError,OSError)as exc:return self.send(403,{'error':{'message':str(exc)}})
    class Server(http.server.ThreadingHTTPServer):
        request_queue_size=64
    server=Server(('127.0.0.1',0),Handler);server.daemon_threads=True
    server.service=service;server.group_secret=service.group_secret
    server.group_base_url='http://127.0.0.1:%d'%server.server_port
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server,server.group_base_url
