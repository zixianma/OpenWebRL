import asyncio
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.harmful_native_pilot import ProtocolError, canonical, digest
from openwebrl.harmful_native_adapters import (
    LocalActor, LocalHead, NativeGUI, NativePromptBuilder, PreparedPrompt, ScreenshotStore,
    TokenizedReply, local_endpoint, FrozenFeatureEngine,
)


CONFIG = dict(text_config=dict(hidden_size=8, num_attention_heads=2, head_dim=4,
    num_key_value_heads=1, intermediate_size=16, num_hidden_layers=2, vocab_size=32),
    vision_config=dict(hidden_size=8, intermediate_size=16, depth=2, spatial_merge_size=2,
        in_channels=3, temporal_patch_size=2, patch_size=2, out_hidden_size=8, deepstack_visual_indexes=[1]))


class Tensor:
    def __init__(self, value): self.value = value; self.shape = (len(value), len(value[0]))
    def tolist(self): return self.value


class Page:
    def __init__(self):
        self.url = 'http://127.0.0.1:8000/orders'; self.viewport_size = dict(width=1280,height=1000)
        self.calls = []
        self.context = Context(self)
        async def click(*args,**kwargs): self.calls.append(('click',args,kwargs))
        async def move(*args,**kwargs): self.calls.append(('move',args,kwargs))
        async def wheel(*args,**kwargs): self.calls.append(('wheel',args,kwargs))
        async def press(*args,**kwargs): self.calls.append(('press',args,kwargs))
        async def insert(*args,**kwargs): self.calls.append(('insert',args,kwargs))
        self.mouse = SimpleNamespace(click=click,move=move,wheel=wheel)
        self.keyboard = SimpleNamespace(press=press,insert_text=insert)
    async def screenshot(self,**kwargs): return b'\x89PNG\r\n\x1a\nfixture'
    async def title(self): return 'Native orders'
    def locator(self, value):
        async def text(): return 'Only observed page text'
        return SimpleNamespace(inner_text=text)
    async def goto(self,*args,**kwargs): self.calls.append(('goto',args,kwargs))
    async def go_back(self,*args,**kwargs): self.calls.append(('back',args,kwargs))


class Context:
    def __init__(self,page): self.pages=[page];self.service_workers=[];self.routes=[];self.sockets=[]
    async def route(self,pattern,handler): self.routes.append(handler)
    async def route_web_socket(self,pattern,handler): self.sockets.append(handler)


class GuiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.events=[];self.page=Page();self.store=ScreenshotStore(Path(self.temp.name)/'screens')
        self.gui=NativeGUI(page=self.page,world_id='world',application_url=self.page.url,
            screenshots=self.store,journal=lambda kind,data:self.events.append((kind,data)))
        await self.gui.install_guards()

    async def test_normalized_click_and_literal_write_without_code_execution(self):
        obs=await self.gui.observe()
        self.assertEqual(self.store.get(asdict(obs))['visible_text'],'Only observed page text')
        await self.gui.execute(dict(name='click',arguments=dict(point_2d=[500,250])))
        self.assertEqual(self.page.calls[0],('click',(640.0,250.0),dict(button='left',click_count=1)))
        literal="__import__('os').system('not executed')"
        await self.gui.execute(dict(name='write',arguments=dict(message=literal)))
        self.assertEqual(self.page.calls[-1],('insert',(literal,),{}))
        self.assertEqual(sum(e[0]=='gui_subcall_dispatched' for e in self.events),4)

    async def test_external_egress_and_generated_script_tools_blocked(self):
        outcomes=[]
        async def yes():outcomes.append('allowed')
        async def no():outcomes.append('blocked')
        for url in ('https://external.test/',self.page.url,'javascript:alert(1)'):
            route=SimpleNamespace(request=SimpleNamespace(url=url),continue_=yes,abort=no)
            await self.page.context.routes[0](route)
        self.assertEqual(outcomes,['blocked','allowed','blocked'])
        for action in (dict(name='evaluate',arguments=dict(script='alert(1)')),
                       dict(name='goto_url',arguments=dict(url='http://elsewhere.test'))):
            with self.assertRaises(ProtocolError):await self.gui.execute(action)
        self.assertEqual(self.page.calls,[])

    async def test_popups_or_existing_service_worker_are_unknown(self):
        self.page.context.pages.append(object())
        with self.assertRaises(ProtocolError):await self.gui.observe()
        self.page.context.pages=[self.page];self.page.context.service_workers=[object()]
        with self.assertRaises(ProtocolError):await self.gui.install_guards()


class FakePromptAdapter:
    def __init__(self):
        def template(messages,tools,**kwargs):
            assert kwargs==dict(add_generation_prompt=True,enable_thinking=False)
            return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
        self.token_handler=SimpleNamespace(apply_chat_template=template)
    def message(self,text,native):
        return dict(role='user',content=[dict(type='text',text=text+' screenshot:\n<|vision_start|><|image_pad|><|vision_end|>'),
            dict(type='image_url',image_url='data:image/png;base64,fixture')])
    def _get_latest_user_message(self,task,native,step_id):return self.message(task,native)
    def _convert_tool_responses_to_msgs(self,responses,observation):
        return [self.message(' '.join(r['tool_response'] for r in responses),observation)]
    def _process_multimodal_messages(self,messages):
        images=[];out=[]
        for msg in messages:
            if isinstance(msg['content'],str):out.append(msg);continue
            images.extend(p['image_url'] for p in msg['content'] if p['type']=='image_url')
            out.append(dict(role=msg['role'],content=' '.join(p['text'] for p in msg['content'] if p['type']=='text')))
        return out,images


class PromptTests(unittest.TestCase):
    def test_identical_screenshot_keeps_only_latest_and_only_selected_reasoning(self):
        with tempfile.TemporaryDirectory() as temp:
            store=ScreenshotStore(temp)
            obs=store.put('world',dict(screenshot=b'\x89PNG\r\n\x1a\nfixture',active_tab_url='http://127.0.0.1:8000',visible_text='page'))
            selected='<think>selected thought</think><tool_call>{"name":"wait","arguments":{}}</tool_call>'
            unselected='<think>SECRET UNSELECTED THOUGHT</think><tool_call>{"name":"wait","arguments":{}}</tool_call>'
            events=[dict(kind='observation_received',data=dict(observation=asdict(obs))),
                dict(kind='model_reply',data=dict(role='actor',reply=dict(text=unselected))),
                dict(kind='model_reply',data=dict(role='actor',reply=dict(text=selected))),
                dict(kind='selected',data=dict(proposal_sha256=digest(selected))),
                dict(kind='browser_reply',data=dict(response=dict(text='observed response'))),
                dict(kind='observation_received',data=dict(observation=asdict(obs)))]
            def processor(**kwargs):return dict(input_ids=Tensor([[1,2,3,4]]),image_grid_thw=Tensor([[1,4,4]]))
            builder=NativePromptBuilder(adapter=FakePromptAdapter(),processor=processor,policy='native policy',tools=[{}],
                screenshots=store,ledger=SimpleNamespace(events=events))
            image=SimpleNamespace(open=lambda stream:SimpleNamespace(convert=lambda mode:object()))
            with patch.dict('sys.modules',{'PIL':SimpleNamespace(Image=image)}):
                result=builder.prepare(canonical(dict(task='user task',observation=asdict(obs))))
            self.assertIn('selected thought',result.text)
            self.assertNotIn('SECRET',result.text)
            self.assertEqual(result.text.count('<|image_pad|>'),1)
            self.assertTrue(result.text.endswith('<think>\n'))


class ModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.prepared=PreparedPrompt('prompt',dict(input_ids=Tensor([[1,2,3,4]])),
            'data:image/png;base64,png',[1,4,4],4,'a'*64)
        self.prompts=SimpleNamespace(prepare=lambda request:self.prepared)
        self.receipts=[]

    async def test_actor_preserves_exact_generation_ids_and_local_transport(self):
        posts=[]
        value=dict(text='native completion',meta_info=dict(prompt_tokens=4,completion_tokens=2,
            output_token_logprobs=[[-.1,101],[-.2,202]],finish_reason=dict(type='stop'),cached_tokens=0))
        async def post(url,json):
            posts.append((url,json));return SimpleNamespace(raise_for_status=lambda:None,json=lambda:value)
        actor=LocalActor(endpoint='http://127.0.0.1:8010',client=SimpleNamespace(post=post),prompts=self.prompts,
            config=CONFIG,journal=lambda k,v:self.receipts.append((k,v)))
        reply=await actor.generate(canonical(dict(state={},seed=43)))
        self.assertIsInstance(reply,TokenizedReply)
        self.assertEqual(reply.token_ids,(101,202))
        self.assertEqual(self.receipts[0][1]['processor_input_ids'],[1,2,3,4])
        self.assertEqual(posts[0][1]['sampling_params']['sampling_seed'],43)
        self.assertFalse(actor.tainted)

    async def test_length_completion_retains_actual_usage_but_cannot_execute(self):
        value=dict(text='partial',meta_info=dict(prompt_tokens=4,completion_tokens=1,
            output_token_logprobs=[[-.1,101]],finish_reason=dict(type='length')))
        async def post(*args,**kwargs):return SimpleNamespace(raise_for_status=lambda:None,json=lambda:value)
        actor=LocalActor(endpoint='http://127.0.0.1:8010',client=SimpleNamespace(post=post),prompts=self.prompts,
            config=CONFIG,journal=lambda *args:None)
        reply=await actor.generate(canonical(dict(state={},seed=43)))
        self.assertFalse(reply.usable);self.assertTrue(reply.usage_complete)
        self.assertEqual(reply.output_tokens,1)
        with self.assertRaises(ProtocolError):await actor.generate(canonical(dict(state={},seed=43)))

    async def test_head_refuses_retokenized_or_wrong_state_candidates(self):
        head=LocalHead(endpoint='http://127.0.0.1:8011',client=None,prompts=self.prompts,config=CONFIG,scorer_identity={})
        for values in ([dict(text='candidate')]*5,[dict(token_ids=[1],prompt_sha256='wrong',screenshot_sha256='a'*64)]*5):
            with self.assertRaises(ProtocolError):await head.select('{}',canonical(values))
        bound=head.bound('{}',5)
        self.assertEqual(bound['output_tokens'],0)
        self.assertEqual(bound['input_tokens'],4+5*1025)

    def test_no_external_model_endpoint(self):
        for value in ('https://api.example.com','http://localhost:8010','http://127.0.0.1:8010/path',
                      'http://user:pass@127.0.0.1:8010'):
            with self.assertRaises(ProtocolError):local_endpoint(value)


class FeatureDeviceTests(unittest.TestCase):
    def test_accelerator_events_follow_sidecar_device_and_cover_feature_and_head_work(self):
        events = []
        active = ['cuda:0']
        stream = object()
        test = self

        class Device:
            def __init__(self, device): self.device = device
            def __enter__(self):
                self.previous = active[0]; active[0] = self.device
                events.append(('enter', self.device))
            def __exit__(self, *args):
                events.append(('exit', active[0])); active[0] = self.previous

        class Event:
            def __init__(self, enable_timing):
                test.assertTrue(enable_timing)
                test.assertEqual(active[0], 'cuda:1')
                self.device = active[0]
            def record(self, selected_stream):
                test.assertIs(selected_stream, stream)
                test.assertEqual(active[0], self.device)
                events.append(('record', self.device))
            def elapsed_time(self, end):
                test.assertEqual((self.device, end.device), ('cuda:1', 'cuda:1'))
                return 125

        def current_stream(device):
            self.assertEqual((device, active[0]), ('cuda:1', 'cuda:1'))
            return stream

        def synchronize(device):
            self.assertEqual((device, active[0]), ('cuda:1', 'cuda:1'))
            events.append(('synchronize', device))

        class Feature:
            def flatten(self, *args): return self
            def __getitem__(self, key): return self

        class Prefix(dict):
            def to(self, device):
                test.assertEqual(device, 'cuda:1')
                return self

        class Head:
            config = SimpleNamespace(listwise=False, layers=(18,27,36),
                pools=('thought_mean','action_mean','last'), d_model=512)
            def eval(self): pass
            def __call__(self, *args):
                test.assertEqual(active[0], 'cuda:1'); events.append(('head', active[0]))
                return SimpleNamespace(argmax=lambda dim:SimpleNamespace(item=lambda:2)), None

        def extract(*args, **kwargs):
            self.assertEqual(active[0], 'cuda:1'); events.append(('extract', active[0]))
            return Feature(), Feature()

        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        class Processor:
            tokenizer = processor.tokenizer
            def __call__(self, **kwargs):
                return Prefix(input_ids=Tensor([[1,2,3,4]]), image_grid_thw=Tensor([[1,4,4]]))
        torch = SimpleNamespace(cuda=SimpleNamespace(device=Device, current_stream=current_stream,
            Event=Event, synchronize=synchronize), inference_mode=nullcontext,
            isfinite=lambda logits:SimpleNamespace(all=lambda:True))
        features = SimpleNamespace(extract=extract, candidate_tokens_from_ids=lambda tokenizer,ids:(ids, {}))
        pil = SimpleNamespace(Image=SimpleNamespace(open=lambda stream:SimpleNamespace(convert=lambda mode:object())))
        engine = FrozenFeatureEngine(model=SimpleNamespace(eval=lambda:None), processor=Processor(),
            head=Head(), config=CONFIG, identity={'frozen':'fixture'}, device='cuda:1')
        with patch.dict('sys.modules', {'torch':torch, 'PIL':pil,
                                       'openwebrl.selection_head_features':features}):
            result = engine.score(dict(scorer_identity={'frozen':'fixture'}, candidates=[[1,2]]*5,
                image_data='data:image/png;base64,cG5n', prompt='prompt', prompt_tokens=4))
        self.assertEqual(events, [('enter','cuda:1'), ('record','cuda:1'), ('extract','cuda:1'),
            ('head','cuda:1'), ('record','cuda:1'), ('synchronize','cuda:1'), ('exit','cuda:1')])
        self.assertEqual(active[0], 'cuda:0')
        self.assertEqual(result['reply']['accelerator_seconds'], .125)


if __name__=='__main__':unittest.main()
