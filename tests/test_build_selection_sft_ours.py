"""CPU tests for scripts/build_selection_sft_ours.py: a built SFT sample's prompt is byte-identical to what the crossbench
``piotr5`` path sends to the selector's sglang server for the same state.

The inference side is the unmodified code path: ``CrossbenchSelector`` (selection kind) builds the /select payload from
the live-state inputs, and ``scripts/serve_arm_sglang.py serve`` renders it (its sglang client and uvicorn are stubbed
to capture the text it would POST). The training side is the built ShareGPT sample rendered with LLaMA-Factory's
``qwen3_vl`` template; the LLaMA-Factory test (run with the venv-llamafactory interpreter) uses the real template,
multimodal plugin and tokenizer.
"""
import asyncio
import base64
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

from openwebrl import crossbench_selector_eval as core

REPO = Path(__file__).resolve().parents[1]
PANELS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/supervision-ablation-20261017/panels/panels.jsonl')


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / 'scripts'))
    spec.loader.exec_module(module)
    return module


ours = load_script('build_selection_sft_ours')
HAVE_ASSETS = PANELS.exists() and ours.BASE.exists() and (ours.SOURCE_ROOT / 'inference/selection_prompt.py').exists()


def sample_panel(index=0):
    with PANELS.open() as stream:
        for i, line in enumerate(stream):
            if i == index:
                return json.loads(line)


def crossbench_payload(panel, screenshot):
    """The payload CrossbenchSelector('selection') POSTs to /select at this state (actor outputs = panel candidates)."""
    captured = []

    async def request_fn(endpoint, payload, timeout):
        captured.append(payload)
        return {'raw': '{"selection": 1}'}

    async def infer(url, text, params, images, timeout_secs=None):
        j = len(calls)
        calls.append(params)
        return panel['candidates'][j], [], [], 'stop'

    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        selector = core.CrossbenchSelector('selection', tmp, seed=42, candidates=5, endpoint='http://selector.test',
                                           request_fn=request_fn)
        asyncio.run(selector(infer=infer, url='http://actor.test', input_text='actor prompt', sampling_params={},
                             images=[], observation={'active_tab_url': panel['url'], 'screenshot': screenshot},
                             history=panel['history'], task=panel['task'], task_id='t', turn=0, timeout=10))
    assert len(captured) == 1
    return captured[0]


def serve_sglang_text(payload):
    """Run serve_arm_sglang's /select handler with a stub sglang client; return the exact text it would send."""
    serve_mod = load_script('serve_arm_sglang')
    sent, apps = [], []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'text': '{"selection": 1}', 'meta_info': {'prompt_tokens': 0, 'completion_tokens': 0}}

    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def post(self, url, json=None):
            sent.append(json)
            return Response()

    import httpx
    import uvicorn
    args = type('A', (), dict(sglang='http://sglang.test', port=0, timeout=10, max_new_tokens=128))()
    with mock.patch.object(httpx, 'AsyncClient', FakeAsyncClient), \
            mock.patch.object(uvicorn, 'run', lambda app, **k: apps.append(app)):
        serve_mod.serve(args)
        select = next(r.endpoint for r in apps[0].routes if getattr(r, 'path', None) == '/select')
        asyncio.run(select(payload))
    assert len(sent) == 1 and sent[0]['image_data'] == ['data:image/png;base64,' + payload['screenshot']]
    return sent[0]['text']


class TargetRules(unittest.TestCase):
    def test_argmax_ties_and_all_equal(self):
        self.assertEqual(ours.target_labels([0, 1, 0, 0, 0]), ([1], None))
        self.assertEqual(ours.target_labels([0, .5, .5, 0, 0]), ([1, 2], None))
        self.assertEqual(ours.target_labels([0, .5, .5, 0, 0], 'first'), ([1], None))
        self.assertEqual(ours.target_labels([0, .5, .5, 0, 0], 'skip'), (None, 'tied_target'))
        self.assertEqual(ours.target_labels([.2] * 5), (None, 'all_equal_target'))
        self.assertEqual(ours.target_labels([1 / 3] * 3 + [0, 0]), ([0, 1, 2], None))
        self.assertEqual(ours.selection_text(2), '{"selection": 3}')

    def test_image_tokens_match_between_training_and_serving_for_screenshots(self):
        self.assertEqual(ours.image_tokens(1280, 720), (252, 252))


@unittest.skipUnless(HAVE_ASSETS, 'branch panels, base model or pinned ARM source not available')
class PromptEquivalence(unittest.TestCase):
    def test_built_sample_prompt_is_byte_identical_to_piotr5_inference(self):
        for index in (0, 7):
            panel = sample_panel(index)
            shot = Path(panel['image']).read_bytes()
            payload = crossbench_payload(panel, shot)
            self.assertEqual(payload, ours.selector_payload(panel, shot))
            inference = serve_sglang_text(payload)
            builder = ours.load_selection_builder(ours.SOURCE_ROOT)
            messages = ours.prompt_messages(builder, ours.selector_payload(panel, shot))
            # piotr format (arm_lora.yaml, template qwen3_vl): the server appends the base chat template's
            # generation prefix "<think>\n" -- the same relation as Piotr's released model under piotr5
            self.assertEqual(ours.render_qwen3_vl(messages) + '<think>\n', inference)
            self.assertEqual(inference.count('<|image_pad|>'), 1)
            # serving-exact format (qwen3_vl_nothink): the training sequence = serving prompt + selection tokens
            label = ours.selection_text(2, 'serving-exact')
            full = ours.render_qwen3_vl(messages, label, 'serving-exact')
            self.assertEqual(full, inference + '{"selection": 3}<|im_end|>\n')

    def test_build_end_to_end(self):
        panels = [sample_panel(i) for i in range(3)]
        targets = {'sources': ['teacher_k1'], 'states': {
            panels[0]['state_id']: dict(task=panels[0]['task_id'], fold=0, targets={'teacher_k1': [0, 0, 1, 0, 0]}),
            panels[1]['state_id']: dict(task=panels[1]['task_id'], fold=2, targets={'teacher_k1': [.5, 0, .5, 0, 0]}),
            panels[2]['state_id']: dict(task=panels[2]['task_id'], fold=3, targets={'teacher_k1': [.2] * 5})}}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'panels.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in panels))
            (tmp / 'targets.json').write_text(json.dumps(targets))
            args = type('A', (), dict(panels=tmp / 'panels.jsonl', targets=tmp / 'targets.json', source='teacher_k1',
                                      out_dir=tmp / 'out', name='t', test_fold=0, all_folds=False, ties='all',
                                      source_root=ours.SOURCE_ROOT, tokenizer=str(ours.BASE), format='piotr',
                                      cutoff_len=ours.CUTOFF_LEN))()
            summary = ours.build(args)
            train = json.loads((tmp / 'out/t_train.json').read_text())
            test = json.loads((tmp / 'out/t_test.json').read_text())
            info = json.loads((tmp / 'out/dataset_info.json').read_text())
            self.assertEqual([s['messages'][2]['content'] for s in train], ['{"selection": 1}', '{"selection": 3}'])
            self.assertEqual(train[0]['messages'][:2], ours.prompt_messages(
                ours.load_selection_builder(ours.SOURCE_ROOT),
                ours.selector_payload(panels[1], Path(panels[1]['image']).read_bytes())))
            self.assertEqual([s['messages'][2]['content'] for s in test], ['{"selection": 3}'])
            self.assertEqual(summary['counts']['train/skipped_all_equal_target'], 1)
            self.assertEqual(summary['counts']['train/tied_states'], 1)
            self.assertEqual(info['t']['file_name'], 't_train.json')
            self.assertTrue((tmp / 'out' / train[0]['images'][0]).read_bytes() == Path(panels[1]['image']).read_bytes())
            args.all_folds, args.ties = True, 'skip'
            summary = ours.build(args)
            self.assertEqual(summary['counts']['train/examples'], 1)
            self.assertEqual(summary['counts']['train/skipped_tied_target'], 1)


try:
    import llamafactory  # noqa: F401
    HAVE_LF = True
except ImportError:
    HAVE_LF = False


@unittest.skipUnless(HAVE_ASSETS and HAVE_LF, 'needs venv-llamafactory (run with its interpreter)')
class LlamaFactoryEncoding(unittest.TestCase):
    def test_llamafactory_prompt_ids_equal_inference_prompt_ids(self):
        import io
        from PIL import Image
        from transformers import AutoProcessor
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments
        panel = sample_panel(0)
        shot = Path(panel['image']).read_bytes()
        inference = serve_sglang_text(crossbench_payload(panel, shot))
        builder = ours.load_selection_builder(ours.SOURCE_ROOT)
        messages = ours.prompt_messages(builder, ours.selector_payload(panel, shot))
        # inference side: the selector server's processor (min 65,536 / max 262,144 pixels)
        serve_proc = AutoProcessor.from_pretrained(ours.BASE, local_files_only=True, min_pixels=65536, max_pixels=262144)
        image = Image.open(io.BytesIO(shot)).convert('RGB')
        serve_ids = serve_proc(text=[inference], images=[image], return_tensors='pt')['input_ids'][0].tolist()
        for fmt, expected_response in (('piotr', '<think>\n\n</think>\n\n{"selection": 3}<|im_end|>\n'),
                                       ('serving-exact', '<think>\n{"selection": 3}<|im_end|>\n')):
            with self.subTest(fmt=fmt):
                # training side: what SupervisedDatasetProcessor does (mm_plugin.process_messages, encode_multiturn)
                processor = AutoProcessor.from_pretrained(ours.BASE, local_files_only=True)
                tokenizer = processor.tokenizer
                processor.image_max_pixels, processor.image_min_pixels = ours.TRAIN_IMAGE_MAX_PIXELS, 32 * 32
                processor.video_max_pixels, processor.video_min_pixels = 16384, 16 * 16
                processor.video_fps, processor.video_maxlen = 2.0, 128
                template = get_template_and_fix_tokenizer(
                    tokenizer, DataArguments(template=ours.FORMATS[fmt]['template']))
                convo = [messages[1], {'role': 'assistant', 'content': ours.selection_text(2, fmt)}]
                convo = template.mm_plugin.process_messages(convo, [panel['image']], [], [], processor)
                (prompt_ids, response_ids), = template.encode_multiturn(tokenizer, convo, messages[0]['content'])
                prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                self.assertEqual(re.sub(r'(<\|image_pad\|>)+', '<|image_pad|>', prompt) + '<think>\n', inference)
                self.assertEqual(tokenizer.decode(response_ids, skip_special_tokens=False), expected_response)
                think = tokenizer('<think>\n', add_special_tokens=False)['input_ids']
                self.assertEqual(prompt_ids + think, serve_ids)                 # same visual token count too
                if fmt == 'serving-exact':
                    self.assertEqual((prompt_ids + response_ids)[:len(serve_ids)], serve_ids)
                self.assertEqual(prompt.count('<|image_pad|>'), ours.image_tokens(*image.size)[0])

if __name__ == '__main__':
    unittest.main()
