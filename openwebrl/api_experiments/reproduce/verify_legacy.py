"""Run a zero-paid-call diagnostic through the real legacy browser/policy loop.

Every hosted actor, selector and judge SDK call terminates at an in-process fake.
Chromium, tokenizer/processor, native history, metering, saved dispositions and
owned browser teardown use the same frozen implementation as real collection.
The local data-URL task and diagnostic outcomes are not benchmark results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import legacy
import legacy_runtime


class OfflineResponse:
    def __init__(self, request):
        self.model = request['model']
        self.status = 'completed'
        self.usage = SimpleNamespace(model_dump=lambda: {
            'input_tokens': 100, 'output_tokens': 10, 'input_tokens_details': {'cached_tokens': 0},
            'output_tokens_details': {'reasoning_tokens': 0}})
        self.output_text = '{"selected_index":0}'
        if 'text' in request:
            self.output = [{'type': 'message', 'content': [{'type': 'output_text', 'text': self.output_text}]}]
        else:
            self.output = [{'type': 'function_call', 'name': 'done', 'arguments': '{"response":"ready"}',
                            'call_id': 'offline-' + uuid.uuid4().hex}]

    def model_dump(self):
        return {'model': self.model, 'status': self.status, 'usage': self.usage.model_dump(),
                'output_text': self.output_text, 'output': self.output}


class OfflineClient:
    calls = []

    def __init__(self, **kwargs):
        if kwargs['max_retries'] != 0:
            raise ValueError('SDK retries were enabled')
        self.responses = SimpleNamespace(create=self.actor)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.judge))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self.structured_judge)))
        self.max_retries = 0

    async def actor(self, **request):
        self.calls.append(('actor_or_selector', request))
        return OfflineResponse(request)

    async def judge(self, **request):
        from openai.types.chat import ChatCompletion
        self.calls.append(('judge', request))
        if 'response_format' in request:
            text = '{"thought":"diagnostic only","verdict":"SUCCESS"}'
        elif 'gpt-4o' in request['model']:
            text = 'SUCCESS'
        else:
            text = 'status: success'
        return ChatCompletion(id='offline', created=0, object='chat.completion', model=request['model'],
            choices=[{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text}}],
            usage={'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110})

    async def structured_judge(self, **request):
        from openai.types.chat import ParsedChatCompletion
        response = await self.judge(**request)
        value = response.model_dump()
        value['choices'][0]['message']['parsed'] = request['response_format'](thought='diagnostic only', verdict='SUCCESS')
        return ParsedChatCompletion[request['response_format']].model_validate(value)

    async def close(self):
        pass


def verify(runtime, output):
    record = json.loads((runtime / 'portable-runtime.json').read_text())
    if not record.get('tokenizer_source'):
        raise ValueError('Prepare with --dependencies from the --legacy installer or --tokenizer-source')
    base = output.resolve()
    if base.is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError('Choose a project or temporary output directory')
    if base.exists() or base.is_symlink():
        raise FileExistsError(base)
    base.mkdir(parents=True)
    cloned = base / 'runtime'
    legacy.prepare(record['campaign'], cloned, tokenizer_source=record['tokenizer_source'])
    rows = [json.loads(line) for line in (cloned / 'public-tasks.jsonl').read_text().splitlines()]
    for row in rows:
        row['metadata'].update(intent='Return ready', start_url='data:text/html,<h1>ready</h1>')
    (cloned / 'public-tasks.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    out = base / 'output'
    out.mkdir()
    config = {'runtime': str(cloned), 'output': str(out), 'limit': 1, 'arm': None,
              'model_usd': 10., 'judge_usd': 10., 'model_calls': 20, 'judge_calls': 2, 'browser_starts': 2,
              'tokenizer_source': record['tokenizer_source'], 'browser_port': 28300}
    path = base / 'config.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    os.environ.update(PYTHONPATH=str(cloned / 'source'), WANDB_PROJECT='openwebrl-evals', WANDB_MODE='disabled',
        WANDB_DIR=str(out), HF_HOME=str(base / 'huggingface-cache'), XDG_CACHE_HOME=str(base / 'cache'),
        OPENAI_API_KEY='offline-in-process-client', CUDA_VISIBLE_DEVICES='')
    if record.get('browser_cache'):
        os.environ['PLAYWRIGHT_BROWSERS_PATH'] = record['browser_cache']
    import openai
    original_client = openai.AsyncOpenAI
    OfflineClient.calls = []
    openai.AsyncOpenAI = OfflineClient
    original_argv = sys.argv
    try:
        sys.argv = ['legacy_runtime', '--config', str(path)]
        legacy_runtime.main()
    finally:
        sys.argv = original_argv
        openai.AsyncOpenAI = original_client
    completion = json.loads((out / 'completion.json').read_text())
    counts = {kind: sum(role == kind for role, unused in OfflineClient.calls) for kind in ['actor_or_selector', 'judge']}
    if completion['dispositions'] != 2 or completion['valid'] != 2 or counts != {'actor_or_selector': 5, 'judge': 2}:
        raise ValueError('The full diagnostic episode did not match expected policy calls/dispositions')
    report = {'diagnostic_only': True, 'campaign': record['campaign'], 'complete': True, 'paid_requests': 0,
              'sdk_calls_intercepted': counts, 'dispositions_verified': 2, 'task_success_measured': False}
    (base / 'diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.runtime.resolve(), args.output), indent=2))


if __name__ == '__main__':
    main()
