"""Run the unchanged WEBPRMBench evaluator with an explicit inference backend.

The released evaluator depends on two vLLM interfaces. This small compatibility
layer preserves its raw prompts, sampling arguments, parsing and metrics while
allowing the already validated SGLang runtime to supply generated text.
"""
from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path


def run_official(evaluator, *, model, data, output, generate, batch_size=8):
    """Call generate(texts, sampling_parameters); never pass expected labels."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ['pairs.jsonl', 'metrics.json']):
        raise FileExistsError('Refusing to overwrite an evaluation attempt')

    class SamplingParams:
        def __init__(self, **kwargs):
            self.values = kwargs

    class LLM:
        def __init__(self, **kwargs):
            if kwargs['model'] != str(model) or kwargs['tensor_parallel_size'] != 1:
                raise ValueError('Unexpected model/topology in official evaluator')

        def generate(self, texts, params):
            if not all(isinstance(text, str) for text in texts):
                raise TypeError('Only rendered text prompts may enter inference')
            results = generate(list(texts), dict(params.values))
            if len(results) != len(texts) or not all(isinstance(text, str) for text in results):
                raise ValueError('Missing or malformed generation outputs')
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]) for text in results]

    module = types.ModuleType('vllm')
    module.LLM, module.SamplingParams = LLM, SamplingParams
    old_module, old_argv = sys.modules.get('vllm'), sys.argv
    sys.modules['vllm'] = module
    sys.argv = [str(evaluator), '--model', str(model), '--data', str(data),
                '--tensor_parallel_size', '1', '--batch_size', str(batch_size),
                '--max_model_len', '32768', '--temperature', '0', '--top_p', '1',
                '--max_tokens', '1024', '--save_jsonl', str(output / 'pairs.jsonl'),
                '--save_metrics', str(output / 'metrics.json')]
    try:
        runpy.run_path(str(evaluator), run_name='__main__')
    finally:
        sys.argv = old_argv
        if old_module is None:
            sys.modules.pop('vllm', None)
        else:
            sys.modules['vllm'] = old_module
    return json.loads((output / 'metrics.json').read_text())


def metric_fixture(evaluator, folder):
    """Exercise actual released aggregation on colliding source-local state IDs."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, state in [('source_a', 0), ('source_b', 0), ('source_b', 1)]:
        for pair in range(4):
            rows.append(dict(source_name=source, state_idx=state, pair_idx=pair, expected=1,
                             prompt=f'offline-fixture:{source}:{state}:{pair}'))
    data = folder / 'synthetic.jsonl'
    with data.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    calls = []

    def synthetic(texts, params):
        assert params == dict(temperature=0., top_p=1., max_tokens=1024, stop=['</Answer>'])
        assert all(text.startswith('offline-fixture:') for text in texts)
        calls.extend(texts)
        return ['<Answer>Response 2</Answer>' if text == 'offline-fixture:source_b:1:3'
                else '<Answer>Response 1</Answer>' for text in texts]

    metrics = run_official(evaluator, model='offline-synthetic-model', data=data,
                           output=folder / 'results', generate=synthetic, batch_size=5)
    overall = metrics['overall']
    assert overall['pairs_total'] == 12 and overall['pairs_ok'] == 11
    assert overall['states_total'] == 3 and overall['states_ok'] == 2
    assert overall['state_acc'] == .75 and len(calls) == 12
    return {'verified': True, 'synthetic_only': True, 'provider_calls': 0,
            'model_generations': 0, 'synthetic_responses': len(calls),
            'metrics': metrics, 'checks': ['source-local state identity', 'all-four-correct state metric',
                                          'macro-over-sources state accuracy', 'exact raw-prompt transport',
                                          'official sampling arguments', 'batch tail retained']}
