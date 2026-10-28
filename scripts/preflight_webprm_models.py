#!/usr/bin/env python3
"""Validate local frozen critics on the public browser fixture; zero paid calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.jev_eval import write_json
from openwebrl.webprm_source import load_shepherd, load_arbiter, load_arbiter_causal_renderer
from openwebrl.webprm_worker import completion

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    fixture = json.loads((root/'fixture/fixture.json').read_text())
    assert fixture['passed'] and fixture['native_arbiter_tab_stop_passed']
    target = root/'model-preflight'
    target.mkdir(exist_ok=True)
    journal = target/'requests.jsonl'
    config = dict(deadline=time.time()+480, budget_path=str(target/'unused-budget.json'))
    shepherd = load_shepherd(root/'sources/shepherd')
    arbiter = load_arbiter(root/'sources/arbiter')
    renderer = load_arbiter_causal_renderer(root/'WebPRMBench.jsonl')
    context = dict(intent='Enter Ada and save', trajectory='', current_url='about:blank',
                   start_url='about:blank', text_observation=fixture['observation'])
    checklist_response = completion(config, dict(model='WebShepherd8B',
        messages=shepherd.get_messages(context,'checklist_generation','web_shepherd'),
        temperature=0.,max_tokens=1024), journal, local=True, endpoint='http://127.0.0.1:8011/v1')
    raw_checklist = checklist_response['choices'][0]['message']['content']
    checklist = raw_checklist.split('<answer>')[-1].split('</answer>')[0].strip() if '<answer>' in raw_checklist else raw_checklist.split('[CHECKLISTS]')[-1].strip()
    assert checklist
    response_text, scores = [], []
    payload = dict(model='WebShepherd8B', messages=shepherd.get_messages(
        dict(context, checklist=checklist, thought='Type the requested name',action="fill('1', 'Ada')"),
        'judge_progress','web_shepherd',text_obs=True,image_obs=False),
        temperature=0.,max_tokens=2048,logprobs=True,top_logprobs=10)
    for _ in range(2):
        response = completion(config,payload,journal,local=True,endpoint='http://127.0.0.1:8011/v1')
        tokens=response['choices'][0].get('logprobs',{}).get('content')
        assert tokens and all('top_logprobs' in token for token in tokens)
        probabilities=shepherd.get_judge_probs(tokens)
        assert probabilities and all(sum(p.values())>0 for p in probabilities)
        response_text.append(response['choices'][0]['message']['content'])
        scores.append(shepherd.get_score([probabilities],'prob')[0])
    arbiter_context=dict(context,text_observation=fixture['native_arbiter_observation'])
    prompt=renderer.render(**arbiter_context,thought1='Complete the form',action1='click [1]',
                           thought2='Open an unrelated tab',action2='new_tab')
    response=completion(config,dict(model='WebArbiter7B',messages=[dict(role='user',content=prompt)],
        temperature=0.,max_tokens=2048),journal,local=True,endpoint='http://127.0.0.1:8012/v1')
    verdict=arbiter.parse_answer(response['choices'][0]['message']['content'])
    assert verdict in (1,2)
    context_probe = None
    plan = json.loads((root/'plan.json').read_text())
    probe = plan.get('local_context_regression_probe')
    if probe:
        saved = Path(probe['path']).read_bytes()
        assert hashlib.sha256(saved).hexdigest() == probe['sha256']
        request = json.loads(saved)
        assert request['model'] == 'WebShepherd8B'
        assert request['temperature'] == 0 and request['max_tokens'] == 2048
        response = completion(config, request, journal, local=True,
                              endpoint='http://127.0.0.1:8011/v1', purpose='context_regression')
        assert response['model'] == 'WebShepherd8B'
        assert response['choices'][0]['finish_reason'] == 'stop'
        probabilities = shepherd.get_judge_probs(response['choices'][0]['logprobs']['content'])
        assert probabilities and all(sum(p.values()) > 0 for p in probabilities)
        context_probe = dict(passed=True, prompt_tokens=response['usage']['prompt_tokens'],
            output_tokens=response['usage']['completion_tokens'], request_sha256=probe['sha256'],
            full_prompt_preserved=True, native_label_parser_passed=True)
        assert context_probe['prompt_tokens'] > 10000
    models=json.loads((root/'models.json').read_text())
    effective={name:json.loads((Path(model['path'])/'generation_config.json').read_text())
               for name,model in models.items()}
    write_json(target/'result.json',dict(passed=True,paid_api_calls=0,
        context_regression=context_probe,
        shepherd_scores=scores,repeat_text_identical=response_text[0]==response_text[1],
        repeat_score_identical=scores[0]==scores[1],arbiter_verdict=verdict,
        inherited_generation_config=effective,source_manifests=dict(shepherd=shepherd.manifest,
            arbiter=arbiter.manifest,arbiter_prompt=renderer.manifest)))
    print(json.dumps(dict(passed=True,paid_api_calls=0,shepherd_scores=scores,arbiter_verdict=verdict)))

if __name__ == '__main__':
    main()
