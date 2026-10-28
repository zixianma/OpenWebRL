"""v11 (2026-10-28; copy of fuzzy-rescore-gpt41-v9/rescore.py retargeted to v11b/v11c): offline re-score of every v9 cell invalid only because of the retired fuzzy judge, all four sites
(copy of fuzzy-rescore-gpt41-v8/rescore.py; user 2026-10-25: "Re-score with gpt-4.1").

The released evaluator's fuzzy judge, gpt-4-1106-preview, is retired. Each cell saved a hash-bound
evaluator observation transcript up to the failed judge call. This replays the pinned native evaluator
on those recorded page observations; the only substitution is the judge model inside the released
llm_fuzzy_match prompt/parse (identical messages, temperature 0, max_tokens 768, top_p 1).
Results are a labelled sensitivity analysis, never the paper-exact protocol.
"""
import glob, hashlib, json, os, sys, time, urllib.request
from pathlib import Path

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
HERE = Path(__file__).resolve().parent
SRC = D/'launch-preparation-v11b-shop-a/source'
sys.path.insert(0, str(SRC))
from openwebrl.webprm_webarena import load_native_evaluator, digest, NATIVE_EVALUATOR_PROTOCOL, TASK_FILE
from openwebrl.webprm_webarena_episode import TaskEnvelope
from openwebrl.webprm_webarena_helpers import NativeHelpers
from openwebrl.webprm_evaluator_transcript import ReplayCursor, ReplayPage, _validate_events

JUDGE = 'gpt-4.1-2025-04-14'
PACKAGES = sorted(p.name for p in D.glob('launch-preparation-v11[bc]-*') if 'probe' not in p.name)


def api_key():
    env = json.loads((D/'launch-preparation-v11b-shop-a/proposal.json').read_text())['env_file']
    for line in Path(env).read_text().splitlines():
        if line.startswith('OPENAI_API_KEY='):
            return line.split('=', 1)[1].strip().strip('"').strip("'")
    raise SystemExit('OPENAI_API_KEY missing from the pinned env file')


class Judge:
    enabled = True

    def __init__(self, key, journal):
        self.key, self.journal, self.provider_calls = key, journal, 0
        self.manifest = dict(model=JUDGE, prompt='released llm_fuzzy_match (helper_functions.py:501)', substitute_for='gpt-4-1106-preview')

    def fuzzy_match(self, pred, reference, question):
        message = ("Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
                   f"question: {question}\nreference answer: {reference}\n"
                   "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
                   f"student answer: {pred}\n"
                   "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else.")
        payload = dict(model=JUDGE, messages=[dict(role='system', content='You are a helpful assistant'), dict(role='user', content=message)],
                       temperature=0, max_tokens=768, top_p=1.0)
        request = urllib.request.Request('https://api.openai.com/v1/chat/completions', data=json.dumps(payload).encode(),
                                         headers={'Authorization': 'Bearer '+self.key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
        self.provider_calls += 1
        text = body['choices'][0]['message']['content']
        with open(self.journal, 'a') as f:
            f.write(json.dumps(dict(time=time.time(), payload=payload, model=body.get('model'), text=text, usage=body.get('usage')))+'\n')
        text = text.lower()
        if 'partially correct' in text or 'incorrect' in text:
            return 0.0
        assert 'correct' in text, text
        return 1.0

    def ua_match(self, *a, **k):
        raise RuntimeError('ua_match not supported in this re-score')


def main():
    key = api_key(); out = []
    journal = HERE/'judge-calls.jsonl'
    for package in PACKAGES:
        for result_path in sorted(glob.glob(str(D/package/'campaign/attempt-*/cells/*/episodes/webarena_lite_*/*/result.json'))):
            episode = Path(result_path).parent
            result = json.loads(Path(result_path).read_text())
            if result.get('valid') or 'fuzzy' not in str(result.get('error', '')).lower():
                continue
            document = json.loads((episode/'native-evaluator-transcript.json').read_text())
            plan = json.loads((episode.parents[2]/'worker-plan.json').read_text())
            raw = [t for t in json.loads((Path(plan['arbiter_source'])/TASK_FILE).read_text()) if f"webarena_lite_{t['task_id']:03d}" == result['task_id']][0]
            task = TaskEnvelope.from_release(raw, plan['endpoints']).native_config()  # what the evaluator transcript binds
            trajectory = json.loads((episode/'native-evaluator-trajectory.json').read_text())
            binding = document['binding']
            assert binding['task_sha256'] == digest(task) and binding['trajectory_sha256'] == digest(trajectory), 'transcript binding'
            assert document.get('event_count') == _validate_events(document['events'])
            root = plan['arbiter_source']
            cursor = ReplayCursor(document['events'])
            helpers = NativeHelpers(root); helpers.manifest = binding['helper_manifest']
            judge = Judge(key, journal)
            evaluator = load_native_evaluator(root, fuzzy_judge=judge, scoring_protocol=binding.get('scoring_protocol', NATIVE_EVALUATOR_PROTOCOL),
                                              deployment_helpers=helpers, observation_replay=cursor)
            assert digest(evaluator.manifest) == binding['source_sha256'], 'pinned evaluator source'
            try:
                verdict = evaluator.evaluate(task, trajectory, ReplayPage(cursor), termination=binding['termination'])
                cursor.assert_complete()
            except Exception as error:  # evaluator needs page operations recorded only after the failed judge call
                row = dict(package=package, task_id=result['task_id'], selector=result['selector'], rescorable=False,
                           reason=f'{type(error).__name__}: {str(error)[:200]}', judge_calls=judge.provider_calls)
                out.append(row); print(json.dumps(row), flush=True); continue
            row = dict(package=package, task_id=result['task_id'], selector=result['selector'], rescorable=True, original_error=result.get('error'),
                       turns=result.get('turns'), substitute_judge=JUDGE, judge_calls=judge.provider_calls,
                       success=int(verdict.get('success', 0)), native_success=int(verdict.get('native_success', verdict.get('success', 0))),
                       verdict=verdict, transcript_sha256=hashlib.sha256((episode/'native-evaluator-transcript.json').read_bytes()).hexdigest())
            out.append(row); (HERE/'rescore-results.json').write_text(json.dumps(out, indent=1, default=str)+'\n')
            print(json.dumps({k: row[k] for k in ('package', 'task_id', 'selector', 'success', 'judge_calls')}), flush=True)
    (HERE/'rescore-results.json').write_text(json.dumps(out, indent=1, default=str)+'\n')


if __name__ == '__main__':
    main()
