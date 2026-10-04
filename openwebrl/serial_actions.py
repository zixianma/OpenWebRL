"""Pure-Python contracts for the approved serial-alternatives SFT experiment."""
import hashlib
import html
import json
import math
import re


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def instructions(variant):
    if variant not in ('all5', 'diverse3'):
        raise ValueError('Unknown serial variant')
    count = 'exactly five' if variant == 'all5' else 'one to three meaningfully different'
    return f'''\n\nSerial alternatives response protocol (overrides earlier response-format examples):
Within a single <think> block, propose {count} alternatives for the SAME current
browser state. Each alternative must have the following format, with consecutive
IDs starting at 1:
<alternative id="1">
Reasoning: your full reasoning for this alternative
Proposed action: [{{"name":"tool_name","arguments":{{...}}}}]
</alternative>
Proposed action is a JSON array of one or more ordered calls. These are hypothetical,
not executed. Escape literal &, < and > within reasoning and proposed JSON as XML
entities. Never put tool_call tags or chat control tokens inside alternatives.
After the last alternative write exactly:
Selected alternative: N
</think>
Then output the chosen alternative's identical action sequence using one actual
<tool_call>{{"name":"tool_name","arguments":{{...}}}}</tool_call> per call.
Only these final calls execute. Stop after the final call. Do not report outcomes
for hypothetical actions. Do not repeat alternatives just to fill a quota.
On subsequent turns, history retains only the chosen alternative's reasoning and
executed action, with its actual browser observation.\n'''


def augment_prompt(prompt, variant):
    if not prompt.startswith('<|im_start|>system\n'):
        raise ValueError('Expected system-first actor prompt')
    i = prompt.index('<|im_end|>')
    return prompt[:i] + instructions(variant) + prompt[i:]


def action_distance(a, b):
    """Typed novelty; coordinates are actor-normalized [0,1000].

    Exact equality is zero. Near coordinates are retained as distinct but ranked
    below changes of operation/meaningful non-coordinate arguments. No DOM
    equivalence is inferred from pixel proximity.
    """
    if canonical(a) == canonical(b):
        return 0.0
    if [x['name'] for x in a] != [x['name'] for x in b]:
        return 1.0
    values = []
    for x, y in zip(a, b):
        u, v = x['arguments'], y['arguments']
        if set(u) != set(v):
            return 1.0
        for key in u:
            if u[key] == v[key]:
                continue
            if key in ('point_2d', 'start_point_2d', 'end_point_2d'):
                values.append(min(.9, .1 + math.dist(u[key], v[key]) / 1000))
            elif x['name'] in ('scroll', 'wait') and key in ('amount', 'pixels', 'duration', 'seconds') and isinstance(u[key], (float, int)) and isinstance(v[key], (float, int)):
                values.append(.1 + .4 * min(1, abs(u[key]-v[key]) / max(1, abs(u[key]), abs(v[key]))))
            else:
                values.append(1.0)
    return max(values, default=1.0)


def retained_indices(candidates, winner, variant, state_id):
    tie = lambda i: hashlib.sha256(f'serial-v1:42:{state_id}:{i}'.encode()).hexdigest()
    if variant == 'all5':
        keep = list(range(len(candidates)))
    elif variant == 'diverse3':
        # Preserve the actual winning rationale among action-identical candidates.
        keep = [winner]
        remaining = sorted((i for i in range(len(candidates)) if i != winner), key=tie)
        unique = {canonical(candidates[winner]['actions'])}
        pool = []
        for i in remaining:
            key = canonical(candidates[i]['actions'])
            if key not in unique:
                unique.add(key); pool.append(i)
        while pool and len(keep) < 3:
            scores = {i:min(action_distance(candidates[i]['actions'], candidates[j]['actions']) for j in keep) for i in pool}
            best = sorted(pool, key=lambda i:(-scores[i], tie(i)))[0]
            if scores[best] <= 0:
                break
            keep.append(best); pool.remove(best)
    else:
        raise ValueError('Unknown variant')
    # Same underlying permutation in both arms, never winner-anchored output order.
    return sorted(keep, key=tie)


def serialize(candidates, keep, winner):
    parts = []
    for j, i in enumerate(keep, 1):
        c = candidates[i]
        parts.append(f'<alternative id="{j}">\nReasoning: {html.escape(c["reasoning"], quote=False)}\nProposed action: {html.escape(canonical(c["actions"]), quote=False)}\n</alternative>\n')
    alternatives = ''.join(parts)
    decision = f'Selected alternative: {keep.index(winner)+1}\n</think>\n'
    final = '\n'.join('<tool_call>' + canonical(c) + '</tool_call>' for c in candidates[winner]['actions']) + '<|im_end|>\n'
    return alternatives + decision + final, [len(alternatives), len(alternatives)+len(decision)]


def parse_serial(text, variant):
    """Fail closed. Returns executable text, compressed history and diagnostics."""
    value = text.strip()
    if value.startswith('<think>'):
        value = value[len('<think>'):].lstrip()
    if value.endswith('<|im_end|>'):
        value = value[:-len('<|im_end|>')].rstrip()
    if '<|' in value or '<think>' in value or value.count('</think>') != 1:
        raise ValueError('serial: invalid thinking/control boundaries')
    thought, final = value.split('</think>')
    pattern = r'<alternative id="(\d+)">\s*Reasoning: (.*?)\nProposed action: (.*?)\s*</alternative>\s*'
    matches = list(re.finditer(pattern, thought, re.S))
    count = len(matches)
    if (variant == 'all5' and count != 5) or (variant == 'diverse3' and not 1 <= count <= 3):
        raise ValueError('serial: wrong alternative count')
    if variant not in ('all5', 'diverse3'):
        raise ValueError('serial: unknown variant')
    cursor = 0; candidates = []
    for i, match in enumerate(matches, 1):
        if thought[cursor:match.start()].strip() or int(match[1]) != i:
            raise ValueError('serial: malformed alternative order')
        if '<' in match[2] or '>' in match[2] or '<' in match[3] or '>' in match[3]:
            raise ValueError('serial: unescaped hypothetical markup')
        reasoning = html.unescape(match[2]).strip()
        actions = json.loads(html.unescape(match[3]))
        if not reasoning or not isinstance(actions, list) or not actions:
            raise ValueError('serial: empty alternative')
        for action in actions:
            if not isinstance(action, dict) or set(action) != {'name', 'arguments'} or not isinstance(action['name'], str) or not isinstance(action['arguments'], dict):
                raise ValueError('serial: invalid action structure')
        candidates.append(dict(reasoning=reasoning, actions=actions)); cursor = match.end()
    choice = re.fullmatch(r'\s*Selected alternative: (\d+)\s*', thought[cursor:])
    if not choice or not 1 <= int(choice[1]) <= count:
        raise ValueError('serial: invalid selection')
    selected = int(choice[1])-1
    blocks = list(re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', final, re.S))
    if not blocks:
        raise ValueError('serial: missing final action')
    cursor = 0; calls = []
    for block in blocks:
        if final[cursor:block.start()].strip():
            raise ValueError('serial: extra final text')
        calls.append(json.loads(block[1])); cursor = block.end()
    if final[cursor:].strip() or canonical(calls) != canonical(candidates[selected]['actions']):
        raise ValueError('serial: final action does not match chosen alternative')
    executable = '\n'.join('<tool_call>' + canonical(c) + '</tool_call>' for c in calls)
    # Escape possible quoted tool tags in the winner's rationale in carried history.
    history = '<think>\n' + html.escape(candidates[selected]['reasoning'], quote=False) + '\n</think>\n' + executable
    return executable, history, dict(candidate_count=count, selected_index=selected,
        distinct_actions=len({canonical(c['actions']) for c in candidates}), candidates=candidates)
