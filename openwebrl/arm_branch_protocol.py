"""Pure contracts and outcome-blind reconstruction checks for branch collection."""
from __future__ import annotations

import hashlib
import io
import json
import re

from openwebrl.arm_branch_browser import digest, visible_state

SAMPLING = dict(temperature=1., top_p=.95, top_k=-1, max_new_tokens=4096, repetition_penalty=1.)
TOOLS = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


class ReconstructionMismatch(ValueError):
    pass


def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([20261006, *parts]).encode()).digest()[:4], "big") % (2**31-1)


def action_key(output):
    text = output[0]
    if output[3] != "stop":
        raise ValueError("Truncated or interrupted candidate")
    # Only executable text after the thinking block may define the action.
    text = text.rsplit('</think>', 1)[-1].replace('<|im_end|>', '').strip()
    chunks = TOOLS.findall(text)
    if not chunks:
        raise ValueError("No executable tool calls")
    calls = [json.loads(chunk) for chunk in chunks]
    if any(not isinstance(c, dict) or set(c) != {"name", "arguments"}
           or not isinstance(c["name"], str) or not isinstance(c["arguments"], dict) for c in calls):
        raise ValueError("Malformed action bundle")
    return json.dumps(calls, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def choose_distinct(outputs, count=3):
    selected, keys = [], set()
    for index, output in enumerate(outputs):
        try:
            key = action_key(output)
        except (ValueError, TypeError, KeyError, IndexError):
            continue
        if key in keys:
            continue
        keys.add(key)
        selected.append(dict(draw=index, output=output, action_key=key))
        if len(selected) == count:
            return selected
    raise ValueError(f"Fewer than {count} distinct parseable action bundles")


def fixed_candidates(outputs, count):
    """Keep the exact sampled panel, including duplicate executable actions."""
    if len(outputs) != count:
        raise ValueError('Exactly the prespecified number of draws is required')
    return [dict(draw=i, output=output, action_key=action_key(output))
            for i, output in enumerate(outputs)]


def observation_evidence(observation):
    from openwebrl.controlled_sft_worker import validate_observation
    validate_observation(observation)
    snap = observation.get("branch_snapshot")
    if not isinstance(snap, dict) or digest(snap) != observation.get("branch_snapshot_sha256"):
        raise ReconstructionMismatch("Missing or changed branch-state evidence")
    return dict(snapshot=snap, protocol=observation.get('branch_replay_protocol', 'observable-replay-v1'),
                page=observation["selection_page"],
                screenshot_sha256=hashlib.sha256(observation["screenshot"]).hexdigest(),
                browser_operations=observation["controlled"]["action_attempts"])


def compare_observations(expected, actual, expected_image, actual_image):
    """No learned judge/fuzzy text matching. Visibility is not a hidden-state clone."""
    protocol = expected.get('protocol', 'observable-replay-v1')
    if protocol != actual.get('protocol', 'observable-replay-v1'):
        raise ReconstructionMismatch('Replay protocol differs')
    if protocol not in ('observable-replay-v1', 'visible-replay-v2'):
        raise ReconstructionMismatch('Unknown replay protocol')
    project = visible_state if protocol == 'visible-replay-v2' else lambda x: x
    if project(expected['snapshot']) != project(actual['snapshot']):
        raise ReconstructionMismatch('Replayed snapshot differs')
    for key in ("page", "browser_operations"):
        if expected[key] != actual[key]:
            raise ReconstructionMismatch("Replayed " + key + " differs")
    from PIL import Image, ImageChops, ImageStat
    left = Image.open(io.BytesIO(expected_image)).convert("RGB")
    right = Image.open(io.BytesIO(actual_image)).convert("RGB")
    if left.size != right.size:
        raise ReconstructionMismatch("Screenshot size differs")
    diff = ImageChops.difference(left, right)
    mean = sum(ImageStat.Stat(diff).mean) / 3
    changed = sum(max(pixel) > 8 for pixel in diff.getdata()) / (left.width * left.height)
    if mean > .5 or changed > .001:
        raise ReconstructionMismatch("Screenshot differs beyond frozen raster tolerance")
    return dict(passed=True, pixel_mean_absolute_difference=mean,
                fraction_pixels_over_8=changed, exact_snapshot=expected['snapshot']==actual['snapshot'],
                exact_visible_snapshot=True, storage_difference_recorded=expected['snapshot']!=actual['snapshot'],
                protocol=protocol, exact_page=True,
                limitation="Observable replay equivalence only; remote server state is not cloned.")


def teacher_view(anchor, after=None):
    """Explicit allowlist: suffixes, returns and verdicts cannot enter teacher inputs."""
    result = {key: anchor[key] for key in ("task", "history", "before_observation", "candidates")}
    result["candidates"] = [{"id": i+1, "response": c["output"][0]} for i,c in enumerate(result["candidates"])]
    if after is not None:
        if len(after) != len(anchor['candidates']):
            raise ValueError("Every candidate requires immediate execution evidence")
        result["immediate_results"] = [{k: row[k] for k in ("candidate", "observation", "tool_feedback", "image")} for row in after]
    return result


def estimate_selected_success(records, choices):
    """Evaluate frozen choices by (state, repetition); no best-of-repeat oracle."""
    lookup = {(r["state"], r["candidate"], r["repeat"]): r for r in records}
    if len(lookup) != len(records):
        raise ValueError("Duplicate branch returns")
    selected = []
    for choice in choices:
        row = lookup[choice["state"], choice["candidate"], choice["repeat"]]
        if not row["valid"]:
            selected.append(None)
        else:
            selected.append(int(row["reward"] == 1))
    valid = [r for r in selected if r is not None]
    return dict(scheduled=len(selected), valid=len(valid), successes=sum(valid),
                overall=sum(valid)/len(selected) if selected else None,
                valid_only=sum(valid)/len(valid) if valid else None)
