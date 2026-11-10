#!/usr/bin/env python3
"""Count action-schema branch opportunities in fixed-reasoning selector panels.

This is a structural audit only. It does not validate inherited teacher labels
under the substituted reasoning prefix or measure downstream action quality.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def first_action(suffix):
    match = TOOL_CALL.search(suffix)
    if match is None:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def audit(path):
    counts = defaultdict(Counter)
    source_hash = hashlib.sha256()
    with path.open("rb") as source:
        for raw in source:
            source_hash.update(raw)
            row = json.loads(raw)
            count = counts[row["split"]]
            count["panels"] += 1
            actions = [first_action(s) for s in row["action_suffixes"]]
            count["all_first_actions_parseable"] += all(a is not None for a in actions)
            names = [a.get("name") if a is not None else None for a in actions]
            count["panels_with_distinct_action_names"] += len(set(names)) > 1
            count["panels_with_one_action_name"] += len(set(names)) == 1
            for i in row["positive_indices"]:
                for j in row["negative_indices"]:
                    count["preferred_loser_pairs"] += 1
                    a, b = actions[i], actions[j]
                    if a is None or b is None:
                        count["unparseable_pairs"] += 1
                    elif a.get("name") != b.get("name"):
                        count["different_action_name_pairs"] += 1
                    elif a.get("arguments") != b.get("arguments"):
                        count["different_argument_pairs"] += 1
                        if (isinstance(a.get("arguments"), dict)
                                and isinstance(b.get("arguments"), dict)
                                and "point_2d" in a["arguments"]
                                and "point_2d" in b["arguments"]):
                            count["coordinate_argument_pairs"] += 1
                    else:
                        count["same_first_action_pairs"] += 1
    return {
        "source_sha256": source_hash.hexdigest(),
        "label_scope": "original_full_response_teacher_choice_after_reasoning_substitution",
        "unit": "panel or preferred-versus-loser pair as named",
        "first_action_only": True,
        "splits": {name: dict(count) for name, count in sorted(counts.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.panels)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
