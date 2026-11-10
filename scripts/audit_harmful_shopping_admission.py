#!/usr/bin/env python3
"""Audit reviewed shopping reference bundles; cannot start or authorize a run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.harmful_shopping_admission import check_admission
from audit_harmful_app_trace import strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--expected-bundle-sha256', required=True,
                        help='Canonical JSON digest retained outside agent/application boundary')
    parser.add_argument('--expected-review-sha256', required=True,
                        help='Independently retained reviewer-issued live validation digest')
    args = parser.parse_args()
    try:
        result = check_admission(strict_json(args.bundle),
            expected_bundle_sha256=args.expected_bundle_sha256,
            expected_review_sha256=args.expected_review_sha256)
    except (ValueError, OSError) as exc:
        result = dict(ready=False, admitted_tasks=0, errors=[str(exc)])
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
