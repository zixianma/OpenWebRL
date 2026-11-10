#!/usr/bin/env python3
"""Serve a local synthetic task. No model, judge, network effect or GPU use."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.harmful_states import HarmfulStatesSandbox, list_cases
from openwebrl.harmful_states_server import SandboxHTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-cases', action='store_true')
    parser.add_argument('--case', default='send_recipient', choices=[x['case_id'] for x in list_cases()])
    parser.add_argument('--db', type=Path, default=Path('/tmp/openwebrl-harmful-states.sqlite'))
    parser.add_argument('--campaign', help='Resume an existing campaign without resetting its effects')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.list_cases:
        print(json.dumps(list_cases(), indent=2))
        return
    sandbox = HarmfulStatesSandbox(args.db)
    campaign = args.campaign or sandbox.start_campaign(args.case)
    server = SandboxHTTPServer(('127.0.0.1', args.port), sandbox, campaign)
    print(json.dumps(dict(url=server.url, campaign_id=campaign,
                          episode_id=server.episode_id, evaluator_token=server.control_token,
                          notice='Keep the evaluator token outside policy observations.')), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
