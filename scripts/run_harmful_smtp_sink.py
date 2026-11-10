#!/usr/bin/env python3
"""Run the bounded local sink inside an already isolated evaluation namespace."""
import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.harmful_smtp_sink import SinkLimits, SmtpSink


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='New evaluator-owned directory; must not exist')
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--sink-id', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--max-messages', type=int, required=True)
    parser.add_argument('--max-message-bytes', type=int, required=True)
    parser.add_argument('--max-storage-bytes', type=int, required=True)
    parser.add_argument('--max-runtime-seconds', type=float, required=True)
    parser.add_argument('--max-connections', type=int, default=256)
    parser.add_argument('--max-commands-per-connection', type=int, default=1000)
    parser.add_argument('--max-recipients', type=int, default=100)
    parser.add_argument('--max-line-bytes', type=int, default=65536)
    parser.add_argument('--idle-seconds', type=float, default=15)
    parser.add_argument('--heartbeat-seconds', type=float, default=5)
    args = parser.parse_args()
    limits = SinkLimits(**{key: getattr(args, key) for key in SinkLimits.__dataclass_fields__})
    sink = SmtpSink(args.output_dir, campaign_id=args.campaign_id,
                    sink_id=args.sink_id, limits=limits, port=args.port)
    signal.signal(signal.SIGTERM, lambda *_: sink.request_stop())
    signal.signal(signal.SIGINT, lambda *_: sink.request_stop())
    print(json.dumps(dict(event='ready', host=sink.address[0], port=sink.address[1],
                          start_sha256=sink.start_sha256)), flush=True)
    final = sink.serve_forever()
    print(json.dumps(dict(event='closed', final=final)), flush=True)
    return 0 if final and final['capture_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
