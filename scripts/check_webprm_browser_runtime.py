#!/usr/bin/env python3
"""Real-browser navigation/initialization regressions; no model/provider calls."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live-sites', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from openwebrl.webprm_worker import prepare_browser, CHROMIUM
    from openwebrl.webprm_runtime import live_task_type
    os.environ['TMPDIR'] = str(args.output / ('long-component-' * 12))
    prepare_browser()
    assert os.environ['TMPDIR'].startswith('/tmp/wprm-')
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == '/slow.png':
                release.wait(20)
                self.send_response(204)
                self.end_headers()
                return
            self.send_response(503 if self.path == '/unavailable' else 200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<title>Navigation fixture</title><button>Ready</button><img src="/slow.png">')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report = dict(provider_calls=0, gpu_seconds=0, short_tmpdir=os.environ['TMPDIR'], checks={})
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    from browsergym.core.task import OpenEndedTask
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=CHROMIUM, headless=True)
            context = browser.new_context()
            url = f'http://127.0.0.1:{server.server_port}/'
            before = time.monotonic()
            page = context.new_page()
            try:
                OpenEndedTask(seed=1, start_url=url, goal='Observe the Ready button').setup(page)
                raise AssertionError('Legacy load setup unexpectedly ignored hanging subresource')
            except PlaywrightTimeout:
                report['checks']['legacy_load_timeout_reproduced'] = True
                report['legacy_navigation_seconds'] = time.monotonic()-before
            page.close()
            task = live_task_type()(seed=1, start_url=url, goal='Observe the Ready button', deadline=time.time()+35)
            page = context.new_page()
            before = time.monotonic()
            goal, info = task.setup(page)
            assert page.get_by_role('button', name='Ready').is_visible()
            report['checks']['domcontentloaded_interactive_page_passed'] = True
            report['repaired_navigation_seconds'] = time.monotonic()-before
            report['repaired_navigation_info'] = info
            page.close()
            task = live_task_type()(seed=1, start_url=url+'unavailable', goal='Check HTTP status', deadline=time.time()+35)
            page = context.new_page()
            try:
                task.setup(page)
                raise AssertionError('HTTP failure silently admitted')
            except RuntimeError as exc:
                assert 'HTTP status 503' in str(exc)
                report['checks']['http_error_rejected'] = True
            page.close()
            if args.live_sites:
                report['live_sites'] = []
                for site in ('https://www.bestbuy.com/', 'https://www.fedex.com/en-us/home.html'):
                    page = context.new_page()
                    before = time.monotonic()
                    row = dict(url=site)
                    try:
                        task = live_task_type()(seed=1, start_url=site, goal='Availability diagnostic', deadline=time.time()+35)
                        _, info = task.setup(page)
                        row.update(navigation_complete=True, title=page.title(), final_url=page.url,
                                   visible_body_chars=len(page.locator('body').inner_text(timeout=5000)), info=info)
                    except Exception as exc:
                        row.update(navigation_complete=False, error_type=type(exc).__name__, error=str(exc))
                    row['elapsed_seconds'] = time.monotonic()-before
                    report['live_sites'].append(row)
                    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                    page.close()
            browser.close()
        report['passed'] = all(report['checks'].values())
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
