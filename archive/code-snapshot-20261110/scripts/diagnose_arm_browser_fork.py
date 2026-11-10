#!/usr/bin/env python3
"""Zero-model-call, localhost-only check of browser-state branching semantics."""
import argparse
import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

HTML = b'''<!doctype html><title>Branch fixture</title>
<input id="query"><button id="prepare">Prepare</button>
<button id="next">Continue</button><output id="result">ready</output>
<script>
(() => {
  let hidden = 0;
  document.querySelector('#prepare').onclick = () => { hidden += 1; };
  document.querySelector('#next').onclick = () => {
    document.querySelector('#result').textContent = String(hidden);
  };
})();
</script>'''


class Handler(BaseHTTPRequestHandler):
    backend_value = 0

    def do_GET(self):
        if self.path == '/server-read':
            data = str(type(self).backend_value).encode()
        else:
            data = HTML
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != '/server-increment':
            self.send_error(404)
            return
        type(self).backend_value += 1
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


async def run(binary, url):
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=binary, headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage'])
        try:
            original = await browser.new_context(viewport={'width': 1280, 'height': 720})
            page = await original.new_page()
            await page.goto(url)
            for _ in range(7):
                await page.locator('#prepare').click()
            await page.locator('#query').fill('saved query')
            await page.evaluate('''async () => {
              document.activeElement.blur();
              document.cookie = 'fixture=saved; path=/';
              localStorage.setItem('fixture', 'saved');
              sessionStorage.setItem('fixture', 'saved');
              await new Promise((resolve, reject) => {
                const request = indexedDB.open('fixture', 1);
                request.onupgradeneeded = () => request.result.createObjectStore('kv');
                request.onerror = () => reject(request.error);
                request.onsuccess = () => {
                  const db = request.result;
                  const tx = db.transaction('kv', 'readwrite');
                  tx.objectStore('kv').put('saved', 'fixture');
                  tx.oncomplete = () => { db.close(); resolve(); };
                };
              });
            }''')
            state = await original.storage_state(indexed_db=True)
            await page.mouse.move(1200, 700)
            screenshot = await page.screenshot()
            baseline = hashlib.sha256(screenshot).hexdigest()

            async def restore(index, replay=False):
                context = await browser.new_context(storage_state=state,
                    viewport={'width': 1280, 'height': 720})
                other = await context.new_page()
                await other.goto(url)
                without_ui = await other.locator('#query').input_value()
                without_session = await other.evaluate("sessionStorage.getItem('fixture')")
                # Extra, application-specific reconstruction beyond storage_state.
                await other.evaluate("sessionStorage.setItem('fixture', 'saved')")
                if replay:
                    for _ in range(7):
                        await other.locator('#prepare').click()
                await other.locator('#query').fill('saved query')
                await other.evaluate('document.activeElement.blur()')
                await other.mouse.move(1200, 700)
                restored_state = await context.storage_state(indexed_db=True)
                identical = hashlib.sha256(await other.screenshot()).hexdigest() == baseline
                await other.locator('#next').click()
                return context, other, dict(index=index, prefix_replayed=replay,
                    cookies_localstorage_indexeddb_preserved=restored_state == state,
                    form_value_without_explicit_restore=without_ui,
                    session_storage_without_explicit_restore=without_session,
                    identical_pixels_before_action=identical,
                    same_action_result=await other.locator('#result').inner_text())

            branches = await asyncio.gather(*(restore(i) for i in range(3)), restore(3, replay=True))
            await page.locator('#next').click()
            original_result = await page.locator('#result').inner_text()
            # Context isolation protects local storage, not the remote application.
            await branches[0][1].evaluate("localStorage.setItem('fixture', 'branch0')")
            sibling_local = await branches[1][1].evaluate("localStorage.getItem('fixture')")
            await branches[0][1].evaluate("fetch('/server-increment', {method:'POST'})")
            sibling_server = await branches[1][1].evaluate("fetch('/server-read').then(r=>r.text())")
            records = [x[2] for x in branches]
            assert original_result == '7'
            assert all(r['cookies_localstorage_indexeddb_preserved'] for r in records)
            assert all(r['identical_pixels_before_action'] for r in records)
            assert all(r['same_action_result'] == '0' for r in records[:3])
            assert records[3]['same_action_result'] == '7'
            assert sibling_local == 'saved' and sibling_server == '1'
            return dict(status='passed', original_action_result=original_result, branches=records,
                sibling_local_storage_after_other_branch_write=sibling_local,
                sibling_server_value_after_other_branch_write=sibling_server,
                paid_model_calls=0, external_web_requests=0,
                conclusion='Storage restoration plus identical pixels does not imply an executable-state clone; '
                           'replaying the prefix restores this fixture, but independent contexts share server state.')
        finally:
            await browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = asyncio.run(run(args.binary, 'http://127.0.0.1:%d/' % server.server_port))
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
