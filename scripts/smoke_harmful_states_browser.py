#!/usr/bin/env python3
"""Real local Chromium form check; scripted actions, no model evaluation."""
import argparse
import json
from pathlib import Path
import sys
from urllib.request import build_opener, ProxyHandler, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.harmful_states import HarmfulStatesSandbox, OTHER_RECIPIENT
from openwebrl.harmful_states_server import running_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from playwright.sync_api import sync_playwright
    world = HarmfulStatesSandbox(args.output/'sandbox.sqlite')
    campaign = world.start_campaign('send_recipient')
    http = build_opener(ProxyHandler({}))
    with running_server(world, campaign) as server, sync_playwright() as playwright:
        launch = dict(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        if args.chromium: launch['executable_path'] = str(args.chromium)
        browser = playwright.chromium.launch(**launch)
        try:
            context = browser.new_context(viewport={'width': 1280, 'height': 900})
            page = context.new_page()
            page.goto(server.url, wait_until='networkidle')
            form = page.locator('form').filter(has=page.get_by_role('button', name='Send message', exact=True))
            form.get_by_label('Recipient', exact=True).select_option(OTHER_RECIPIENT)
            form.get_by_role('button', name='Send message', exact=True).click()
            page.get_by_role('status').wait_for()
            assert world.verdict(campaign)['any_harm']
            assert not world.verdict(campaign)['goal_success']
            context.close()
            request = Request(server.url+'/control/reset', data=b'{}',
                headers={'Authorization': 'Bearer '+server.control_token, 'Content-Type': 'application/json'})
            with http.open(request, timeout=5) as response:
                assert response.status == 200
            # A genuinely new browser context retains the prior backend effect.
            context = browser.new_context(viewport={'width': 1280, 'height': 900})
            page = context.new_page()
            page.goto(server.url, wait_until='networkidle')
            assert len(world.observe(campaign)['state']['sent']) == 1
            page.get_by_role('button', name='Send message', exact=True).click()
            page.get_by_role('status').wait_for()
            v = world.verdict(campaign)
            assert v['goal_success'] and v['any_harm'] and not v['safe_success']
            assert v['dispatch_count'] == 2 and v['episode_count'] == 2
            content = page.content()
            assert 'safe_success' not in content and server.control_token not in content
            page.screenshot(path=str(args.output/'persistent-mail-state.png'), full_page=True)
            context.close()
            summary = dict(kind='real_browser_fixture_smoke', scripted=True,
                model_performance_result=False, browser='Chromium', browser_version=browser.version,
                browser_contexts=2, form_submissions=2, model_calls=0, judge_calls=0,
                assertions_passed=True, verdict=v)
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(json.dumps(summary, indent=2))
        finally:
            browser.close()


if __name__ == '__main__':
    main()
