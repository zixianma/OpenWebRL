import os

import pytest

from openwebrl.jev_harness import revision


@pytest.mark.skipif(os.environ.get('JEV_LOCAL_BROWSER_TEST')!='1',reason='Opt-in local Chromium fixture')
def test_covered_click_and_select_offered_upstream_but_filtered_in_v2(tmp_path,monkeypatch):
    monkeypatch.setenv('BROWSER_HARNESS_HOME',str(tmp_path/'harness'))
    from playwright.sync_api import sync_playwright
    import jev_ultrafast.browser as browser
    with sync_playwright() as p:
        chromium=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=chromium.new_page(viewport={'width':1120,'height':780})
        page.set_content('''<button id="under">Underlying target</button>
            <select><option>A</option><option>B</option></select>
            <div id="cover" style="position:fixed;inset:0;background:#ffffffee;z-index:20">
              <button onclick="document.getElementById('cover').remove()">Dismiss</button></div>''')
        original=page.evaluate(browser.READ_STATE)
        assert any(a['label']=='Underlying target' for a in original['actions'])
        assert any(a['kind']=='select' for a in original['actions'])
        old=browser.READ_STATE,browser.MARKER
        with revision(browser,'actionable-v2'):
            filtered=page.evaluate(browser.READ_STATE)
            assert not any(a['label']=='Underlying target' or a['kind']=='select' for a in filtered['actions'])
            assert any(a['label']=='Dismiss' for a in filtered['actions'])
            assert page.evaluate(browser.MARKER)==filtered['marker']
            page.get_by_text('Dismiss',exact=True).click()
            exposed=page.evaluate(browser.READ_STATE)
            assert any(a['label']=='Underlying target' for a in exposed['actions'])
            assert any(a['kind']=='select' for a in exposed['actions'])
        assert (browser.READ_STATE,browser.MARKER)==old
        chromium.close()
