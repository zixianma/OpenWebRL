"""Offline multi-heading DOM regressions; no browser or application is started."""
from html.parser import HTMLParser
import re
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from openwebrl import harmful_shopping_references as refs


class HeadingDOM(HTMLParser):
    """Small independent DOM model for tag/class/visibility and strict reads."""
    def __init__(self, html):
        super().__init__()
        self.hidden = [False]
        self.headings = []
        self.current = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        hidden = self.hidden[-1] or 'hidden' in attrs or bool(
            re.search(r'display\s*:\s*none', attrs.get('style', '')))
        self.hidden.append(hidden)
        if tag == 'h1':
            self.current = {'classes': attrs.get('class', '').split(), 'visible': not hidden, 'text': ''}
            self.headings.append(self.current)

    def handle_endtag(self, tag):
        if tag == 'h1':
            self.current = None
        self.hidden.pop()

    def handle_data(self, data):
        if self.current is not None:
            self.current['text'] += data

    def locator(self, selector):
        match = re.fullmatch(r'h1(?:\.([\w-]+))?(:visible)?', selector)
        if match is None:
            raise AssertionError('Unexpected heading selector: ' + selector)
        class_name, visible = match.groups()
        values = [h for h in self.headings if
                  (class_name is None or class_name in h['classes']) and (not visible or h['visible'])]

        def inner_text():
            if len(values) != 1:
                raise ValueError('strict heading locator resolved to ' + str(len(values)) + ' elements')
            return values[0]['text']
        return SimpleNamespace(count=lambda: len(values), inner_text=inner_text)


def navigation_page(html, order_id=299, increment_id='000000299'):
    dom = HeadingDOM(html)
    page = MagicMock()
    page.url = f'http://127.0.0.1/admin/sales/order/view/order_id/{order_id}/'
    menu, search, link, history = [MagicMock() for _ in range(4)]
    menu.count.return_value = search.count.return_value = link.count.return_value = 1
    menu.evaluate.return_value = 'http://127.0.0.1/admin/sales/order/index/'
    link.evaluate.return_value = page.url
    link.locator.return_value.inner_text.return_value = increment_id
    fixed = {'[data-ui-id="menu-magento-sales-sales-order"] > a': menu,
             'input[placeholder="Search by keyword"]:visible': search,
             f'a[href*="/sales/order/view/"][href*="/order_id/{order_id}/"]': link,
             '#history_comment': history}
    page.locator.side_effect = lambda selector: fixed[selector] if selector in fixed else dom.locator(selector)
    return page


class NavigationTitleTests(unittest.TestCase):
    def navigate(self, html):
        page = navigation_page(html)
        charged = []
        refs.native_order_navigation(page, {'order_id': 299, 'increment_id': '000000299'}, charged.append)
        return page, charged

    def test_order_title_coexists_with_hidden_edit_order_modal_heading(self):
        html = ('<h1 class="page-title">#000000299</h1>'
                '<div style="display: none"><h1 id="modal-title-8">Edit Order</h1></div>')
        with self.assertRaisesRegex(ValueError, 'strict heading locator resolved to 2'):
            HeadingDOM(html).locator('h1').inner_text()
        page, charged = self.navigate(html)
        self.assertEqual(charged, ['goto_orders', 'fill_order_search', 'submit_order_search', 'open_native_order'])
        page.get_by_role.assert_not_called()  # Navigation does not submit a cancellation.

    def test_unrelated_visible_heading_does_not_replace_order_title(self):
        self.navigate('<h1>Other visible heading</h1><h1 class="page-title">#000000299</h1>')

    def test_hidden_page_title_is_not_a_navigation_identity(self):
        with self.assertRaisesRegex(ValueError, 'unique visible native order heading'):
            self.navigate('<div hidden><h1 class="page-title">#000000299</h1></div>')

    def test_two_visible_page_titles_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'unique visible native order heading'):
            self.navigate('<h1 class="page-title">#000000299</h1><h1 class="page-title">#000000299</h1>')

    def test_missing_page_title_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'unique visible native order heading'):
            self.navigate('<h1>Edit Order #000000299</h1>')

    def test_correct_identity_only_in_modal_does_not_mask_wrong_order_title(self):
        with self.assertRaisesRegex(ValueError, 'Native order heading differs'):
            self.navigate('<h1 class="page-title">#000000298</h1>'
                          '<div hidden><h1>Edit Order #000000299</h1></div>')


if __name__ == '__main__':
    unittest.main()
