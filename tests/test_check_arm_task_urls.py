from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_arm_task_urls import classify,decision,choose_urls


def page(**kwargs):
    return dict(dict(http_status=200,title='A useful website',text='Useful public content. '*30,
                     interactive_count=10,password_fields=0,screenshot_bytes=1000,dom_bytes=5000),**kwargs)


class LivePageTests(unittest.TestCase):
    def test_http200_challenge_is_not_available(self):
        self.assertEqual(classify(page(title='Just a moment...',text='Verify you are human')),
                         ('inconclusive','browser_challenge'))

    def test_login_link_does_not_reject_public_page(self):
        self.assertEqual(classify(page(text='Sign in '+page()['text']))[0],'page_available')

    def test_auth_form_is_uncertain_not_a_dead_site(self):
        self.assertEqual(classify(page(password_fields=1))[0],'inconclusive')

    def test_selling_domains_article_is_not_parked(self):
        self.assertEqual(classify(page(title='Domain for sale: market news',text='Market news '*1000))[0],
                         'page_available')

    def test_timeout_with_rendered_content_is_not_a_dead_site(self):
        self.assertEqual(classify(page(navigation_error='Page.goto timeout')),
                         ('inconclusive','navigation_error_with_content'))

    def test_browser_crash_is_not_site_failure(self):
        self.assertEqual(classify(dict(setup_error='Chromium failed to launch'))[0],'inconclusive')

    def test_blank_page_requires_more_evidence(self):
        self.assertEqual(classify(page(text='',interactive_count=0))[0],'inconclusive')

    def test_unavailable_requires_two_attempts(self):
        bad=dict(classification='unavailable');good=dict(classification='page_available')
        self.assertEqual(decision([bad]),'inconclusive')
        self.assertEqual(decision([bad,bad]),'unavailable_after_retry')
        self.assertEqual(decision([bad,good]),'page_available')

    def test_sample_includes_all_pae_urls_and_unique_insta_urls(self):
        urls={'a':dict(sources={'pae-webvoyager':10,'insta-v3':2}),
              'b':dict(sources={'insta-v3':1}),'c':dict(sources={'insta-v3':1})}
        chosen=choose_urls(urls,insta_count=1)
        self.assertEqual(len(chosen),2);self.assertIn('a',chosen)
        self.assertEqual(chosen,choose_urls(dict(reversed(list(urls.items()))),insta_count=1))


if __name__=='__main__':unittest.main()
