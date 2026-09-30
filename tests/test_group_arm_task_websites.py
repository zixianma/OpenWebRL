"""Domain boundaries matter to benchmark overlap, especially hosted tenants."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('group_sites', Path(__file__).resolve().parents[1] / 'scripts/group_arm_task_websites.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class WebsiteMatchingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        psl = Path(self.temp.name) / 'public_suffix_list.dat'
        psl.write_text('// ===BEGIN ICANN DOMAINS===\ncom\norg\nuk\nco.uk\njp\n*.kawasaki.jp\n!city.kawasaki.jp\n// ===END ICANN DOMAINS===\n// ===BEGIN PRIVATE DOMAINS===\nblogspot.com\n// ===END PRIVATE DOMAINS===\n')
        self.sites = M.Sites(psl)

    def test_subdomains_and_suffix_boundaries(self):
        self.assertEqual(self.sites.parse('https://WWW.Example.Co.Uk/a?x=2')[:2], ('example.co.uk', 'example.co.uk'))
        self.assertEqual(self.sites.parse('https://maps.google.com/a')[1], 'google.com')
        self.assertNotEqual(self.sites.parse('amazon.com')[1], self.sites.parse('amazon.co.uk')[1])
        self.assertEqual(self.sites.parse('shop.city.kawasaki.jp')[1], 'city.kawasaki.jp')
        self.assertEqual(self.sites.parse('shop.foo.kawasaki.jp')[1], 'shop.foo.kawasaki.jp')

    def test_hosted_tenants_are_distinct(self):
        self.assertNotEqual(self.sites.parse('alice.blogspot.com')[1], self.sites.parse('bob.blogspot.com')[1])
        self.assertEqual(self.sites.parse('www.blogspot.com')[1], 'blogspot.com')
        self.assertEqual(self.sites.parse('www.alice.blogspot.com')[1], 'alice.blogspot.com')

    def test_sentence_punctuation_and_not_email(self):
        self.assertEqual(self.sites.explicit_domains('Visit AwayTravel.com. Then https://www.example.co.uk/a?b=2, and google.com!'),
                         ['awaytravel.com', 'example.co.uk', 'google.com'])

    def test_unknown_suffix_and_substring_not_site(self):
        self.assertEqual(self.sites.explicit_domains('x.invalid and a@foo.com'), [])
        self.assertEqual(self.sites.parse('amazon.com.evil.org')[1], 'evil.org')

    def test_apostrophe_normalization(self):
        self.assertEqual(M.normalized_text('Lowe’s'), "Lowe's")


if __name__ == '__main__':
    unittest.main()
