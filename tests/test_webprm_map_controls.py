import unittest
from unittest.mock import Mock
from openwebrl import webprm_map_controls as m


class MapControlTests(unittest.TestCase):
    def test_private_hosts_restores_native_nodename_for_rails_logging(self):
        hosts=m.private_hosts('g007')
        self.assertEqual(hosts,'127.0.0.1 localhost g007\n::1 localhost ip6-localhost ip6-loopback\n')
        for invalid in ('','g007\nexternal.example','g007 10.0.0.1','-g007','a'*254):
            with self.assertRaises(m.Closed):m.private_hosts(invalid)

    def test_explicit_native_web_role_without_public_database_or_root_role(self):
        original='NOMINATIM_TOKENIZER=icu\nNOMINATIM_IMPORT_STYLE=extratags\n'
        result=m.web_database_environment(original)
        self.assertTrue(result.startswith(original))
        self.assertIn('NOMINATIM_DATABASE_DSN="pgsql:dbname=nominatim;user=www-data"',result)
        self.assertNotIn('host=',result)
        self.assertNotIn('user=root',result)

    def test_changed_or_preconfigured_environment_requires_review(self):
        for value in ('', 'NOMINATIM_TOKENIZER=legacy\n',
                      'NOMINATIM_TOKENIZER=icu\nNOMINATIM_DATABASE_DSN=pgsql:dbname=nominatim\n',
                      'NOMINATIM_TOKENIZER=icu\nexport NOMINATIM_DATABASE_DSN=external\n'):
            with self.subTest(value=value),self.assertRaises(m.Closed):
                m.web_database_environment(value)

    def test_native_routing_rejects_hidden_duplicate_and_missing_controls(self):
        page=Mock();page.locator.return_value.count.return_value=1
        self.assertIs(m.visible_route_input(page,'route_from'),page.locator.return_value)
        page.locator.assert_called_with('input[name="route_from"]:visible')
        for count in (0,2):
            page.locator.return_value.count.return_value=count
            with self.assertRaises(m.Closed):m.visible_route_input(page,'route_to')
        with self.assertRaises(m.Closed):m.visible_route_input(page,'arbitrary')


if __name__=='__main__':unittest.main()
