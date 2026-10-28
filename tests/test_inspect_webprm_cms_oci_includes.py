import unittest

from scripts.inspect_webprm_cms_oci_includes import selected


class IncludesSelectionTest(unittest.TestCase):
    def test_exact_startup_dependencies_included(self):
        for path in ('etc/my.cnf.d/mariadb-server.cnf', 'etc/crontabs/root',
                     'var/spool/cron/crontabs/www-data', 'run/mysqld/.init',
                     'etc/supervisor.d/pgsql', 'var/www/magento2/nginx.conf.sample',
                     'usr/share/java/elasticsearch/bin/elasticsearch-env'):
            self.assertTrue(selected(path), path)

    def test_database_application_and_unrelated_contents_excluded(self):
        for path in ('var/lib/mysql/ibdata1', 'var/lib/mysql/magentodb/customer_entity.ibd',
                     'var/www/magento2/app/etc/env.php', 'etc/shadow',
                     'root/.ash_history', 'usr/share/java/elasticsearch/lib/large.jar'):
            self.assertFalse(selected(path), path)
