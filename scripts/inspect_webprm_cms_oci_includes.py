#!/usr/bin/env python3
"""Additive bounded read of only remaining CMS startup contracts; never execute."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import inspect_webprm_cms_oci_startup as reader

EXACT = frozenset((
    'etc/my.cnf', 'etc/nginx/nginx.conf', 'etc/nginx/fastcgi_params',
    'etc/nginx/mime.types', 'var/www/magento2/nginx.conf.sample',
    'run/mysqld/.init', 'run/postgresql/.init', 'var/lib/mysql/mysql_upgrade_info',
    'usr/share/java/elasticsearch/bin/elasticsearch',
    'usr/share/java/elasticsearch/bin/elasticsearch-env',
    'usr/share/java/elasticsearch/bin/elasticsearch-env-from-file',
    'usr/share/java/elasticsearch/bin/elasticsearch-cli',
    'usr/share/java/elasticsearch/config/jvm.options.d/heap.options',
))
PREFIXES = ('etc/my.cnf.d/', 'etc/mysql/', 'etc/nginx/conf.d/',
            'etc/supervisor.d/', 'etc/crontabs/', 'etc/cron.d/',
            'var/spool/cron/crontabs/', 'etc/periodic/')


def selected(name):
    return name in EXACT or any(name.startswith(prefix) for prefix in PREFIXES)


def main():
    reader.selected = selected
    reader.OUTPUT = reader.N / 'cms-thirdparty-startup-includes-v1'
    reader.main()


if __name__ == '__main__':
    main()
