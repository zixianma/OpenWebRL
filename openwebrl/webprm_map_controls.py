"""Prospective Map controls; no services or browsers start on import.

The pinned Nominatim 4.2 PHP driver uses PDO with a configured DSN and null
credentials. Bind its native read-only web role explicitly instead of relying
on PGUSER surviving Apache. This changes private hosting configuration only.
Live geocoding, native evaluator and reset proofs remain mandatory.
"""
from __future__ import annotations

import re


class Closed(ValueError):
    pass


WEB_DSN = 'pgsql:dbname=nominatim;user=www-data'


def private_hosts(hostname):
    """Docker supplies this node-name loopback entry; Rails logging needs it."""
    if not isinstance(hostname,str) or not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?',hostname):
        raise Closed('Bounded native hostname required')
    return '127.0.0.1 localhost '+hostname+'\n::1 localhost ip6-localhost ip6-loopback\n'


def web_database_environment(original: str) -> str:
    """Append one explicit role to the inspected original private .env."""
    if not isinstance(original,str) or '\x00' in original or len(original)>65536:
        raise Closed('Bounded original Nominatim environment required')
    if not re.search(r'^NOMINATIM_TOKENIZER=icu$',original,re.M):
        raise Closed('Pinned original Nominatim tokenizer marker missing')
    if re.search(r'^\s*(?:export\s+)?NOMINATIM_DATABASE_DSN\s*=',original,re.M):
        raise Closed('Existing Nominatim DSN requires independent review')
    return original.rstrip('\n')+'\nNOMINATIM_DATABASE_DSN="'+WEB_DSN+'"\n'


def visible_route_input(page, name):
    """Resolve the rendered route field, rejecting missing or duplicate controls."""
    if name not in ('route_from','route_to'):
        raise Closed('Unknown native routing field')
    field=page.locator('input[name="'+name+'"]:visible')
    if field.count()!=1:
        raise Closed('One visible native routing field required')
    return field
