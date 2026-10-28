"""Bind native evaluation helpers to a worker's already verified original site.

Task preparation checks declarations only. Runtime construction additionally
requires the claimed instance/reset identity and, for Shopping, a hash-bound
private account receipt. This module starts no site, browser, or provider.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat

from .webprm_webarena import EvaluationNotReady
from .webprm_webarena_helpers import (NativeHelpers, HelperNotReady, HELPER_FILE,
    HELPER_SHA256, REDDIT, GITLAB, SHOPPING, _parse)

HELPER_RUNTIME_PROTOCOL = 'claimed_original_site_native_helpers_v1'
HELPER_CONFIGURATION = dict(protocol=HELPER_RUNTIME_PROTOCOL, source_sha256=HELPER_SHA256)
HELPER_SITES = {REDDIT: 'reddit', GITLAB: 'gitlab', SHOPPING: 'shopping'}


def validate_helper_plan(task, config):
    """Check frozen expressions and opt-in identity without reading credentials."""
    if config.get('native_helpers') != HELPER_CONFIGURATION:
        raise EvaluationNotReady('Task requires explicit pinned original-site helper configuration')
    source = Path(config['arbiter_source']) / HELPER_FILE
    if hashlib.sha256(source.read_bytes()).hexdigest() != HELPER_SHA256:
        raise EvaluationNotReady('Pinned native helper source changed')
    native = task.native_config()
    sites = tuple(task.actor_view['sites'])
    if len(sites) != 1:
        raise EvaluationNotReady('Native helper runtime currently supports single-site tasks only')
    names = set()
    for target in native['eval'].get('program_html', []):
        for field in ('url', 'locator'):
            expression = target.get(field, '')
            if not expression.startswith('func:'):
                continue
            try:
                name, _ = _parse(expression, field)
            except HelperNotReady as error:
                raise EvaluationNotReady(str(error)) from None
            if HELPER_SITES[name] != sites[0]:
                raise EvaluationNotReady('Native helper does not belong to the task original site')
            names.add(name)
    return names


def _private_shopping_accounts(instance):
    from scripts.check_webprm_reddit_reset import safe_path
    receipt = instance.get('native_helper_accounts')
    if (not isinstance(receipt, dict) or set(receipt) != {'path', 'sha256'}
            or not isinstance(receipt['path'], str)
            or not isinstance(receipt['sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', receipt['sha256'])):
        raise EvaluationNotReady('Owned Shopping instance lacks its private helper-account receipt')
    original = Path(receipt['path'])
    if not original.is_absolute() or original.resolve() != original or original.is_symlink():
        raise EvaluationNotReady('Shopping helper accounts require a canonical non-symlink path')
    path = safe_path(original)
    status = path.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > 65536 or stat.S_IMODE(status.st_mode) & 0o077:
        raise EvaluationNotReady('Shopping helper accounts must be a bounded private regular file')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise EvaluationNotReady('Shopping helper-account receipt changed')
    try:
        accounts = json.loads(raw)
        if (set(accounts) != {'shopping_site_admin'} or
                set(accounts['shopping_site_admin']) != {'username', 'password'} or
                any(not isinstance(value, str) or not value for value in accounts['shopping_site_admin'].values())):
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise EvaluationNotReady('Shopping helper accounts have an invalid private schema') from None
    return accounts


def owned_native_helpers(task, config, instance):
    """Call only after the site adapter has validated and claimed its live lease."""
    names = validate_helper_plan(task, config)
    if not names:
        raise EvaluationNotReady('An owned native helper binding requires a task with helper expressions')
    site = tuple(task.actor_view['sites'])[0]
    reset = config.get('reset_id')
    if (not isinstance(instance, dict) or instance.get('scope') != 'original'
            or not isinstance(reset, str) or not re.fullmatch('[0-9a-f]{64}', reset)
            or instance.get('state_provenance_sha256') != reset
            or not isinstance(instance.get('endpoint'), str)
            or config.get('endpoints', {}).get(site, '').rstrip('/') != instance['endpoint'].rstrip('/')):
        raise EvaluationNotReady('Native helper binding differs from the claimed original site/reset')
    kwargs = {}
    if SHOPPING in names:
        kwargs.update(shopping_origin=instance['endpoint'].rstrip('/'),
            shopping_accounts=_private_shopping_accounts(instance))
    try:
        helpers = NativeHelpers(config['arbiter_source'], **kwargs)
        helpers.validate_task(task.native_config())
    except HelperNotReady as error:
        raise EvaluationNotReady(str(error)) from None
    # Provenance contains no credential payload or account-file path.
    helpers.manifest.update(runtime_protocol=HELPER_RUNTIME_PROTOCOL, owned_site=site,
        state_provenance_sha256=reset, configured_task_sha256=task.configured_config_sha256,
        task_helpers=sorted(names))
    return helpers
