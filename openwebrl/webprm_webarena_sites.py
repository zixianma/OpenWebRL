"""Explicit original-site lease and authentication interfaces.

Reddit and the third-party CMS (shopping_admin) and Shopping images have live validators. Other domain entries describe required
interfaces; neither a plan nor a receipt can make an unimplemented adapter ready.
No site, browser, allocation or network request starts at import.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
from types import MappingProxyType
from typing import Callable
from urllib.parse import urlsplit

from .webprm_webarena import EvaluationNotReady, digest

SITE_RUNTIME_PROTOCOL = 'verified_original_site_registry_v1'
INSTANCE_SCHEMA = 'webprm_original_site_instance_v1'
SITE_RUNTIME_CONFIGURATION = dict(protocol=SITE_RUNTIME_PROTOCOL, instance_schema=INSTANCE_SCHEMA)
NATIVE_LOGIN = 'released_native_ui_login'
NO_LOGIN = 'released_task_unauthenticated'


@dataclass(frozen=True)
class SiteAdapter:
    domain: str
    authentication: str
    # A domain validator must check original artifact identities, actual native
    # service/reset controls and any shared read-only backend evidence. Merely
    # checking a JSON status or accepting the shared structural checks is not enough.
    validate_instance: Callable | None = None
    authenticate: Callable | None = None
    legacy_reddit: bool = False

    @property
    def ready(self):
        return self.validate_instance is not None and (
            self.authentication == NO_LOGIN or self.authenticate is not None or self.legacy_reddit)


def _reddit_validator(instance):
    from scripts.check_webprm_reddit_reset import validate_instance
    return validate_instance(instance)


def _cms_validator(instance):
    from .webprm_cms_site import validate_instance
    return validate_instance(instance)


def _cms_authenticate(source, instance, chromium, output):
    from .webprm_cms_site import authenticate
    return authenticate(source, instance, chromium, output)


def _shopping_validator(instance):
    from .webprm_shopping_site import validate_instance
    return validate_instance(instance)


def _shopping_authenticate(source, instance, chromium, output):
    from .webprm_shopping_site import authenticate
    return authenticate(source, instance, chromium, output)


SITE_ADAPTERS = MappingProxyType({
    'reddit': SiteAdapter('reddit', NATIVE_LOGIN, _reddit_validator, legacy_reddit=True),
    'map': SiteAdapter('map', NO_LOGIN),
    # Third-party Shopping image validated by job 358540 (see webprm_shopping_site).
    'shopping': SiteAdapter('shopping', NATIVE_LOGIN, _shopping_validator, _shopping_authenticate),
    # Third-party CMS image validated by job 357831 (see webprm_cms_site).
    'shopping_admin': SiteAdapter('shopping_admin', NATIVE_LOGIN, _cms_validator, _cms_authenticate),
    'gitlab': SiteAdapter('gitlab', NATIVE_LOGIN),
})


def validate_site_configuration(value):
    if value != SITE_RUNTIME_CONFIGURATION:
        raise EvaluationNotReady('Unknown explicit original-site runtime configuration')


def site_adapter(task, config, *, require_ready=True):
    sites = tuple(task.actor_view['sites'])
    if len(sites) != 1 or sites[0] not in SITE_ADAPTERS:
        raise EvaluationNotReady('A supported single original domain is required')
    adapter = SITE_ADAPTERS[sites[0]]
    configured = config.get('site_runtime')
    if configured is not None:
        validate_site_configuration(configured)
    elif not adapter.legacy_reddit:
        raise EvaluationNotReady('Domain requires the explicit original-site runtime protocol')
    if adapter.authentication == NO_LOGIN and task.requires_storage_state:
        raise EvaluationNotReady('Unauthenticated domain task unexpectedly requires storage state')
    if require_ready and not adapter.ready:
        raise EvaluationNotReady('Original ' + adapter.domain + ' hosting/authentication adapter is unverified')
    return adapter


def registry_status():
    return {name: dict(authentication=entry.authentication, live_adapter_ready=entry.ready,
                       original_instance_schema='legacy_original_reddit' if entry.legacy_reddit else INSTANCE_SCHEMA)
            for name, entry in SITE_ADAPTERS.items()}


def canonical_runtime_path(value):
    from scripts.check_webprm_reddit_reset import safe_path
    if not isinstance(value, str):
        raise EvaluationNotReady('Expected a canonical runtime path')
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink():
        raise EvaluationNotReady('Runtime receipt paths must be canonical without symlink redirects')
    return safe_path(path)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def verified_receipt_reference(value):
    """Verify a bounded evidence file; the domain validator interprets its contents."""
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'} or not _sha(value['sha256'])):
        raise EvaluationNotReady('Expected a hash-bound native validation receipt')
    path = canonical_runtime_path(value['path'])
    if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise EvaluationNotReady('Native validation receipt is absent or exceeds 4 MiB')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != value['sha256']:
        raise EvaluationNotReady('Native validation receipt changed')
    return json.loads(raw)


def _owned_descendant(pid, controller_pid, controller_start):
    from .webprm_lifecycle import _process
    visited = set()
    while pid not in visited and pid > 0:
        visited.add(pid)
        current = _process(pid)
        if pid == controller_pid:
            return current['start_ticks'] == controller_start and current['state'] != 'Z'
        pid = current['ppid']
    return False


def validate_original_site_receipt(instance, domain):
    """Common live ownership/schema checks; never replaces a domain validator.

    `base_identity` is the domain's composite original-image/config/dataset
    identity. `state_provenance` binds that base, the private writable state and
    the native reset receipt. Backend reuse needs additional domain-specific
    read-only evidence. No filesystem is reset or deleted by these checks.
    """
    required = {'schema', 'domain', 'scope', 'instance_id', 'endpoint', 'state_path',
        'base_identity', 'base_identity_sha256', 'state_provenance', 'state_provenance_sha256',
        'job_id', 'hostname', 'controller_pid', 'controller_start_ticks',
        'service_pid', 'service_start_ticks', 'network_namespace', 'service_validation', 'reset_validation'}
    if not isinstance(instance, dict) or not required <= instance.keys():
        raise EvaluationNotReady('Incomplete generic original-site instance receipt')
    if instance['schema'] != INSTANCE_SCHEMA or instance['domain'] != domain or instance['scope'] != 'original':
        raise EvaluationNotReady('Original-site schema/domain/scope differs')
    if not isinstance(instance['instance_id'], str) or not instance['instance_id']:
        raise EvaluationNotReady('Original-site instance needs a unique identity')
    state = canonical_runtime_path(instance['state_path'])
    if not state.is_dir():
        raise EvaluationNotReady('Original-site private writable state is absent')
    base = instance['base_identity']
    if not isinstance(base, dict) or not base or digest(base) != instance['base_identity_sha256']:
        raise EvaluationNotReady('Composite original-site base identity changed')
    expected = dict(domain=domain, base_identity_sha256=instance['base_identity_sha256'],
        instance_id=instance['instance_id'], state_path=str(state), reset_validation=instance['reset_validation'])
    provenance = instance['state_provenance']
    if (not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in expected.items())
            or not _sha(instance['state_provenance_sha256'])
            or digest(provenance) != instance['state_provenance_sha256']):
        raise EvaluationNotReady('Original-site private reset provenance differs')
    parsed = urlsplit(instance['endpoint'])
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        port = parsed.port
    except (ValueError, TypeError):
        loopback, port = False, None
    allowed_paths = ('/admin', '/admin/') if domain == 'shopping_admin' else ('', '/')
    if (parsed.scheme != 'http' or not loopback or port is None or parsed.path not in allowed_paths
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or any(c.isspace() for c in instance['endpoint'])):
        raise EvaluationNotReady('Original site requires an explicit credential-free loopback HTTP origin')
    if (str(instance['job_id']) != os.environ.get('SLURM_JOB_ID')
            or instance['hostname'] != socket.gethostname()):
        raise EvaluationNotReady('Original-site allocation/host is not owned by this worker')
    from .webprm_lifecycle import _process
    for prefix in ('controller', 'service'):
        pid, ticks = instance[prefix + '_pid'], instance[prefix + '_start_ticks']
        if type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0:
            raise EvaluationNotReady('Original-site process identity is invalid')
        identity = _process(pid)
        if identity['start_ticks'] != ticks or identity['state'] == 'Z':
            raise EvaluationNotReady('Original-site process identity is stale')
    owner = instance['controller_pid'], instance['controller_start_ticks']
    if not all(_owned_descendant(pid, *owner) for pid in (os.getpid(), instance['service_pid'])):
        raise EvaluationNotReady('Worker and service must belong to the recorded live controller')
    if not all(os.readlink(f'/proc/{pid}/ns/net') == instance['network_namespace']
               for pid in (os.getpid(), instance['service_pid'])):
        raise EvaluationNotReady('Worker and original service network namespaces differ')
    # Recheck PID identity after ancestry/namespace inspection.
    for prefix in ('controller', 'service'):
        if _process(instance[prefix + '_pid'])['start_ticks'] != instance[prefix + '_start_ticks']:
            raise EvaluationNotReady('Original-site process identity changed during validation')
    verified_receipt_reference(instance['service_validation'])
    verified_receipt_reference(instance['reset_validation'])


def validate_site_instance(task, config, instance):
    adapter = site_adapter(task, config)
    if instance.get('scope') != 'original':
        raise EvaluationNotReady('Benchmark cells require original-image provenance')
    if not adapter.legacy_reddit:
        validate_original_site_receipt(instance, adapter.domain)
    adapter.validate_instance(instance)
    if config['endpoints'][adapter.domain].rstrip('/') != instance['endpoint'].rstrip('/'):
        raise EvaluationNotReady('Lease endpoint differs from configured task')
    return adapter


def site_authentication(task, config, instance, chromium, output, *, reddit_authenticate):
    """Run after claim_lease; Map intentionally supplies no storage-state object."""
    adapter = site_adapter(task, config)
    if adapter.authentication == NO_LOGIN:
        claim = instance.get('episode_claim', {})
        if (instance.get('scope') != 'original' or instance.get('domain') != adapter.domain
                or instance.get('state_provenance_sha256') != config.get('reset_id')
                or claim.get('configured_task_sha256') != task.configured_config_sha256
                or claim.get('lease_sha256') != config.get('lease_sha256')
                or claim.get('reset_id') != config.get('reset_id')):
            raise EvaluationNotReady('Unauthenticated browser requires its verified fresh original-site claim')
        return None, dict(authenticated=False, authentication_required=False,
                          authentication_policy=NO_LOGIN, reset_id=config['reset_id'], ready=True)
    authenticate = reddit_authenticate if adapter.legacy_reddit else adapter.authenticate
    state = authenticate(config['arbiter_source'], instance, chromium, output)
    return state, dict(authenticated=True, authentication_required=True,
                       authentication_policy=NATIVE_LOGIN, reset_id=config['reset_id'], ready=True)
