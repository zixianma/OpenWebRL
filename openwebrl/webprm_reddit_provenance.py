"""Fail-closed provenance checks for the privately staged original Reddit image.

Receipts establish an observed build/copy chain, not a cryptographic filesystem
snapshot after services begin mutating their private state.
"""
import hashlib
import json
from pathlib import Path

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
SCRATCH = Path('/tmp/webprm-wa-lsuqcbna')
ORIGINAL_MANIFEST = 'e5b9d29b26ca911048fb14a8d4806a871b796db85c2991d514a13df2af0e427f'
ORIGINAL_CONFIG = 'bd8ae768c200fc4d005672af33fe03d49c9a058ae33997f3d50576e84cf51a1d'


def checked_path(value, root, *, directory=False):
    original = Path(value)
    resolved = original.resolve()
    if not original.is_absolute() or root.resolve() not in resolved.parents:
        raise ValueError('Provenance path outside approved root')
    if original.is_symlink() or str(original) != str(resolved):
        raise ValueError('Provenance path must be canonical and not a symlink')
    if directory and not resolved.is_dir():
        raise ValueError('Provenance directory missing')
    return resolved


def read_receipt(path, expected_sha256=None):
    path = checked_path(path, RUNTIME)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Provenance metadata exceeds size cap')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Provenance receipt SHA256 mismatch')
    return json.loads(raw), digest


def root_identity(path):
    value = Path(path).stat()
    return {'device': value.st_dev, 'inode': value.st_ino}


def original_layers():
    manifest, _ = read_receipt(RUNTIME/'original-registry-mirrors/webarenaimages-manifest.json', ORIGINAL_MANIFEST)
    config, _ = read_receipt(RUNTIME/'original-registry-mirrors/webarenaimages-config.json', ORIGINAL_CONFIG)
    if manifest['config']['digest'] != 'sha256:' + ORIGINAL_CONFIG:
        raise ValueError('Original config changed')
    layers = manifest['layers']
    diffs = config['rootfs']['diff_ids']
    if len(layers) != 29 or len(diffs) != 29:
        raise ValueError('Original layer count changed')
    return [dict(index=i, compressed_digest=x['digest'], compressed_bytes=x['size'], raw_diff_id=diffs[i])
            for i, x in enumerate(layers)]


def validate_original_verification(path, sha256=None):
    receipt, digest = read_receipt(path, sha256)
    if (receipt.get('status') != 'verified' or receipt.get('all_original_layers_verified') is not True
            or receipt.get('startup_content_final') is not True
            or receipt.get('missing_or_incomplete_layers') != []
            or receipt.get('source_manifest_digest') != 'sha256:' + ORIGINAL_MANIFEST
            or receipt.get('config_sha256') != ORIGINAL_CONFIG):
        raise ValueError('Original raw-layer verification is incomplete or mismatched')
    expected = original_layers()
    actual = receipt.get('layers', [])
    if len(actual) != 29:
        raise ValueError('Original verification requires all 29 layers')
    for row, wanted in zip(actual, expected):
        if row.get('verified') is not True or any(row.get(k) != v for k, v in wanted.items()):
            raise ValueError('Original verified layer identity mismatch')
    return receipt, digest


def validate_original_download(path, sha256=None):
    receipt, digest = read_receipt(path, sha256)
    if (receipt.get('status') != 'complete'
            or receipt.get('source_manifest_digest') != 'sha256:' + ORIGINAL_MANIFEST
            or receipt.get('config_digest') != 'sha256:' + ORIGINAL_CONFIG):
        raise ValueError('Original compressed download incomplete or mismatched')
    layout = checked_path(receipt['layout_dir'], RUNTIME, directory=True)
    if receipt.get('oci_reference') != 'oci:' + str(layout):
        raise ValueError('OCI layout reference mismatch')
    expected = original_layers()
    actual = receipt.get('layers', [])
    if len(actual) != 29:
        raise ValueError('Original compressed download requires all 29 layers')
    for row, wanted in zip(actual, expected):
        if (row.get('index') != wanted['index'] or row.get('digest') != wanted['compressed_digest']
                or row.get('size') != wanted['compressed_bytes']
                or row.get('expected_diff_id') != wanted['raw_diff_id']
                or row.get('compressed_sha256_verified') is not True):
            raise ValueError('Compressed original layer identity mismatch')
        blob = checked_path(layout/'blobs/sha256'/row['digest'][7:], RUNTIME)
        if blob.stat().st_size != row['size']:
            raise ValueError('Compressed original layer size changed')
    index, _ = read_receipt(layout/'index.json')
    if len(index['manifests']) != 1 or index['manifests'][0]['digest'] != receipt['oci_manifest_digest']:
        raise ValueError('OCI index mismatch')
    oci_digest = receipt['oci_manifest_digest'].removeprefix('sha256:')
    converted, _ = read_receipt(layout/'blobs/sha256'/oci_digest, oci_digest)
    if converted['config']['digest'] != 'sha256:' + ORIGINAL_CONFIG:
        raise ValueError('OCI config mismatch')
    read_receipt(layout/'blobs/sha256'/ORIGINAL_CONFIG, ORIGINAL_CONFIG)
    if [(r['digest'], r['size']) for r in converted['layers']] != [(r['compressed_digest'], r['compressed_bytes']) for r in expected]:
        raise ValueError('OCI converted layer identities mismatch')
    return receipt, digest


def validate_original_build(path, sha256=None):
    receipt, digest = read_receipt(path, sha256)
    if (receipt.get('schema_version') != 1 or receipt.get('kind') != 'original_reddit_sandbox_build'
            or receipt.get('status') != 'completed' or receipt.get('returncode') != 0
            or receipt.get('source_manifest_sha256') != ORIGINAL_MANIFEST
            or receipt.get('config_sha256') != ORIGINAL_CONFIG):
        raise ValueError('Original build receipt invalid')
    sandbox = checked_path(receipt['sandbox_path'], SCRATCH, directory=True)
    if root_identity(sandbox) != receipt.get('sandbox_root_identity'):
        raise ValueError('Original build sandbox was replaced')
    download, _ = validate_original_download(receipt['download_receipt_path'], receipt['download_receipt_sha256'])
    expected = ['apptainer', 'build', '--disable-cache', '--sandbox', str(sandbox), download['oci_reference']]
    if receipt.get('argv') != expected:
        raise ValueError('Original build command does not match pinned OCI and sandbox')
    return receipt, digest


def validate_original_provenance(instance):
    if instance.get('scope') != 'original' or instance.get('image_sha256') != ORIGINAL_MANIFEST:
        raise ValueError('Explicit original image scope and identity required')
    state_path = checked_path(instance['state_path'], SCRATCH, directory=True)
    state, state_digest = read_receipt(instance['state_provenance_path'], instance['state_provenance_sha256'])
    if (state.get('schema_version') != 1 or state.get('kind') != 'original_reddit_state_copy'
            or state.get('status') != 'completed' or state.get('returncode') != 0
            or state.get('state_path') != str(state_path)
            or state.get('state_root_identity') != root_identity(state_path)):
        raise ValueError('Fresh original state copy receipt invalid')
    for name in ('build_receipt_path', 'build_receipt_sha256'):
        if state.get(name) != instance.get(name):
            raise ValueError('Instance and state build provenance differ')
    build, build_digest = validate_original_build(state['build_receipt_path'], state['build_receipt_sha256'])
    source = build['sandbox_path']
    if (state_path == Path(source) or Path(source) in state_path.parents
            or state_path in Path(source).parents or state.get('source_sandbox_path') != source):
        raise ValueError('Original services require a fresh copy of the preserved build')
    if state.get('argv') != ['cp', '-a', '--reflink=auto', '--', source, str(state_path)]:
        raise ValueError('Original copy command source/destination mismatch')
    _, verification_digest = validate_original_verification(state['original_verification_path'], state['original_verification_sha256'])
    return {'image_sha256': ORIGINAL_MANIFEST, 'config_sha256': ORIGINAL_CONFIG,
            'verified_original_layers': 29, 'build_receipt_sha256': build_digest,
            'state_provenance_sha256': state_digest, 'original_verification_sha256': verification_digest,
            'state_path': str(state_path)}
