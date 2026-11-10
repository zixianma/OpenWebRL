"""Lossless, campaign-local processor tensor owners and descriptor-only payloads.

This is an integrity boundary for trusted private artifacts, not a safe loader
for external pickle. References never carry a trusted root through pickle. A
caller first verifies its enclosing artifact, then explicitly binds against the
opened campaign. Mapped tensors use private/COW storage; consumers must not
mutate them. BackendRow still hashes the actual values at every native forward.
No file is deleted, overwritten, or adopted after an interrupted exclusive save.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import stat

from .action_credit import digest, _sha
from .action_credit_megatron import tensor_payload_sha256
from .action_credit_state import _safe_path


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _hash(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _schema(tensors):
    return tuple((key, str(value.dtype), tuple(value.shape), tuple(value.stride()),
                  value.storage_offset(), value.untyped_storage().nbytes())
                 for key, value in sorted(tensors.items()))


def _file_identity(path):
    info = path.stat(follow_symlinks=False)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError('tensor owner must be a private regular single-link owned file')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _load_owner(path, ref):
    import mmap
    import torch
    with torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
        tensors = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    if type(tensors) is not dict or _schema(tensors) != ref.tensor_schema:
        raise ValueError('tensor owner schema mismatch')
    return tensors


@dataclass(frozen=True)
class TensorBlobRef:
    schema_version: int
    manifest_sha256: str
    root_sha256: str
    source_sha256: str
    attempt_id: str
    draw_id: str
    turn: int
    relative_path: str
    file_sha256: str
    file_bytes: int
    tensor_payload_sha256: str
    tensor_schema: tuple

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return digest(self.to_dict())


class TensorBlobMapping(Mapping):
    """Mapping protocol for native training, with a descriptor-only pickle."""
    __slots__ = ('_ref', '_path', '_identity', '_cache')

    def __init__(self, ref):
        if type(ref) is not TensorBlobRef:
            raise ValueError('typed tensor owner reference required')
        self._ref, self._path, self._identity, self._cache = ref, None, None, None

    @property
    def ref(self):
        return self._ref

    def __reduce_ex__(self, protocol):
        # In particular, neither a warm tensor cache nor root authority survives.
        return (type(self), (self.ref,))

    def __len__(self):
        return len(self.ref.tensor_schema)

    def __iter__(self):
        return iter(item[0] for item in self.ref.tensor_schema)

    def __getitem__(self, key):
        if self._path is None:
            raise ValueError('unbound tensor reference requires a verified campaign inventory')
        if _file_identity(_safe_path(self._path)) != self._identity:
            raise ValueError('tensor owner changed after inventory verification')
        if self._cache is None:
            # MAP_PRIVATE protects the retained file from accidental tensor writes.
            self._cache = _load_owner(self._path, self.ref)
        return self._cache[key]


def _owner_path(context, ref):
    root, manifest_sha, source_sha, attempts = context
    for name in ('manifest_sha256', 'root_sha256', 'source_sha256', 'file_sha256', 'tensor_payload_sha256'):
        _sha(getattr(ref, name), name)
    relative = Path(ref.relative_path)
    expected = Path('online_artifacts') / digest(ref.attempt_id) / f'{ref.turn:04d}-processor.pt'
    if (ref.schema_version != 1 or ref.manifest_sha256 != manifest_sha
            or ref.root_sha256 != digest(str(root)) or ref.source_sha256 != source_sha
            or type(ref.turn) is not int or ref.turn < 0 or relative.is_absolute()
            or relative != expected or '..' in relative.parts
            or type(ref.file_bytes) is not int or ref.file_bytes <= 0):
        raise ValueError('tensor owner root/manifest/source/path identity mismatch')
    attempt = attempts.get(ref.attempt_id)
    if not attempt or attempt['draw_id'] != ref.draw_id:
        raise ValueError('tensor owner differs from its retained charged attempt')
    path = _safe_path(root / relative)
    if root not in path.parents:
        raise ValueError('tensor owner escaped campaign root')
    return path


def create_tensor_blob(store, attempt_id, turn, tensors):
    """Write one already-isolated CPU payload; an existing file is ambiguous."""
    import torch
    semantic = tensor_payload_sha256(tensors)
    if type(tensors) is not dict or any(value.device.type != 'cpu' for value in tensors.values()):
        raise ValueError('exact isolated CPU processor dict required')
    attempt = store.snapshot()['attempts'].get(attempt_id)
    if not attempt or attempt['status'] != 'active' or type(turn) is not int or turn < 0:
        raise ValueError('tensor owner requires the active retained attempt')
    root = _safe_path(store.root)
    relative = Path('online_artifacts') / digest(attempt_id) / f'{turn:04d}-processor.pt'
    path = _safe_path(root / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError as exc:
        raise ValueError('existing tensor owner requires explicit crash reconciliation') from exc
    with os.fdopen(fd, 'wb') as handle:
        torch.save(tensors, handle)
        handle.flush(); os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    ref = TensorBlobRef(1, store.manifest.sha256, digest(str(root)), source_sha256(),
        attempt_id, attempt['draw_id'], turn, str(relative), _hash(path), path.stat().st_size,
        semantic, _schema(tensors))
    mapping = TensorBlobMapping(ref)
    bind_tensor_inventory(store, (mapping,), [ref.to_dict()])
    return mapping


def _references(mappings):
    result = {}
    for mapping in mappings:
        if type(mapping) is TensorBlobMapping:
            ref = mapping.ref
            key = ref.relative_path
            if key in result and result[key] != ref:
                raise ValueError('conflicting tensor owner references')
            result[key] = ref
        elif type(mapping) is not dict or mapping:
            # Empty mappings carry no tensor bytes (text-only rows/CPU fixtures).
            raise ValueError('nonempty processor payload must use a lossless tensor reference')
    return tuple(result[key] for key in sorted(result))


def tensor_inventory(store, mappings, *, leader=lambda fn: fn()):
    mappings = tuple(mappings)
    inventory = [ref.to_dict() for ref in _references(mappings)]
    bind_tensor_inventory(store, mappings, inventory, leader=leader)
    return inventory


def bind_tensor_inventory(store, mappings, inventory, *, leader=lambda fn: fn()):
    """Verify all owners, including unselected rows, once on collective leader.

    The enclosing immutable artifact must already have passed its own digest.
    The returned stat identities bind every rank to the files actually checked;
    ordinary BackendRow validation still hashes current tensor bytes on forward.
    """
    mappings = tuple(mappings)
    refs = _references(mappings)
    if [ref.to_dict() for ref in refs] != inventory:
        raise ValueError('transitive tensor owner inventory differs from payload references')
    context = (_safe_path(store.root), store.manifest.sha256, source_sha256(), store.snapshot()['attempts'])
    paths = {ref.relative_path: _owner_path(context, ref) for ref in refs}
    def verify():
        result = {}
        for ref in refs:
            path = paths[ref.relative_path]
            identity = _file_identity(path)
            if identity[2] != ref.file_bytes or _hash(path) != ref.file_sha256:
                raise ValueError('tensor owner file size/hash mismatch')
            # Load actual keys even for an empty claimed schema: a descriptor
            # may never hide additional stored tensors from the inventory audit.
            tensors = _load_owner(path, ref)
            if tensor_payload_sha256(tensors) != ref.tensor_payload_sha256:
                raise ValueError('tensor owner semantic hash mismatch')
            del tensors
            if _file_identity(path) != identity:
                raise ValueError('tensor owner changed during verification')
            result[ref.relative_path] = identity
        return result
    identities = leader(verify)
    for mapping in mappings:
        if type(mapping) is not TensorBlobMapping:
            continue
        path = paths[mapping.ref.relative_path]
        identity = tuple(identities[mapping.ref.relative_path])
        if _file_identity(path) != identity:
            raise ValueError('tensor owner changed after collective verification')
        # Preserve a warm cache so accidental mutations cannot be erased by
        # rebinding and hidden from the mandatory actual-value row hash.
        if mapping._path is not None and (mapping._path != path or mapping._identity != identity):
            raise ValueError('bound tensor owner identity changed')
        mapping._path, mapping._identity = path, identity
        if mapping._cache is not None and tensor_payload_sha256(mapping) != mapping.ref.tensor_payload_sha256:
            raise ValueError('materialized processor tensor payload changed')


def episode_tensor_mappings(episode):
    return tuple(turn.tensors for turn in episode.turns) + tuple(
        getattr(native, 'multimodal_train_inputs', None) or {} for native in episode.native_samples)
