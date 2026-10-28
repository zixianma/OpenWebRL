#!/usr/bin/env python3
"""Prepare an additive, offline Piotr processor view; never load model weights.

The released checkpoint stays untouched. The image pixel ceiling and a tokenizer
metadata field representation change; model weights remain exact symlinks to the
pinned local checkpoint. Actual GPU/vision/JSON serving stays unverified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.webprm_webarena_piotr import CHECKPOINT, METADATA, MODEL, REVISION, WEIGHTS

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
MAX_PIXELS = 262144
CHANGES = [dict(file='processor_config.json', path='image_processor.size.longest_edge',
                before=16777216, after=MAX_PIXELS),
           dict(file='tokenizer_config.json', path='extra_special_tokens',
                operation='rename ordered list to additional_special_tokens', token_values_and_order_unchanged=True)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path == RUNTIME or RUNTIME not in path.parents:
        raise ValueError('Processor view must be a strict descendant of the approved runtime topic')
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            raise ValueError('No symlink component is allowed in a writable processor-view path')
    if path.resolve() != path:
        raise ValueError('Canonical processor-view path required')
    return path


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path, data):
    path = safe_path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def processor_metadata(original):
    value = deepcopy(original)
    if value.get('processor_class') != 'Qwen3VLProcessor':
        raise ValueError('Published processor class changed')
    image = value['image_processor']
    if image.get('image_processor_type') != 'Qwen2VLImageProcessor' or image['size'] != {
            'longest_edge': 16777216, 'shortest_edge': 65536}:
        raise ValueError('Published image processor settings changed')
    image['size']['longest_edge'] = MAX_PIXELS
    return value


def tokenizer_metadata(original):
    value = deepcopy(original)
    tokens = value.pop('extra_special_tokens')
    if not isinstance(tokens, list) or not tokens or any(type(t) is not str for t in tokens) or 'additional_special_tokens' in value:
        raise ValueError('Published tokenizer metadata representation changed')
    value['additional_special_tokens'] = tokens
    return value


def view_bytes(name):
    raw = (CHECKPOINT / name).read_bytes()
    if name == 'processor_config.json':
        return encoded(processor_metadata(json.loads(raw)))
    if name == 'tokenizer_config.json':
        return encoded(tokenizer_metadata(json.loads(raw)))
    return raw


def source_identity():
    if CHECKPOINT.resolve() != CHECKPOINT:
        raise ValueError('Canonical original checkpoint required')
    for name, expected in METADATA.items():
        path = CHECKPOINT / name
        if not stat.S_ISREG(path.lstat().st_mode) or sha(path) != expected:
            raise ValueError('Original checkpoint metadata changed: ' + name)
    weights = {}
    for name, (size, expected) in WEIGHTS.items():
        path = CHECKPOINT / name
        info = path.lstat()
        cached = (CHECKPOINT / '.cache/huggingface/download' / (name + '.metadata')).read_text().splitlines()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size or cached[:2] != [REVISION, expected]:
            raise ValueError('Original checkpoint weight identity changed: ' + name)
        weights[name] = dict(source=str(path), bytes=size, cached_lfs_sha256=expected,
                             device=info.st_dev, inode=info.st_ino, mtime_ns=info.st_mtime_ns)
    return weights


def verify_weight_content(path, expected_size, expected_sha256):
    """Hash a stable regular file; never deserialize tensors or follow symlinks.

    Device/inode values identify an open file only within this node/process.
    Cross-node identity is the pinned content digest, size, and canonical path.
    """
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ValueError('Weight is not the expected regular file: ' + str(path))
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
        after = os.fstat(stream.fileno())
        current = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if (not stat.S_ISREG(current.st_mode)
            or any(getattr(before, key) != getattr(after, key) or
                   getattr(before, key) != getattr(current, key) for key in fields)):
        raise ValueError('Weight changed during content verification: ' + str(path))
    observed = hasher.hexdigest()
    if observed != expected_sha256:
        raise ValueError('Weight content digest differs from the pinned release: ' + str(path))
    return dict(bytes=before.st_size, sha256=observed, stable=True)


def verify_weight_bindings(recorded, observed):
    """Preserve portable manifest identity and recheck exact tensor-file bytes."""
    if not isinstance(recorded, dict) or set(recorded) != set(WEIGHTS) or set(observed) != set(WEIGHTS):
        raise ValueError('Weight inventory changed')
    portable = ('source', 'bytes', 'cached_lfs_sha256')
    proof = {}
    for name, (size, expected) in WEIGHTS.items():
        if (not isinstance(recorded[name], dict)
                or any(recorded[name].get(key) != observed[name][key] for key in portable)):
            raise ValueError('Portable weight identity changed: ' + name)
        proof[name] = verify_weight_content(CHECKPOINT / name, size, expected)
    return proof


def prepare_view(output):
    """Create a fresh view; refuses existing outputs and preserves failed attempts."""
    output = safe_path(output)
    weights = source_identity()
    output.mkdir(mode=0o700)
    sync_directory(output.parent)
    metadata = {}
    for name in METADATA:
        data = view_bytes(name)
        write_new(output / name, data)
        metadata[name] = hashlib.sha256(data).hexdigest()
    for name, identity in weights.items():
        (output / name).symlink_to(identity['source'])
    sync_directory(output)
    manifest = dict(schema_version=1, model=MODEL, revision=REVISION,
        original_checkpoint=str(CHECKPOINT), serving_view=str(output),
        source_metadata_sha256=METADATA, view_metadata_sha256=metadata, weights=weights,
        changes=CHANGES, original_checkpoint_modified=False,
        weights_content_freshly_hashed=False, weight_bodies_loaded=False,
        runtime_model_loading_verified=False, source_script_sha256=sha(__file__))
    write_new(output / 'serving-view-manifest.json', encoded(manifest))
    validate_view(output)
    return manifest


def validate_view(output):
    """Recheck metadata, exact symlink targets and portable weight content hashes."""
    output = safe_path(output)
    manifest_path = output / 'serving-view-manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('View manifest must be a regular local file')
    manifest = json.loads(manifest_path.read_text())
    weights = source_identity()
    if (manifest.get('serving_view') != str(output) or manifest.get('original_checkpoint') != str(CHECKPOINT)
            or manifest.get('revision') != REVISION or manifest.get('model') != MODEL
            or manifest.get('source_metadata_sha256') != METADATA
            or manifest.get('changes') != CHANGES):
        raise ValueError('View manifest or original checkpoint identity changed')
    allowed = set(METADATA) | set(WEIGHTS) | {'serving-view-manifest.json', 'verification-cache', 'processor-verification.json'}
    if {p.name for p in output.iterdir()} - allowed:
        raise ValueError('Unexpected file could alter the local processor or tokenizer load')
    for name, source_hash in METADATA.items():
        path = output / name
        expected = hashlib.sha256(view_bytes(name)).hexdigest()
        if not stat.S_ISREG(path.lstat().st_mode) or sha(path) != expected or manifest['view_metadata_sha256'].get(name) != expected:
            raise ValueError('View metadata differs from the approved transformations: ' + name)
    for name in WEIGHTS:
        path = output / name
        if not path.is_symlink() or os.readlink(path) != str(CHECKPOINT / name) or path.resolve() != CHECKPOINT / name:
            raise ValueError('View weight symlink changed: ' + name)
    verify_weight_bindings(manifest.get('weights'), weights)
    return manifest


def verify_processor(output):
    """Load only local processor/tokenizer metadata, with network and GPU hidden."""
    output = safe_path(output)
    manifest = validate_view(output)
    cache = safe_path(output / 'verification-cache')
    cache.mkdir(mode=0o700)
    for key in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE',
                'PYTORCH_TRANSFORMERS_CACHE', 'PYTORCH_PRETRAINED_BERT_CACHE', 'HF_MODULES_CACHE',
                'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'TORCH_HOME',
                'TMPDIR', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
        target = safe_path(cache / key.lower())
        target.mkdir(mode=0o700)
        os.environ[key] = str(target)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TOKENIZERS_PARALLELISM='false',
                      CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    previous_connect, previous_connect_ex = socket.socket.connect, socket.socket.connect_ex
    def denied(*args, **kwargs):
        raise RuntimeError('Network is forbidden during offline processor verification')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    try:
        import transformers
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(output), local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
        image = processor.image_processor
        if processor.__class__.__name__ != 'Qwen3VLProcessor' or image.size != {
                'longest_edge': MAX_PIXELS, 'shortest_edge': 65536}:
            raise ValueError('Installed AutoProcessor did not preserve the approved pixel limits')
        if (image.patch_size, image.merge_size, image.temporal_patch_size) != (16, 2, 2):
            raise ValueError('Installed image processor geometry differs from published metadata')
        if getattr(image, 'max_pixels', None) not in (None, MAX_PIXELS):
            raise ValueError('Installed image processor max_pixels overrides the approved size')
        if getattr(image, 'min_pixels', None) not in (None, 65536):
            raise ValueError('Installed image processor min_pixels overrides the approved size')
        original_tokenizer = json.loads((CHECKPOINT / 'tokenizer.json').read_text())
        vocabulary = dict(original_tokenizer['model']['vocab'])
        vocabulary.update({row['content']: row['id'] for row in original_tokenizer['added_tokens']})
        if processor.tokenizer.get_vocab() != vocabulary:
            raise ValueError('Instantiated tokenizer vocabulary differs from unchanged tokenizer.json')
        special = {row['content']: row['id'] for row in original_tokenizer['added_tokens'] if row['special']}
        observed_special = {token: processor.tokenizer.convert_tokens_to_ids(token) for token in special}
        if observed_special != special or set(processor.tokenizer.all_special_ids) != set(special.values()):
            raise ValueError('Tokenizer special-token IDs or membership changed')
        from tokenizers import Tokenizer
        original_backend = Tokenizer.from_file(str(CHECKPOINT / 'tokenizer.json'))
        prompt = '<|im_start|>user\nChoose candidate 2 for the café. <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n'
        original_ids = original_backend.encode(prompt).ids
        effective_ids = processor.tokenizer(prompt)['input_ids']
        if original_ids != effective_ids:
            raise ValueError('Default tokenization changed for the fixed control prompt')
        proof = dict(status='offline_processor_metadata_verified', transformers_version=transformers.__version__,
            python=sys.executable, processor_class=processor.__class__.__name__,
            image_processor_class=image.__class__.__name__, tokenizer_class=processor.tokenizer.__class__.__name__,
            image_size=dict(image.size), patch_size=image.patch_size, merge_size=image.merge_size,
            temporal_patch_size=image.temporal_patch_size,
            vocabulary_entries=len(vocabulary), vocabulary_exact_match=True,
            all_special_token_ids=special, special_token_ids_exact_match=True,
            fixed_prompt=prompt, original_token_ids=original_ids, effective_token_ids=effective_ids,
            chat_template_sha256=hashlib.sha256(processor.chat_template.encode()).hexdigest(),
            manifest_sha256=sha(output / 'serving-view-manifest.json'),
            source_script_sha256=sha(__file__), images_processed=0, model_weights_loaded=False,
            gpu_model_validation=False, inference_requests=0, network_allowed=False)
        if proof['chat_template_sha256'] != METADATA['chat_template.jinja']:
            raise ValueError('Installed processor chat template differs from the pinned source')
        validate_view(output)
        write_new(output / 'processor-verification.json', encoded(proof))
        return proof
    finally:
        socket.socket.connect, socket.socket.connect_ex = previous_connect, previous_connect_ex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-processor', action='store_true')
    parser.add_argument('--check-existing', action='store_true')
    args = parser.parse_args()
    manifest = validate_view(args.output) if args.check_existing else prepare_view(args.output)
    result = dict(manifest=manifest)
    if args.verify_processor:
        result['processor'] = verify_processor(args.output)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
