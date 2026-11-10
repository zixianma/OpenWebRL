#!/usr/bin/env python3
"""Uniform weight average ("checkpoint soup") of Megatron torch_dist checkpoints from one run (CPU only).

The output is a new iter_NNNNNNN directory that reuses the base checkpoint for everything except the
averaged tensors: every model parameter (vision_model.*, language_model.*) and the optimizer's fp32 master
copy (optimizer.state.fp32_param.*) become the elementwise mean over --base and --others. Adam moments,
scheduler, RNG and common.pt stay those of --base. Base shard files are hardlinked, never modified; the
averaged tensors go to a new shard file and the merged .metadata points each averaged key at it.
All inputs must share keys, shapes and dtypes. Verification reloads a sample of keys from the output.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import shutil

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys


def averaged_key(k):
    return not k.startswith(('optimizer', 'rng_state')) or k.startswith('optimizer.state.fp32_param.')


def load(path, keys):
    return _load_state_dict_from_keys(keys=set(keys), checkpoint_id=str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True, help='checkpoint whose non-averaged state is kept')
    ap.add_argument('--others', type=Path, nargs='+', required=True)
    ap.add_argument('--output-root', type=Path, required=True, help='new run dir; writes runtime/iter_NNNNNNN')
    ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()
    cks = [a.base, *a.others]
    mds = [FileSystemReader(str(c)).read_metadata() for c in cks]
    keys = sorted(k for k, m in mds[0].state_dict_metadata.items() if averaged_key(k) and hasattr(m, 'size'))
    resplit = set()
    # non-tensor entries (e.g. Transformer Engine _extra_state bytes) stay those of --base
    for c, md in zip(cks[1:], mds[1:]):
        for k in keys:
            m0, m = mds[0].state_dict_metadata[k], md.state_dict_metadata.get(k)
            if m is None or not hasattr(m, 'size') or m.properties.dtype != m0.properties.dtype:
                raise ValueError(f'{c}: key {k} missing or dtype differs')
            if m.size != m0.size:
                # distributed-optimizer state saved with a different data-parallel split:
                # [layers, dp, 1, chunk]; allowed only for fp32 masters with equal per-layer numel
                lead = len(m0.size) == 4
                if not k.startswith('optimizer.state.fp32_param.') or (lead and m.size[0] != m0.size[0]) or \
                        torch.Size(m.size).numel() != torch.Size(m0.size).numel():
                    raise ValueError(f'{c}: key {k} shape {tuple(m.size)} vs {tuple(m0.size)}')
                resplit.add(k)
    out = a.output_root/'runtime'/a.base.name
    if out.exists():
        raise ValueError(f'{out} exists')
    tmp = a.output_root/'soup-tmp'
    out.mkdir(parents=True)
    avg, resplit_check = {}, {}
    for i in range(0, len(keys), a.batch):
        batch = keys[i:i+a.batch]
        parts = [load(c, batch) for c in cks]
        for k in batch:
            shape = parts[0][k].shape
            if k in resplit:
                # per-layer (or whole-tensor) flattened order is DP-split independent; check it against the model weight
                rows = shape[0] if len(shape) == 4 else 1
                model = k[len('optimizer.state.fp32_param.'):]
                ref = [load(c, [model])[model].float().reshape(rows, -1) for c in cks]
                for p, r in zip(parts, ref):
                    flat = p[k].float().reshape(rows, -1)
                    err = float((flat-r).abs().max()/r.abs().max())
                    if err > 1e-2:
                        raise ValueError(f'{k}: fp32 master does not match model weight after reshape ({err:.3g})')
                    resplit_check[k] = max(resplit_check.get(k, 0), err)
                t = torch.stack([p[k].float().reshape(rows, -1) for p in parts]).mean(0).reshape(shape)
            else:
                t = torch.stack([p[k].float() for p in parts]).mean(0)
            avg[k] = t.to(parts[0][k].dtype).contiguous()
        del parts
        print(f'averaged {min(i+a.batch, len(keys))}/{len(keys)}', flush=True)
    dcp.save(avg, storage_writer=FileSystemWriter(str(tmp)), no_dist=True)
    tmd = FileSystemReader(str(tmp)).read_metadata()
    # Merge: base metadata, with averaged keys pointing at the new shard file(s).
    md = mds[0]
    rename = {}
    for info in tmd.storage_data.values():
        rename.setdefault(info.relative_path, f'__soup_{len(rename)}.distcp')
    for src, dst in rename.items():
        shutil.move(str(tmp/src), str(out/dst))
    kept = {idx: info for idx, info in md.storage_data.items() if idx.fqn not in avg}
    for idx, info in tmd.storage_data.items():
        info.relative_path = rename[info.relative_path]
        kept[idx] = info
    md.storage_data = kept
    for k in avg:
        md.state_dict_metadata[k] = tmd.state_dict_metadata[k]
    if getattr(md, 'planner_data', None) and getattr(tmd, 'planner_data', None):
        for k in avg:
            if k in tmd.planner_data:
                md.planner_data[k] = tmd.planner_data[k]
    for f in a.base.iterdir():
        if f.name.endswith('.distcp'):
            os.link(f.resolve(), out/f.name)  # hardlink: shards must resolve inside the checkpoint dir
        elif f.name != '.metadata':
            shutil.copy2(f, out/f.name)
    with open(out/'.metadata', 'wb') as fh:
        pickle.dump(md, fh)
    shutil.rmtree(tmp)
    # rollout cursor and marker, as the evaluation loader expects next to the checkpoint
    it = int(a.base.name.split('_')[1])
    (a.output_root/'runtime/rollout').mkdir(exist_ok=True)
    cur = a.base.parent/'rollout'/f'global_dataset_state_dict_{it}.pt'
    (a.output_root/'runtime/rollout'/cur.name).symlink_to(cur.resolve())
    (a.output_root/'runtime/latest_checkpointed_iteration.txt').write_text(f'{it}\n')
    # verify a sample of averaged and untouched keys
    sample = keys[::max(1, len(keys)//25)]
    untouched = [k for k in mds[0].state_dict_metadata if k.startswith('optimizer.state.exp_avg.')][:5]
    got = load(out, sample+untouched)
    base_u = load(a.base, untouched)
    bad = [k for k in sample if not torch.equal(got[k], avg[k])]+[k for k in untouched if not torch.equal(got[k], base_u[k])]
    diff = {k: float((got[k].float()-load(a.base, [k])[k].float()).abs().mean()) for k in sample[:3]}
    report = dict(base=str(a.base), others=[str(c) for c in a.others], averaged_keys=len(avg),
                  total_keys=len(md.state_dict_metadata), verified_keys=len(sample)+len(untouched), mismatches=bad,
                  mean_abs_change_vs_base_sample=diff, output=str(out),
                  resplit_fp32_keys_max_rel_err_vs_model=resplit_check)
    (a.output_root/'soup-report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if bad:
        raise SystemExit('verification failed')


if __name__ == '__main__':
    main()
