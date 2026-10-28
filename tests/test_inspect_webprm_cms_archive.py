"""Adversarial synthetic archive checks; no network/container/model execution."""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import inspect_webprm_cms_archive as cms


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def tar_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, value in entries:
            member = tarfile.TarInfo(name)
            if isinstance(value, tuple):
                member.type, member.linkname = value
                archive.addfile(member)
            else:
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
    return buffer.getvalue()


class CmsIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='webprm-cms-unit-', dir='/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.deadline = time.time() + 30

    def make_archive(self, entries=None, *, config_change=None, manifest_change=None,
                     outer_extra=(), corrupt_layer=False, corrupt_config=False):
        layer = tar_bytes(entries or [('var/lib/mysql/synthetic-data', b'baseline')])
        config = dict(os='linux', architecture='amd64', rootfs=dict(type='layers',
            diff_ids=['sha256:' + cms.sha(layer)]), config=dict(Entrypoint=['/entrypoint'], Cmd=['serve'], User=''))
        if config_change:
            config_change(config)
        raw_config = encoded(config)
        config_name = cms.sha(raw_config) + '.json'
        manifest = [dict(Config=config_name, RepoTags=[cms.IMAGE_TAG], Layers=['layer/layer.tar'])]
        if manifest_change:
            manifest_change(manifest)
        outer = tar_bytes([('manifest.json', encoded(manifest)),
            (config_name, raw_config + (b' ' if corrupt_config else b'')),
            ('layer/layer.tar', layer + (b'changed' if corrupt_layer else b'')), *outer_extra])
        path = self.root / cms.ARCHIVE_NAME
        path.write_bytes(outer)
        return path, replace(cms.Limits(), archive_bytes=len(outer))

    def inspect(self, path, limits):
        return cms.inspection(path, deadline=self.deadline, limits=limits)

    def test_identity_hashes_but_never_claims_hosting_or_original_provenance(self):
        path, limits = self.make_archive()
        result = self.inspect(path, limits)
        self.assertEqual(result['archive_sha256'], cms.sha(path.read_bytes()))
        self.assertEqual(result['layers'][0]['diff_id'], 'sha256:' + result['layers'][0]['sha256'])
        self.assertEqual(result['cumulative_layer_counts']['regular_bytes'], 8)
        self.assertFalse(result['ready_to_host'])
        self.assertFalse(result['ready_to_import'])
        self.assertFalse(result['extraction_performed'])
        self.assertEqual(result['network_calls'], 0)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [cms.ARCHIVE_NAME])

    def test_original_tag_and_exact_archive_size_are_required(self):
        path, limits = self.make_archive(manifest_change=lambda x: x[0].update(RepoTags=['substitute:latest']))
        with self.assertRaisesRegex(cms.GateClosed, 'original CMS'):
            self.inspect(path, limits)
        with self.assertRaisesRegex(cms.GateClosed, 'byte length'):
            self.inspect(path, replace(limits, archive_bytes=limits.archive_bytes + 1))

    def test_config_and_raw_layer_digests_are_independently_checked(self):
        for kind in ('config', 'layer'):
            with self.subTest(kind=kind):
                path, limits = self.make_archive(**{'corrupt_' + kind: True})
                with self.assertRaisesRegex(cms.GateClosed, 'digest'):
                    self.inspect(path, limits)

    def test_missing_duplicate_and_out_of_order_layer_references_fail(self):
        changes = [lambda x: x[0].update(Layers=['missing.tar']),
                   lambda x: x[0].update(Layers=['layer/layer.tar', 'layer/layer.tar'])]
        for change in changes:
            path, limits = self.make_archive(manifest_change=change)
            with self.assertRaises(cms.GateClosed):
                self.inspect(path, limits)
        path, limits = self.make_archive(config_change=lambda x: x['rootfs'].update(diff_ids=['sha256:' + '0' * 64]))
        with self.assertRaisesRegex(cms.GateClosed, 'digest'):
            self.inspect(path, limits)

    def test_outer_path_traversal_duplicate_and_symlink_fail(self):
        for extra in ([('../escape', b'x')], [('./manifest.json', b'{}')],
                      [('link', (tarfile.SYMTYPE, '/gpfs/home/zixianma'))]):
            path, limits = self.make_archive(outer_extra=extra)
            with self.assertRaises(cms.GateClosed):
                self.inspect(path, limits)

    def test_layer_path_and_link_escape_fail_without_extraction(self):
        for entries in ([('../escape', b'x')], [('/absolute', b'x')],
                        [('var/link', (tarfile.SYMTYPE, '../../escape'))]):
            path, limits = self.make_archive(entries)
            with self.assertRaises(cms.GateClosed):
                self.inspect(path, limits)
            self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_special_files_absolute_links_and_whiteouts_remain_review_gates(self):
        path, limits = self.make_archive([
            ('dev/custom', (tarfile.CHRTYPE, '')),
            ('var/run', (tarfile.SYMTYPE, '/run')),
            ('var/.wh..wh..opq', b'')])
        result = self.inspect(path, limits)
        self.assertEqual(len(result['hazards']), 3)
        self.assertFalse(result['ready_to_import'])

    def test_cumulative_expansion_metadata_and_member_caps(self):
        path, limits = self.make_archive([('one', b'a' * 32), ('two', b'b' * 32)])
        for smaller in (replace(limits, cumulative_regular_bytes=40),
                        replace(limits, metadata_bytes=8), replace(limits, outer_members=2),
                        replace(limits, layer_members=1)):
            with self.assertRaises(cms.GateClosed):
                self.inspect(path, smaller)

    def test_expired_deadline_and_oversized_pax_read_rejected_before_read(self):
        path, limits = self.make_archive()
        with self.assertRaisesRegex(cms.GateClosed, 'deadline'):
            cms.inspection(path, limits=limits, deadline=time.time() - 1)
        stream = io.BytesIO(b'small')
        reader = cms.BoundedReader(stream, 1024, self.deadline)
        with self.assertRaisesRegex(cms.GateClosed, 'oversized'):
            reader.read(2**40)
        self.assertEqual(stream.tell(), 0)

    def test_source_replacement_detected_even_with_same_bytes(self):
        path, limits = self.make_archive()
        real_hash = cms.stream_sha
        def replace_after_hash(stream, expected, deadline):
            value = real_hash(stream, expected, deadline)
            if expected == limits.archive_bytes:
                replacement = self.root / 'replacement'
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            return value
        with patch.object(cms, 'stream_sha', side_effect=replace_after_hash):
            with self.assertRaisesRegex(cms.GateClosed, 'replaced'):
                self.inspect(path, limits)

    def test_fifo_and_symlink_sources_rejected(self):
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        with self.assertRaisesRegex(cms.GateClosed, 'byte length'):
            self.inspect(fifo, cms.Limits())
        link = self.root / 'link'
        link.symlink_to(fifo)
        with self.assertRaisesRegex(cms.GateClosed, 'nonsymlink'):
            self.inspect(link, cms.Limits())

    def test_output_is_fresh_private_and_bounded(self):
        path = self.root / 'inspection.json'
        cms.write_new(path, {'status': 'synthetic'})
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            cms.write_new(path, {})
        with self.assertRaisesRegex(cms.GateClosed, 'one-MiB'):
            cms.write_new(self.root / 'large.json', {'large': 'x' * 2**20})
        with self.assertRaisesRegex(cms.GateClosed, 'nonsymlink'):
            cms.checked_path('/gpfs/home/zixianma/forbidden.json')

    def test_duplicate_json_fields_rejected(self):
        with self.assertRaisesRegex(cms.GateClosed, 'Duplicate JSON'):
            cms.decode(b'{"Config":"one","Config":"two"}')

    def test_approval_must_bind_original_proposal_source_job_and_cumulative_caps(self):
        now = time.time()
        proposal = self.root / 'proposal.json'; proposal.write_bytes(b'{}')
        approval = self.root / 'approval.json'
        approved = dict(status='approved', proposal_sha256=cms.sha(b'{}'),
            script_sha256=cms.sha(Path(cms.__file__).read_bytes()), exact_user_approval='synthetic only',
            job_id='synthetic', allocation_started_epoch=now - 10,
            deadline_epoch=now + 1800, prior_consumed_seconds=100)
        env = dict(SLURM_JOB_ID='synthetic', SLURM_CPUS_PER_TASK='4', SLURM_MEM_PER_NODE='16384')
        with patch.object(cms, 'PROPOSAL_SHA', cms.sha(b'{}')):
            approval.write_bytes(encoded(approved))
            self.assertEqual(cms.approved_deadline(approval, proposal, env), approved['deadline_epoch'] - 600)
            for key, value in [('status', 'pending'), ('script_sha256', 'changed'),
                               ('job_id', 'other'), ('prior_consumed_seconds', 7000),
                               ('deadline_epoch', now + 300), ('exact_user_approval', '')]:
                approval.write_bytes(encoded(dict(approved, **{key: value})))
                with self.assertRaises(cms.GateClosed):
                    cms.approved_deadline(approval, proposal, env)
            approval.write_bytes(encoded(approved))
            for devices in ('0', '1'):
                with self.assertRaises(cms.GateClosed):
                    cms.approved_deadline(approval, proposal, dict(env, SLURM_JOB_GPUS=devices))

    def test_pending_cli_never_opens_the_archive_or_writes_a_receipt(self):
        proposal = self.root / 'proposal.json'; proposal.write_bytes(b'{}')
        approval = self.root / 'pending.json'; approval.write_bytes(b'{"status":"pending"}')
        output = self.root / 'receipt.json'
        argv = ['inspector', '--proposal', str(proposal), '--approval', str(approval),
                '--archive', str(self.root / cms.ARCHIVE_NAME), '--output', str(output)]
        with patch.object(cms, 'PROPOSAL_SHA', cms.sha(b'{}')), patch.object(sys, 'argv', argv):
            with patch.object(cms, 'inspection') as inspect:
                with self.assertRaises(cms.GateClosed):
                    cms.main()
                inspect.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
