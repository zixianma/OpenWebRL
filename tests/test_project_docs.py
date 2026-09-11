"""Protect neighboring experiment records during consolidated report updates."""
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from project_docs import render_section, rewrite_links, write_document_section
from check_project_docs import inspect


def concurrent_writer(directory, name, barrier):
    barrier.wait(timeout=10)
    for n in range(4):
        write_document_section(Path(directory) / name, f'# Endpoint\n\n{name}: update {n}\n')


class ProjectDocsTests(unittest.TestCase):
    def test_repository_consolidation_is_consistent(self):
        self.assertEqual(inspect(), [])

    def test_update_preserves_neighbor_and_does_not_recreate_old_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = 'ARM_JOINT_SFT_RESULTS.md', 'ARM_JOINT_DPO_RESULTS.md'
            target = write_document_section(root / a, '# SFT\n\n102/300\n')
            write_document_section(root / b, '# DPO\n\n104/300\n')
            before = target.read_text()
            write_document_section(root / a, '# SFT\n\nUpdated SFT report\n')
            after = target.read_text()
            self.assertIn('Updated SFT report', after)
            self.assertIn(render_section(b, '# DPO\n\n104/300\n'), after)
            self.assertIn('# ARM results:', before)
            self.assertEqual(after.count(f'<!-- document:{a}:start -->'), 1)
            self.assertFalse((root / a).exists())
            self.assertFalse((root / b).exists())

    def test_concurrent_reports_both_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context('fork')
            barrier = context.Barrier(2)
            names = ['ARM_JOINT_SFT_RESULTS.md', 'ARM_JOINT_DPO_RESULTS.md']
            workers = [context.Process(target=concurrent_writer, args=(directory, name, barrier)) for name in names]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=15)
                self.assertEqual(worker.exitcode, 0)
            text = (Path(directory) / 'ARM_RESULTS.md').read_text()
            for name in names:
                self.assertIn(f'{name}: update 3', text)

    def test_failed_atomic_write_preserves_document(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'ARM_JOINT_SFT_RESULTS.md'
            target = write_document_section(old, '# SFT\n\nOriginal\n')
            before = target.read_bytes()
            original_write = Path.write_text

            def write(path, text, *args, **kwargs):
                if path == target.with_suffix('.md.tmp'):
                    raise OSError('Disk quota exceeded')
                return original_write(path, text, *args, **kwargs)

            with patch.object(Path, 'write_text', write), self.assertRaises(OSError):
                write_document_section(old, '# SFT\n\nReplacement\n')
            self.assertEqual(target.read_bytes(), before)

    def test_fragment_links_and_fenced_headings(self):
        text = '# SFT\n\n## Results\n\n[self](#results) [other](ARM_JOINT_DPO_RESULTS.md)\n'
        text += '[external](https://example.com/ARM_JOINT_DPO_RESULTS.md)\n\n```sh\n# shell comment\n```\n'
        rendered = render_section('ARM_JOINT_SFT_RESULTS.md', text)
        self.assertIn('ARM_RESULTS.md#arm-joint-sft-results--results', rendered)
        self.assertIn('ARM_RESULTS.md#arm-joint-dpo-results', rendered)
        self.assertIn('https://example.com/ARM_JOINT_DPO_RESULTS.md', rendered)
        self.assertIn('```sh\n# shell comment\n```', rendered)
        self.assertNotIn('--shell-comment', rendered)
        self.assertEqual(rewrite_links(rendered, 'ARM_JOINT_SFT_RESULTS.md'), rendered)

    def test_malformed_section_does_not_clobber_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'ARM_RESULTS.md'
            target.write_text('Manual notes\n<!-- document:ARM_JOINT_SFT_RESULTS.md:start -->\n')
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                write_document_section(root / 'ARM_JOINT_SFT_RESULTS.md', '# SFT\nNew\n')
            self.assertEqual(target.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
