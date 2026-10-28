"""Offline source-bound supplemental-audit safety checks."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dispatch_erratum_tests_target',ROOT/'scripts/audit_webprm_table5_dispatch_erratum.py')
erratum=importlib.util.module_from_spec(spec);spec.loader.exec_module(erratum)


class ErratumTests(unittest.TestCase):
    def test_only_two_reviewed_functions_are_executable_replacements(self):
        namespace={}
        exec(erratum.replacement_functions(ROOT/'scripts/audit_webprm_table5_cell.py'),namespace)
        self.assertEqual(set(namespace)-{'__builtins__'},erratum.FUNCTIONS)

    def test_unreviewed_source_is_rejected_before_compilation(self):
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='webprm-erratum-') as folder:
            source=Path(folder)/'auditor.py'
            source.write_text((ROOT/'scripts/audit_webprm_table5_cell.py').read_text()+'\nraise AssertionError("must not execute")\n')
            with self.assertRaisesRegex(ValueError,'reviewed'):
                erratum.replacement_functions(source)

    def test_nonfrozen_dependency_cannot_be_hidden_by_frozen_source_files(self):
        with patch.dict(sys.modules,{'openwebrl.foreign_fixture':SimpleNamespace(__file__='/tmp/foreign.py')}):
            with self.assertRaisesRegex(ValueError,'fresh process'):
                erratum.loaded_dependencies_are_frozen(ROOT)

    def test_frozen_dependency_is_allowed(self):
        with patch.dict(sys.modules,{'openwebrl.fixture':SimpleNamespace(__file__=str(ROOT/'openwebrl/fixture.py'))}):
            erratum.loaded_dependencies_are_frozen(ROOT)


if __name__=='__main__':unittest.main()
