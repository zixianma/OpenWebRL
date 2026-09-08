"""Regression coverage for the actual two-parser router option merge."""
import argparse
import sys
import unittest
from unittest.mock import patch

from sglang_router.launch_router import RouterArgs
from slime.backends.sglang_utils.arguments import sglang_parse_args
from slime.utils.arguments import get_slime_extra_args_provider


class RouterCliPrecedenceTest(unittest.TestCase):
    def parse_router(self, extra):
        argv = ['train', '--rollout-batch-size', '48', *extra]
        with patch.object(sys, 'argv', argv):
            server = sglang_parse_args()
            parser = get_slime_extra_args_provider()(argparse.ArgumentParser(allow_abbrev=False))
            main, _ = parser.parse_known_args()
            for key, value in vars(server).items():
                setattr(main, key, value)
            return RouterArgs.from_cli_args(main, use_router_prefix=True)

    def test_explicit_thresholds_survive_server_namespace_merge(self):
        router = self.parse_router(['--router-balance-abs-threshold', '2',
                                   '--router-balance-rel-threshold', '1.1'])
        self.assertEqual(router.balance_abs_threshold, 2)
        self.assertEqual(router.balance_rel_threshold, 1.1)

    def test_existing_defaults_are_preserved(self):
        router = self.parse_router([])
        self.assertEqual(router.balance_abs_threshold, 10)
        self.assertEqual(router.balance_rel_threshold, 1.2)


if __name__ == '__main__':
    unittest.main()
