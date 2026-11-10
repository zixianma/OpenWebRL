"""Per-HTTP durable accounting, without provider requests or inferred budgets."""
from pathlib import Path
import tempfile
import unittest

from openwebrl.action_credit_state import CampaignManifest, CampaignStore, StateError, BudgetExceeded
from test_action_credit_state import manifest_data, sha, cost, reserve


def http_cost(microdollars=10):
    return dict(gpu_seconds=0, browser_dispatches=0, api_requests=1, api_microdollars=microdollars)


class HTTPAccountingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="action-credit-http-", dir="/tmp")
        self.addCleanup(tmp.cleanup)
        self.store = CampaignStore.create(Path(tmp.name)/"store", CampaignManifest.from_dict(manifest_data()))
        draw = reserve(self.store)["draw_id"]
        self.episode_cost = dict(gpu_seconds=1, browser_dispatches=1, api_requests=0, api_microdollars=0)
        self.store.start_attempt(draw, "episode", reserved_cost=self.episode_cost)

    def reserve(self, request="http1", cents=10):
        return self.store.reserve_subrequest(request, parent_attempt_id="episode",
            payload_sha256=sha(request), reserved_cost=http_cost(cents))

    def test_crash_reservation_persists_and_idempotence_forbids_resend(self):
        self.assertTrue(self.reserve()["created"])
        reopened = CampaignStore(self.store.root, self.store.manifest)
        self.assertFalse(reopened.reserve_subrequest("http1", parent_attempt_id="episode",
            payload_sha256=sha("http1"), reserved_cost=http_cost())["created"])
        self.assertEqual(reopened.snapshot()["charged_resources"]["api_requests"], 1)
        with self.assertRaises(StateError):
            reopened.finish_attempt("episode", status="invalid", actual_cost=self.episode_cost, reason="interrupted")

    def test_unknown_and_failed_retries_keep_all_reserved_costs(self):
        self.reserve()
        self.store.settle_subrequest("http1", status="unknown", actual_cost=http_cost(0), reason="crash after dispatch")
        self.reserve("http2", 20)
        self.store.settle_subrequest("http2", status="failed", actual_cost=http_cost(0), reason="timeout")
        self.store.finish_attempt("episode", status="invalid", actual_cost=self.episode_cost, reason="judge unavailable")
        totals = self.store.snapshot()["charged_resources"]
        self.assertEqual((totals["api_requests"], totals["api_microdollars"]), (2, 30))
        self.assertEqual(totals["browser_dispatches"], 1)

    def test_metered_success_releases_only_unused_reservation_and_keeps_attempt(self):
        self.reserve("http1", 90)
        self.store.settle_subrequest("http1", status="succeeded", actual_cost=http_cost(2),
                                    response_sha256=sha("complete reply and exact usage"))
        self.assertEqual(self.store.snapshot()["charged_resources"]["api_microdollars"], 2)
        self.reserve("http2", 90)
        state = self.store.snapshot()
        self.assertEqual(state["charged_resources"]["api_microdollars"], 92)
        self.assertEqual(state["charged_resources"]["api_requests"], 2)
        self.assertEqual(state["subrequests"]["http1"]["reserved"]["api_microdollars"], 90)
        reopened = CampaignStore(self.store.root, self.store.manifest)
        self.assertEqual(reopened.snapshot()["charged_resources"], state["charged_resources"])

    def test_failed_known_partial_cost_still_retains_full_liability(self):
        self.reserve("http1", 30)
        self.store.settle_subrequest("http1", status="failed", actual_cost=http_cost(2), reason="partial provider reply")
        self.assertEqual(self.store.snapshot()["charged_resources"]["api_microdollars"], 30)

    def test_reservation_overrun_halts_even_below_campaign_total_cap(self):
        self.reserve("http1", 10)
        with self.assertRaises(BudgetExceeded):
            self.store.settle_subrequest("http1", status="succeeded", actual_cost=http_cost(11), response_sha256=sha("reply"))
        self.assertEqual(self.store.snapshot()["charged_resources"]["api_microdollars"], 11)
        self.assertTrue(self.store.snapshot()["halt"])

    def test_http_caps_include_other_attempt_reservations(self):
        self.reserve("http1", 90)
        with self.assertRaises(BudgetExceeded):
            self.reserve("http2", 20)
        self.assertEqual(len(self.store.snapshot()["subrequests"]), 1)

    def test_actual_overrun_is_saved_then_halts(self):
        self.reserve()
        with self.assertRaises(BudgetExceeded):
            self.store.settle_subrequest("http1", status="succeeded", actual_cost=http_cost(110), response_sha256=sha("reply"))
        state = self.store.snapshot()
        self.assertEqual(state["charged_resources"]["api_microdollars"], 110)
        self.assertTrue(state["halt"])
        self.assertEqual(state["subrequests"]["http1"]["status"], "succeeded")

    def test_changed_payload_or_settlement_rejected(self):
        self.reserve()
        with self.assertRaises(StateError):
            self.store.reserve_subrequest("http1", parent_attempt_id="episode",
                payload_sha256=sha("changed"), reserved_cost=http_cost())
        self.store.settle_subrequest("http1", status="succeeded", actual_cost=http_cost(2), response_sha256=sha("reply"))
        with self.assertRaises(StateError):
            self.store.settle_subrequest("http1", status="succeeded", actual_cost=http_cost(3), response_sha256=sha("reply"))

    def test_episode_cannot_duplicate_http_actual_cost(self):
        self.reserve()
        self.store.settle_subrequest("http1", status="failed", actual_cost=http_cost(), reason="timeout")
        with self.assertRaises(StateError):
            self.store.finish_attempt("episode", status="invalid", actual_cost=cost(), reason="timeout")

    def test_legacy_episode_api_reservation_cannot_also_use_subrequests(self):
        self.store.finish_attempt("episode", status="invalid", actual_cost=self.episode_cost, reason="fixture")
        draw_id = next(iter(self.store.snapshot()["draws"]))
        self.store.start_attempt(draw_id, "legacy", reserved_cost=cost())
        with self.assertRaises(StateError):
            self.store.reserve_subrequest("http1", parent_attempt_id="legacy", payload_sha256=sha("payload"), reserved_cost=http_cost())


if __name__ == "__main__":
    unittest.main()
