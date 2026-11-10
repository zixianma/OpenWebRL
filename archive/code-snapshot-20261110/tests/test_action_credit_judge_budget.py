from copy import deepcopy
import unittest

from openwebrl.action_credit_judge_budget import (gpt41_context_upper_bound,
    maximum_request_microdollars, input_bound_protocol_sha256)


def payload():
    return dict(model="gpt-4.1", max_completion_tokens=1024, seed=42,
                messages=[dict(role="system", content="Judge"), dict(role="user", content=[
                    dict(type="text", text="History"),
                    dict(type="image_url", image_url=dict(url="data:image/png;base64,fixture", detail="auto"))])])


class ContextBudget(unittest.TestCase):
    def test_full_context_envelope_covers_all_image_detail_modes(self):
        for detail in ("low", "high", "auto"):
            value = payload()
            value["messages"][1]["content"][1]["image_url"]["detail"] = detail
            self.assertEqual(gpt41_context_upper_bound(value), 1_047_576)
        self.assertEqual(maximum_request_microdollars(1024), 2_103_344)
        self.assertEqual(len(input_bound_protocol_sha256()), 64)

    def test_unsupported_cost_multipliers_and_models_fail_before_dispatch(self):
        for key, value in (("n", 2), ("tools", []), ("stream", True),
                           ("service_tier", "priority"), ("model", "gpt-4.1-mini"),
                           ("max_completion_tokens", True), ("max_completion_tokens", 32769)):
            changed = payload(); changed[key] = value
            with self.assertRaises(ValueError):
                gpt41_context_upper_bound(changed)

    def test_unknown_content_and_image_detail_fail_closed(self):
        for part in (dict(type="input_audio", input_audio={}),
                     dict(type="image_url", image_url=dict(url="https://example.test/x", detail="original")),
                     dict(type="image_url", image_url=dict(url="x", extra="unchecked"))):
            value = deepcopy(payload()); value["messages"][1]["content"].append(part)
            with self.assertRaises(ValueError):
                gpt41_context_upper_bound(value)


if __name__ == "__main__":
    unittest.main()
