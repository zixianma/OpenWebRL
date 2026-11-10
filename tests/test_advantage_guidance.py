import math
import unittest

from openwebrl.advantage_guidance import action_score, guided_token_logprobs


class AdvantageGuidanceTests(unittest.TestCase):
    def test_complete_action_uses_sum_and_end_token(self):
        self.assertAlmostEqual(action_score([-.2, -.3, -.4], [-.3, -.5, -.8]), .7)
        with self.assertRaisesRegex(ValueError, "aligned"):
            action_score([-.2], [-.3, -.5])


    def test_zero_guidance_preserves_actor_and_reference_case_recovers_positive(self):
        base = [math.log(.8), math.log(.2)]
        positive = [math.log(.25), math.log(.75)]
        for actual, wanted in zip(guided_token_logprobs(base, positive, base, 0), base):
            self.assertAlmostEqual(actual, wanted)
        for actual, wanted in zip(guided_token_logprobs(base, positive, base, 1), positive):
            self.assertAlmostEqual(actual, wanted)


    def test_local_guidance_differs_from_global_sequence_tilt(self):
        # At the first step, positive and negative assign equal mass to each
        # branch; their ratios only diverge at the second step. A global tilt
        # anticipates this future difference, while the local first step cannot.
        base_first = [math.log(.5), math.log(.5)]
        local_first = guided_token_logprobs(base_first, base_first, base_first, .5)
        base_second = [[math.log(.9), math.log(.1)], [math.log(.9), math.log(.1)]]
        positive_second = [[math.log(.99), math.log(.01)], [math.log(.1), math.log(.9)]]
        negative_second = [[math.log(.5), math.log(.5)], [math.log(.5), math.log(.5)]]
        future = [sum(math.exp(b + .5 * (p - n)) for b, p, n in zip(bs, ps, ns))
                  for bs, ps, ns in zip(base_second, positive_second, negative_second)]
        globally_tilted_first = future[0] / sum(future)
        self.assertAlmostEqual(math.exp(local_first[0]), .5)
        self.assertGreater(abs(globally_tilted_first - .5), .05)


    def test_invalid_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "support"):
            guided_token_logprobs([0.], [0.], [-math.inf], 1)


if __name__ == "__main__":
    unittest.main()
