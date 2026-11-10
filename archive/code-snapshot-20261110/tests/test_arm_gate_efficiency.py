import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from monitor_arm_gate_efficiency import ingest, summarize


class IncrementalMetrics(unittest.TestCase):
    def test_partial_writes_and_later_phase_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=root/'collection.log'
            line="[worker] - perf 23: {'perf/actor_train_time': 120.0}\n"
            log.write_text(line[:-3]);cache=ingest(root,{})
            self.assertEqual(cache['offset'],0)
            log.write_text(line);cache=ingest(root,cache)
            with log.open('a') as f:f.write("[worker] - rollout 23: {'rollout/raw_reward_mean': 0.5}\n")
            cache=ingest(root,cache);again=ingest(root,cache)
            self.assertEqual(again['rows']['24'],{'perf/actor_train_time':120.,'rollout/raw_reward_mean':.5})
            gate=root/'iterations/0023';gate.mkdir(parents=True)
            (gate/'training_gate.json').write_text(json.dumps({'expected_optimizer_updates':12}))
            self.assertEqual(summarize(cache['rows'],root)[0]['optimizer_seconds_per_update'],10.)


if __name__=='__main__':unittest.main()
