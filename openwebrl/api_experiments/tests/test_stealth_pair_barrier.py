from pathlib import Path
import sys,tempfile,threading,time,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduce import stealth_pilot as p
class Tests(unittest.TestCase):
 def test_released_waiter_can_admit_before_peer_wakes(self):
  for after in (False,True):
   with tempfile.TemporaryDirectory(dir='/tmp')as d:
    c=p.Control(d);c.initialize(time.time()+14400);waiting=threading.Event();release=threading.Event();events=[]
    def pause(_):waiting.set();release.wait(timeout=2)
    def first():
     try:c.pair(p.ARMS[0],0,after=after,sleep=pause);events.append('released')
     except Exception as e:events.append(type(e).__name__)
    thread=threading.Thread(target=first);thread.start();self.assertTrue(waiting.wait(timeout=2))
    c.pair(p.ARMS[1],0,after=after);c.admission(Path(d)/'actual-next-create',{})
    release.set();thread.join(timeout=2);self.assertEqual(events,['released']);self.assertFalse(thread.is_alive())
 def test_prior_finished_phase_required_before_next_admission(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   c=p.Control(d);c.initialize(time.time()+14400)
   with self.assertRaisesRegex(ValueError,'Previous phase must finish'):c.pair(p.ARMS[0],1)
   self.assertEqual(c.snapshot()['arrivals'],{})
 def test_abort_still_blocks_latched_phase(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   c=p.Control(d);c.initialize(time.time()+14400)
   with c.locked()as state:state['released_pair_phases']['arrivals:0']=True
   c.halt('actual-peer-failure')
   with self.assertRaises(ValueError):c.pair(p.ARMS[0],0)
if __name__=='__main__':unittest.main()
