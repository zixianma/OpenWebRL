import hashlib,json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from openwebrl.luna_qwen_metrics import digest
from openwebrl.sol_serial_parallel_lineage import accepted_plan_hashes
class LineageTests(unittest.TestCase):
 def test_registered_unchanged_science_is_accepted_but_new_cohort_or_cap_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   old={'source':'old','source_hashes':{'a':'old'},'task_ids':['a'],'limits':{'calls':10},'protocol':{'actor':'fixed'}}
   path=Path(tmp)/'old.json';path.write_text(json.dumps(old))
   ref={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'plan_sha256':digest(old)}
   new=dict(old,source='new',source_hashes={'a':'new'},prior_plans=[ref])
   self.assertEqual(accepted_plan_hashes(new),{digest(old),digest(new)})
   for key,value in [('task_ids',['b']),('limits',{'calls':11}),('protocol',{'actor':'other'})]:
    bad=deepcopy(new);bad[key]=value
    with self.assertRaisesRegex(ValueError,'scientific'):accepted_plan_hashes(bad)
   path.write_text(json.dumps(dict(old,source='tampered')))
   with self.assertRaisesRegex(ValueError,'bytes'):accepted_plan_hashes(new)
if __name__=='__main__':unittest.main()
