import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC=importlib.util.spec_from_file_location('cluster_tasks',Path(__file__).resolve().parents[1]/'scripts/cluster_arm_task_pool.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class ClusterTaskTests(unittest.TestCase):
    def test_multiple_workflows_are_preserved(self):
        primary,tags,evidence=M.workflow('Compare two cameras and add the cheapest to the cart.')
        self.assertEqual(primary,'cart_purchase')
        self.assertIn('comparison',tags)
        self.assertIn('search_filter_sort',tags)
        self.assertTrue(evidence['cart_purchase'])

    def test_book_noun_not_automatically_booking(self):
        self.assertEqual(M.workflow('Find a book about polar exploration.')[0],'lookup_or_unclassified')
        self.assertEqual(M.workflow('Book a hotel room.')[0],'booking_application')

    def test_partition_keeps_every_task_and_stable_labels(self):
        x=np.array([[1.,0.],[.99,.1],[0.,1.],[.1,.99]],dtype=np.float32)
        x/=np.linalg.norm(x,axis=1)[:,None]
        records=[dict(task_id=str(i),source='fixture',site='example.com',embedding_row=i,
                      instruction=f'task {i}',primary_workflow='lookup_or_unclassified',workflow_tags=[]) for i in range(4)]
        a,c=M.partition(x,np.array([5,5,2,2]),records,'test',50)
        b,d=M.partition(x,np.array([2,2,5,5]),records,'test',50)
        self.assertEqual(a,b);self.assertEqual(c,d)
        self.assertEqual(len(a),4);self.assertEqual({r['task_id'] for r in a},{'0','1','2','3'})
        self.assertEqual(sum(r['size'] for r in c),4)
        for cluster in c:
            ids={r['task_id'] for r in a if r['cluster_id']==cluster['cluster_id']}
            self.assertIn(cluster['representative_task_id'],ids)
            self.assertTrue({e['task_id'] for e in cluster['examples']}<=ids)

    def test_singleton_is_retained(self):
        x=np.array([[1.,0.]],dtype=np.float32)
        self.assertEqual(M.fit(x,1,42).tolist(),[0])


if __name__=='__main__':unittest.main()
