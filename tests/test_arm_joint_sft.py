"""CPU-only checks for the exposure contract; GPU math gates live in the trainer."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_arm_joint_sft import lr_factor, training_order


class ExposureContract(unittest.TestCase):
    def test_no_replacement_and_matched_source_order(self):
        rows=[{'source':'C2','id':i} for i in range(3464)]
        rows += [{'source':'Piotr','id':i} for i in range(3464,5540)]
        order=training_order(rows)
        self.assertEqual(sorted(order),list(range(5540)))
        self.assertEqual(order,training_order(rows))
        self.assertNotEqual(order,training_order(rows,43))
        # Prefixes remain close to the natural source ratio, without oversampling.
        for size in [32,512,1408,2784,4192,5540]:
            counts=Counter(rows[i]['source'] for i in order[:size])
            self.assertLessEqual(abs(counts['C2']-size*3464/5540),1)

    def test_partial_batch_is_four_states_without_repeated_padding(self):
        rows=[{'source':'C2'}]*5540
        order=training_order(rows)
        batches=[order[i:i+32] for i in range(0,len(order),32)]
        self.assertEqual(len(batches),174)
        self.assertEqual(len(batches[-1]),4)
        self.assertEqual(len(set(sum(batches,[]))),5540)
        self.assertEqual(sum([1/len(batches[-1])]*len(batches[-1])),1)

    def test_resume_keeps_identical_remaining_exposure(self):
        rows=[{'source':'C2'}]*3464+[{'source':'Piotr'}]*2076
        uninterrupted=training_order(rows)
        for update in [44,50,87,131,173]:
            cursor=update*32
            self.assertEqual(uninterrupted[cursor:],training_order(rows)[cursor:])

    def test_schedule_uses_states_and_reaches_half_peak(self):
        self.assertAlmostEqual(lr_factor(32,5540),1/16)
        self.assertEqual(lr_factor(512,5540),1)
        self.assertAlmostEqual(lr_factor(5540,5540),.5)
        tail=[lr_factor(n,5540) for n in [512,1408,2784,4192,5540]]
        self.assertEqual(tail,sorted(tail,reverse=True))
        with self.assertRaises(ValueError): lr_factor(5541,5540)


if __name__=='__main__': unittest.main()
