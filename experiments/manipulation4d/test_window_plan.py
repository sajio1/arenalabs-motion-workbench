import unittest
import numpy as np
from window_plan import plan, indices

class WindowPlanTests(unittest.TestCase):
    def test_boundary_lengths_and_coverage(self):
        for n in [1,59,60,95,299,300,301,540,550,600,1917,18000]:
            coverage=np.zeros(n, dtype=int)
            for start, valid in plan(n):
                idx=indices(start,valid)
                self.assertEqual(len(idx),300)
                np.testing.assert_array_equal(idx[:valid],np.arange(start,start+valid))
                coverage[idx[:valid]]+=1
            self.assertTrue(np.all(coverage>0))
            self.assertEqual(start+valid,n)

    def test_shifted_seams_do_not_change_source(self):
        original=np.arange(550*3).reshape(550,3)
        before=original.copy()
        for shift in [None,180,210,240]:
            output=np.zeros_like(original)
            for start, valid in plan(len(original),first_step=shift):
                batch=original[indices(start,valid)]
                output[start:start+valid]=batch[:valid]
            np.testing.assert_array_equal(output,before)
            np.testing.assert_array_equal(original,before)

if __name__=='__main__': unittest.main()
