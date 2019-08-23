import unittest
from numpy.testing import assert_allclose
from math import pi
import numpy as np

from context import SlamGraph
from SlamGraph import SlamGraph


class TestSlamGraph(unittest.TestCase):
    def setUp(self):
        self.SlamGraph = SlamGraph()

    def test_linearize_pose_landmark_constraint(self):
        x1 = [1.1, 0.9, 1]
        x2 = [2.2, 1.9]
        z = [1.3, -0.4]
        e_true = np.array([0.135804, 0.014684, 0])
        delta = 1e-6
        epsilon = 1e-4

        e, A, B = self.SlamGraph.linearize_pose_landmark_constraint(x1, x2, z)
        assert_allclose(e, e_true, rtol=epsilon)

        # test for x1
        ANumeric = np.zeros((3,3))
        for d in range(3):
            curX = x1
            curX[d] += delta
            err, _, _ = self.SlamGraph.linearize_pose_landmark_constraint(curX, x2, z)
            curX[d] -= 2 * delta
            err_tmp, _, _ = self.SlamGraph.linearize_pose_landmark_constraint(curX, x2, z)
            err -= err_tmp

            ANumeric[:, d] = err / (2 * delta)

        assert_allclose(ANumeric, A, rtol=epsilon)

        # test for x2
        BNumeric = np.zeros((3,3))
        for d in range(2):
            curX = x2
            curX[d] += delta
            err, _, _ = self.SlamGraph.linearize_pose_landmark_constraint(x1, curX, z)
            curX[d] -= 2 * delta
            err_tmp, _, _ = self.SlamGraph.linearize_pose_landmark_constraint(x1, curX, z)
            err -= err_tmp

            BNumeric[:, d] = err / (2 * delta)

        assert_allclose(BNumeric, B, rtol=epsilon)

    def test_linearize_pose_pose_constraint(self):
        x1 = [1.1, 0.9, 1]
        x2 = [2.2, 1.85, 1.2]
        z  = [0.9, 1.1, 1.05]
        e_true = np.array([-1.06617, -1.18076, -0.85000])
        delta = 1e-6
        epsilon = 1e-4

        # get the analytic Jacobian
        [e, A, B] = self.SlamGraph.linearize_pose_pose_constraint(x1, x2, z)

        assert_allclose(e_true, e, rtol=epsilon)

        # test for x1
        ANumeric = np.zeros((3,3))
        for d in range(3):
            curX = x1
            curX[d] += delta
            err, _, _ = self.SlamGraph.linearize_pose_pose_constraint(curX, x2, z)
            curX[d] -= 2 * delta
            err_tmp, _, _ = self.SlamGraph.linearize_pose_pose_constraint(curX, x2, z)
            err -= err_tmp

            ANumeric[:, d] = err / (2 * delta)

        assert_allclose(ANumeric, A, rtol=epsilon)

        # test for x2
        BNumeric = np.zeros((3,3))
        for d in range(3):
            curX = x2
            curX[d] += delta
            err, _, _ = self.SlamGraph.linearize_pose_pose_constraint(x1, curX, z)
            curX[d] -= 2 * delta
            err_tmp, _, _ = self.SlamGraph.linearize_pose_pose_constraint(x1, curX, z)
            err -= err_tmp

            BNumeric[:, d] = err / (2 * delta)

        assert_allclose(BNumeric, B, rtol=epsilon)


if __name__ == '__main__':
    unittest.main()
