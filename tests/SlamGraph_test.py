import unittest
from numpy.testing import assert_allclose
from math import pi
import numpy as np

from context import SlamGraph
from SlamGraph import SlamGraph


class TestSlamGraph(unittest.TestCase):
    def setUp(self):
        self.SlamGraph = SlamGraph()

    def test_inititalize(self):
        # [v, theta]
        motion_cmds = np.array([[1, 0, 1],
                                [0, 0.5*pi, 1],
                                [1, 0, 1],
                                [0, -0.5*pi, 1],
                                [1, 0, 1]])
        self.SlamGraph.initialize(motion_cmds)
        mu_0_t = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0.5*pi],
                           [1, 1, 0.5*pi],
                           [1, 1, 0],
                           [2, 1, 0]])
        assert_allclose(mu_0_t, self.SlamGraph.mu_0_t)

    def test_faulty_initialization(self):
        # [v, theta]
        motion_cmds = np.array([[1, 0]])
        with self.assertRaises(AssertionError):
            self.SlamGraph.initialize(motion_cmds)

    def test_linearize_pose_landmark_constraint(self):
        x1 = [1.1, 0.9, 1]
        x2 = [2.2, 1.9]
        z = [1.3, -0.4]
        e_true = np.array([0.135804, 0.014684])

        e, A, B = self.SlamGraph.linearize_pose_landmark_constraint(x1, x2, z)
        assert_allclose(e, e_true, rtol=1e-4)

        delta = 1e-6

        # test for x1
        ANumeric = np.zeros((2,3))
        for d in range(3):
            curX = x1
            curX[d] += delta
            err, _, _ = self.SlamGraph.linearize_pose_landmark_constraint(curX, x2, z)
            curX[d] -= 2 * delta
            err_tmp, _, _ = self.SlamGraph.linearize_pose_landmark_constraint(curX, x2, z)
            err -= err_tmp

            ANumeric[:, d] = err / (2 * delta)

        assert_allclose(ANumeric, A, rtol=1e-4)

        # test for x2
        BNumeric = np.zeros((2,2))
        for d in range(2):
            curX = x2
            curX[d] += delta
            err, _, _ = self.SlamGraph.linearize_pose_landmark_constraint(x1, curX, z)
            curX[d] -= 2 * delta
            err_tmp, _, _ = self.SlamGraph.linearize_pose_landmark_constraint(x1, curX, z)
            err -= err_tmp

            BNumeric[:, d] = err / (2 * delta)

        assert_allclose(BNumeric, B, rtol=1e-4)

if __name__ == '__main__':
    unittest.main()
