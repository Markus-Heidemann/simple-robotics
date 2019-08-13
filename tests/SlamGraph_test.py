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
        motion_cmds = np.array([[1, 0],
                                [0, 0.5*pi],
                                [1, 0],
                                [0, -0.5*pi],
                                [1, 0]])
        self.SlamGraph.initialize(motion_cmds)
        mu_0_t = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0.5*pi],
                           [1, 1, 0.5*pi],
                           [1, 1, 0],
                           [2, 1, 0]])
        assert_allclose(mu_0_t, self.SlamGraph.mu_0_t)


if __name__ == '__main__':
    unittest.main()
