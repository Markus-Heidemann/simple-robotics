import unittest
from math import pi

# TODO: is there a nicer way?
from context import VelocityMotionModel, Robot
from VelocityMotionModel import VelocityMotionModel, ErrorParamsMovement
from Robot import Pose


class TestVelocityMotionModel(unittest.TestCase):
    def setUp(self):
        self.vmm = VelocityMotionModel()

    def test_calculate_new_pose(self):
        pose = Pose()
        new_pose = self.vmm.calculateNewPose(pose, 1.0, 0.0, 2)

        self.assertEqual(2.0, new_pose.x)
        self.assertEqual(0.0, new_pose.y)
        self.assertEqual(0.0, new_pose.theta)


if __name__ == '__main__':
    unittest.main()
