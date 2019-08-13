import unittest
from math import pi

# TODO: is there a nicer way?
from context import Robot
from Robot import Robot, Pose


class TestRobot(unittest.TestCase):
    def setUp(self):
        self.robot = Robot()

    def test_move_straight(self):
        self.robot.move(0.0, 0.5*pi)
        self.robot.move(1.0, 0.0)

        self.assertEqual(0.0, self.robot.pose.x)
        self.assertEqual(1.0, self.robot.pose.y)
        self.assertEqual(0.5*pi, self.robot.pose.theta)


if __name__ == '__main__':
    unittest.main()
