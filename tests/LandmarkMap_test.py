import unittest

from context import LandmarkMap
from LandmarkMap import LandmarkMap


class TestLandmarkMap(unittest.TestCase):
    def setUp(self):
        self.lm_map = LandmarkMap()

    def test_add_landmarks(self):
        x_list = [1, 2, 3, 4, 5]
        y_list = [1, 2, 3, 4, 5]

        self.lm_map.addLandmarks(x_list, y_list)

        self.assertEqual(self.lm_map.x, x_list)
        self.assertEqual(self.lm_map.y, y_list)

    def test_invalid_landmarks_being_rejected(self):
        x_list = [1, 2, 3, 4, 5]
        y_list = [1, 2, 3, 4]

        with self.assertRaises(AssertionError):
            self.lm_map.addLandmarks(x_list, y_list)


if __name__ == '__main__':
    unittest.main()
