import numpy as np
from src.tiler import induced_angle
import unittest


class TestInducedAngles(unittest.TestCase):
    def test_angles(self):
        v1 = np.array(
            [
                [1, 0],
                [1, 0],
                [1, 0.]
            ]
        )

        v2 = np.array(
            [
                [1, 1],
                [-1, 1],
                [0, 1.]
            ]
        )

        angles = induced_angle(v1, v2)
        test_angles = np.array([45, 135, 90])
        self.assertTrue((angles == test_angles).all())

        rev_angles = induced_angle(v2, v1)
        test_rev_angles = np.array([315, 225, 270])
        self.assertTrue((rev_angles == test_rev_angles).all())


if __name__ == "__main__":
    unittest.main()
