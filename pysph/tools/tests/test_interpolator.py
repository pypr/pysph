# Author: Prabhu Ramachandran
# Copyright (c) 2014 Prabhu Ramachandran
# License: BSD Style.

# Standard library imports
import unittest

# Local imports
from pysph.tools.interpolator import get_nx_ny_nz


class TestGetNxNyNz(unittest.TestCase):
    def test_should_work_for_1d_data(self):
        # When
        num_points = 100
        bounds = (0.0, 1.5, 0.0, 0.0, 0.0, 0.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [100, 1, 1])

        # When
        num_points = 100
        bounds = (0.0, 0.0, 0.0, 1.123, 0.0, 0.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [1, 100, 1])

        # When
        num_points = 100
        bounds = (0.0, 0.0, 0.0, 0.0, 0.0, 3.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [1, 1, 100])

    def test_should_work_for_2d_data(self):
        # When
        num_points = 100
        bounds = (0.0, 1.0, 0.5, 1.5, 0.0, 0.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [10, 10, 1])

        # When
        num_points = 100
        bounds = (0.0, 0.0, 0, 1, 0.5, 1.5)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [1, 10, 10])

    def test_should_work_for_3d_data(self):
        # When
        num_points = 1000
        bounds = (0.0, 1.0, 0.5, 1.5, -1.0, 0.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [10, 10, 10])

        # When
        num_points = 1000
        bounds = (0.0, 1.0, 0.5, 1.5, -1.0, 2.0)
        dims = get_nx_ny_nz(num_points, bounds)

        # Then
        self.assertListEqual(list(dims), [7, 7, 21])



if __name__ == '__main__':
    unittest.main()
