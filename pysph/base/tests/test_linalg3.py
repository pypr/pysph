import unittest

import numpy as np

from pysph.base import linalg3

class TestLinalg3(unittest.TestCase):

    def setUp(self):
        self.N = 10

    def assertMatricesEqual(self, result, expected, matrix, info, tol=1e-14):
        diff = result - expected
        msg = "Error for {info} matrix\n{matrix}\n"\
                "result:\n{result}\n"\
                "expected:\n{expected}\n"\
                "difference: {diff}".format(
            info=info, matrix=matrix,
            diff=diff, result=result, expected=expected
        )
        self.assertTrue(np.max(np.abs(diff)) < tol, msg)

    def _get_test_matrix(self):
        a = np.random.random((3,3))
        a += a.T
        return a

    def _get_difficult_matrix(self):
        a = np.identity(3, dtype=float)
        for p in range(3):
            a[p==0][2-(p==2)] = (p==0)*1e-2
            a[2-(p==2)][p==0] = a[p==0][2-(p==2)]

        return a

    def _get_test_matrices(self):
        nasty = [
            [1.823886368900899e-169, -1.2724997010965309e-169, 0.0],
            [-1.2724997010965309e-169, -3.647772737801798e-169, 0.0],
            [0.0, 0.0, 0.0]
        ]

        data = [
            (np.zeros((3,3), dtype=float), 'zero'),
            (np.identity(3, dtype=float), 'identity'),
            (np.diag((1., 2., 3.)), 'diagonal'),
            (np.diag((2., 2., 1.)), 'diagonal repeated eigenvalues'),
            (np.ones((3,3), dtype=float), 'ones'),
            (self._get_test_matrix(), 'random'),
            (self._get_difficult_matrix(), 'difficult'),
            (np.asarray(nasty), 'nasty')

        ]
        return data

    def test_determinant(self):
        for i in range(self.N):
            self._check_determinant()

    def _check_determinant(self):
        # Given/When
        a = self._get_test_matrix()

        # Then
        self.assertAlmostEqual(
            linalg3.py_det(a), np.linalg.det(a), places=14
        )

    def test_eigen_values(self):
        # Given
        data = self._get_test_matrices()

        for matrix, info in data:
            # When
            result = linalg3.py_get_eigenvalues(matrix)
            result.sort()
            # Then
            expected = np.linalg.eigvals(matrix)
            expected.sort()
            self.assertMatricesEqual(result, expected, matrix, info)

    def test_eigen_decompose_eispack(self):
        # Given
        data = self._get_test_matrices()

        for matrix, info in data:
            # When
            val, vec = linalg3.py_eigen_decompose_eispack(matrix)

            # Then
            check = np.dot(np.asarray(vec), np.dot(np.diag(val), np.asarray(vec).T))
            self.assertMatricesEqual(check, matrix, vec, info)

    def test_get_eigenvalvec(self):
        # Given
        data = self._get_test_matrices()

        for matrix, info in data:
            # When
            val, vec = linalg3.py_get_eigenvalvec(matrix)

            # Then
            check = np.dot(np.asarray(vec), np.dot(np.diag(val), np.asarray(vec).T))
            self.assertMatricesEqual(check, matrix, vec, info)

    ##################################################################################
    # Transformation related tests.

    def test_transform_function(self):
        for i in range(self.N):
            self._check_transform_function()

    def _check_transform_function(self):
        # Given 
        a = np.random.random((3,3))
        p = np.random.random((3,3))

        # When
        res = linalg3.py_transform(a, p)

        # Then
        expected = np.dot(p.T, np.dot(a, p))
        self.assertMatricesEqual(res, expected, (a,p), 'transform')

    def test_transform_diag_function(self):
        for i in range(self.N):
            self._check_transform_diag_function()

    def _check_transform_diag_function(self):
        # Given 
        a = np.random.random(3)
        p = np.random.random((3,3))

        # When
        res = linalg3.py_transform_diag(a, p)

        # Then
        expected = np.dot(p.T, np.dot(np.diag(a), p))
        self.assertMatricesEqual(res, expected, (a,p), 'transform')

    def test_transform_diag_inv_function(self):
        for i in range(self.N):
            self._check_transform_diag_inv_function()

    def _check_transform_diag_inv_function(self):
        # Given 
        a = np.random.random(3)
        p = np.random.random((3,3))

        # When
        res = linalg3.py_transform_diag_inv(a, p)

        # Then
        expected = np.dot(p, np.dot(np.diag(a), p.T))
        self.assertMatricesEqual(res, expected, (a,p), 'transform')

if __name__ == '__main__':
    unittest.main()

