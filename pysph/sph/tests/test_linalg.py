from pysph.sph.wc.linalg import (
    augmented_matrix, gj_solve, mat_mult, mat_vec_mult
)
import numpy as np
import unittest


def gj_solve_helper(a, b, n):
    m = np.zeros((n, n+1)).ravel().tolist()
    augmented_matrix(a, b, n, 1, m)
    result = [0.0]*n
    is_singular = gj_solve(m, n, 1, result)
    return is_singular, result


class TestLinalg(unittest.TestCase):

    def _to_array(self, x, shape=None):
        x = np.asarray(x)
        if shape:
            x.shape = shape
        return x

    def test_augmented_matrix(self):
        # Given
        a = np.random.random((3, 3))
        b = np.random.random((3, 2))
        res = np.zeros((3, 5)).ravel().tolist()
        expect = np.zeros((3, 5))
        expect[:, :3] = a
        expect[:, 3:] = b
        # When
        augmented_matrix(a.ravel(), b.ravel(), 3, 2, res)
        res = self._to_array(res, (3, 5))
        # Then
        np.testing.assert_array_almost_equal(res, expect)

    def test_general_matrix(self):
        # Test Gauss Jordan solve.
        """
        This is a general matrix which needs partial pivoting to be
        solved.
        References
        ----------
        http://web.mit.edu/10.001/Web/Course_Notes/GaussElimPivoting.html
        """
        n = 4
        mat = [[0.02, 0.01, 0., 0.], [1., 2., 1., 0.], [0., 1., 2., 1.],
               [0., 0., 100., 200.]]
        b = [0.02, 1., 4., 800.]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_band_matrix(self):
        n = 3
        mat = [[1., -2., 0.], [1., -1., 3.], [2., 5., 0.]]
        b = [-3., 1., 0.5]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_dense_matrix(self):
        n = 3
        mat = [[0.96, 4.6, -3.7], [2.7, 4.3, -0.67], [0.9, 0., -5.]]
        b = [2.4, 3.6, -5.8]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_tridiagonal_matrix(self):
        n = 4
        mat = [[-2., 1., 0., 0.], [1., -2., 1., 0.], [0., 1., -2., 0.],
               [0., 0., 1., -2.]]
        b = [-1., 0., 0., -5.]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_symmetric_matrix(self):
        n = 3
        mat = [[0.96, 4.6, -3.7], [4.6, 4.3, -0.67], [-3.7, -0.67, -5.]]
        b = [2.4, 3.6, -5.8]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_symmetric_positivedefinite_Matrix(self):
        n = 4
        mat = [[1., 1., 4., -1.], [1., 5., 0., -1.], [4., 0., 21., -4.],
               [-1., -1., -4., 10.]]
        b = [2.4, 3.6, -5.8, 0.5]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_inverse(self):
        # Given
        n = 3
        mat = [[1.0, 2.0, 2.5], [2.5, 1.0, 0.0], [0.0, 0.0, 1.0]]
        b = np.identity(3).ravel().tolist()
        A = np.zeros((3, 6)).ravel().tolist()
        augmented_matrix(np.ravel(mat), b, 3, 3, A)
        result = np.zeros((3, 3)).ravel().tolist()

        # When
        sing = gj_solve(A, n, n, result)

        # Then
        mat = np.asarray(mat)
        res = np.asarray(result)
        res.shape = 3, 3
        np.testing.assert_allclose(res, np.linalg.inv(mat))
        self.assertAlmostEqual(sing, 0.0)

    def test_matmult(self):
        # Given
        n = 3
        a = np.random.random((3, 3))
        b = np.random.random((3, 3))
        result = [0.0]*9

        # When
        mat_mult(a.ravel(), b.ravel(), n, result)

        # Then.
        expect = np.dot(a, b)
        result = np.asarray(result)
        result.shape = 3, 3
        np.testing.assert_allclose(result, expect)

    def test_mat_vec_mult(self):
        # Given
        n = 3
        a = np.random.random((3, 3))
        b = np.random.random((3,))
        result = [0.0]*3

        # When
        mat_vec_mult(a.ravel(), b, n, result)

        # Then.
        expect = np.dot(a, b)
        result = np.asarray(result)
        np.testing.assert_allclose(result, expect)

    def test_singular_matrix(self):
        # Given
        n = 3
        mat = [[1., 1., 0.], [1., 1., 0.], [1., 1., 1.]]
        b = [1.0, 1.0, 1.0]
        #
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        self.assertAlmostEqual(sing, 1.0)


if __name__ == '__main__':
    unittest.main()
