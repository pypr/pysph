import numpy as np
import unittest
from pysph.sph.wc.linalg import (
    augmented_matrix, gj_solve, mat_mult, mat_vec_mult, determinant_2d,
    determinant_3d, determinant_4d, replace_vector_in_matrix, linear_solver_2d,
    linear_solver_3d, linear_solver_4d
)


def gj_solve_helper(a, b, n):
    m = np.zeros((n, n+1)).ravel().tolist()
    augmented_matrix(a, b, n, 1, n, m)
    result = [0.0]*n
    is_singular = gj_solve(m, n, 1, result)
    return is_singular, result


def determinant_helper(A, dim):
    det = 0.0
    if dim == 2:
        det = determinant_2d(A)
    elif dim == 3:
        det = determinant_3d(A)
    elif dim == 4:
        det = determinant_4d(A)
    return det


def linear_solver_helper(A, b, dim):
    result = [0.0]*len(b)
    ret = 0.0
    if dim == 2:
        ret = linear_solver_2d(A, b, result)
    elif dim == 3:
        ret = linear_solver_3d(A, b, result)
    elif dim == 4:
        ret = linear_solver_4d(A, b, result)
    return ret, result


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
        augmented_matrix(a.ravel(), b.ravel(), 3, 2, 3, res)
        res = self._to_array(res, (3, 5))
        # Then
        np.testing.assert_array_almost_equal(res, expect)

    def test_augmented_matrix_with_lower_dimension(self):
        # Given
        a = np.random.random((3, 3))
        b = np.random.random((3, 2))
        res = np.zeros((3, 5)).ravel().tolist()
        expect = np.zeros((2, 4))
        expect[:, :2] = a[:2, :2]
        expect[:, 2:] = b[:2, :]
        expect.resize(3, 5)
        # When
        augmented_matrix(a.ravel(), b.ravel(), 2, 2, 3, res)
        res = self._to_array(res, (3, 5))
        # Then
        np.testing.assert_array_almost_equal(res, expect)

    def test_augmented_matrix_with_gjsolve_with_lower_dimension(self):
        # Given
        nmax = 3
        mat = np.array([[7., 4., 2.], [8., 9., 4.], [1., 4., 10.]])
        b = np.array([5., 4., 2.])
        expect = np.linalg.solve(mat[:2, :2], b[:2])
        augmat = np.zeros((3, 4)).ravel().tolist()
        res = np.zeros(2).ravel().tolist()
        # When
        augmented_matrix(mat.ravel(), b.ravel(), 2, 1, nmax, augmat)
        gj_solve(augmat, 2, 1, res)
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
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result)))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_band_matrix(self):
        n = 3
        mat = [[1., -2., 0.], [1., -1., 3.], [2., 5., 0.]]
        b = [-3., 1., 0.5]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result)))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_dense_matrix(self):
        n = 3
        mat = [[0.96, 4.6, -3.7], [2.7, 4.3, -0.67], [0.9, 0., -5.]]
        b = [2.4, 3.6, -5.8]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result)))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_tridiagonal_matrix(self):
        n = 4
        mat = [[-2., 1., 0., 0.], [1., -2., 1., 0.], [0., 1., -2., 0.],
               [0., 0., 1., -2.]]
        b = [-1., 0., 0., -5.]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result)))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_symmetric_matrix(self):
        n = 3
        mat = [[0.96, 4.6, -3.7], [4.6, 4.3, -0.67], [-3.7, -0.67, -5.]]
        b = [2.4, 3.6, -5.8]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result)))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_symmetric_positivedefinite_Matrix(self):
        n = 4
        mat = [[1., 1., 4., -1.], [1., 5., 0., -1.], [4., 0., 21., -4.],
               [-1., -1., -4., 10.]]
        b = [2.4, 3.6, -5.8, 0.5]
        sing, result = gj_solve_helper(np.ravel(mat), b, n)
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result)))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))
        self.assertAlmostEqual(sing, 0.0)

    def test_inverse(self):
        # Given
        n = 3
        mat = [[1.0, 2.0, 2.5], [2.5, 1.0, 0.0], [0.0, 0.0, 1.0]]
        b = np.identity(3).ravel().tolist()
        A = np.zeros((3, 6)).ravel().tolist()
        augmented_matrix(np.ravel(mat), b, 3, 3, 3, A)
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

    def test_determinant(self):
        # Given
        mat2 = np.random.rand(2, 2)
        mat3 = np.random.rand(3, 3)
        mat4 = np.random.rand(4, 4)
        # When
        det2 = determinant_helper(np.ravel(mat2), 2)
        det3 = determinant_helper(np.ravel(mat3), 3)
        det4 = determinant_helper(np.ravel(mat4), 4)
        # Then
        self.assertAlmostEqual(det2, np.linalg.det(mat2))
        self.assertAlmostEqual(det3, np.linalg.det(mat3))
        self.assertAlmostEqual(det4, np.linalg.det(mat4))

    def test_vector_swaping_in_matrix(self):
        # Given
        mat4 = np.random.rand(4, 4)
        b = np.random.rand(4)
        # When
        result = [0.0]*16
        replace_vector_in_matrix(np.ravel(mat4), b, 2, 4, result)
        # Then
        mat4 = np.array(mat4)
        mat4[:, 2] = b
        np.testing.assert_allclose(result, np.ravel(mat4))

    def test_vector_swaping_invalid_column_in_matrix(self):
        # Given
        mat4 = np.random.rand(4, 4)
        b = np.random.rand(4)
        # When
        result = [0.0]*16
        ret = replace_vector_in_matrix(np.ravel(mat4), b, 4, 4, result)
        # Then
        self.assertAlmostEqual(ret, -1.0)

    def test_2d_linear_solver(self):
        # Given
        mat = [[0.96, 4.6], [2.7, 4.3]]
        b = [2.4, 3.6]
        # When
        ret, result = linear_solver_helper(np.ravel(mat), b, 2)
        # Then
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result)))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def test_2d_linear_solver_with_zero_in_diagonal(self):
        # Given
        mat = [[0.96, 4.6], [2.7, 0.0]]
        b = [2.4, 3.6]
        # When
        ret, result = linear_solver_helper(np.ravel(mat), b, 2)
        # Then
        self.assertAlmostEqual(ret, 0.0)

    def test_2d_linear_solver_with_4d_matrix(self):
        # Given
        mat = [[0.96, 4.6], [2.7, 4.3]]
        mat_new = np.zeros(16)
        mat_new[0:4] = np.ravel(mat)
        b = [2.4, 3.6]
        b_new = np.zeros(4)
        b_new[:2] = b
        # When
        ret, result = linear_solver_helper(mat_new, b_new, 2)
        # Then
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result[:2])))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def test_3d_linear_solver(self):
        # Given
        mat = [[0.96, 4.6, -3.7], [2.7, 4.3, -0.67], [0.9, 0., -5.]]
        b = [2.4, 3.6, -5.8]
        # When
        ret, result = linear_solver_helper(np.ravel(mat), b, 3)
        # Then
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result)))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def test_3d_linear_solver_with_zero_in_diagonal(self):
        # Given
        mat = [[0.96, 4.6, -3.7], [2.7, 4.3, -0.67], [0.9, 0., 0.0]]
        b = [2.4, 3.6, -5.8]
        # When
        ret, result = linear_solver_helper(np.ravel(mat), b, 3)
        # Then
        self.assertAlmostEqual(ret, 0.0)

    def test_3d_linear_solver_with_4d_matrix(self):
        # Given
        mat = [[0.96, 4.6, -3.7], [2.7, 4.3, -0.67], [0.9, 0., -5.]]
        mat_new = np.zeros(16)
        mat_new[0:9] = np.ravel(mat)
        b = [2.4, 3.6, -5.8]
        b_new = np.zeros(4)
        b_new[:3] = b
        # When
        ret, result = linear_solver_helper(mat_new, b_new, 3)
        # Then
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result[:3])))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def test_4d_linear_solver(self):
        # Given
        mat = [[0.96, 4.6, -3.7, 1.2], [2.7, 4.3, -0.67, 1.3],
               [0.9, 0., -5., 2.4], [1.4, 2.6, 3.1, 7.8]]
        b = [2.4, 3.6, -5.8, 2.0]
        # When
        ret, result = linear_solver_helper(np.ravel(mat), b, 4)
        # Then
        mat = np.array(mat)
        new_b = np.dot(mat, np.transpose(np.array(result)))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def test_4d_linear_solver_with_zero_in_diagonal(self):
        # Given
        mat = [[0.96, 4.6, -3.7, 1.2], [2.7, 4.3, -0.67, 1.3],
               [0.9, 0., -5., 2.4], [1.4, 2.6, 3.1, 0.0]]
        b = [2.4, 3.6, -5.8, 2.0]
        # When
        ret, result = linear_solver_helper(np.ravel(mat), b, 4)
        # Then
        self.assertAlmostEqual(ret, 0.0)


if __name__ == '__main__':
    unittest.main()
