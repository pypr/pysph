from pysph.sph.wc.density_correction import gj_Solve
import numpy as np
import unittest


class TestGjSolve(unittest.TestCase):

    def testGeneralMatrix(self):
        n = 4
        mat = [[0.02, 0.01, 0., 0.], [1., 2., 1., 0.], [0., 1., 2., 1.],
               [0., 0., 100., 200.]]
        b = [0.02, 1., 4., 800.]
        result = [0.0] * n
        gj_Solve(np.ravel(mat), b, n, result)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def testBandMatrix(self):
        n = 3
        mat = [[1., -2., 0.], [1., -1., 3.], [2., 5., 0.]]
        b = [-3., 1., 0.5]
        result = [0.0] * n
        gj_Solve(np.ravel(mat), b, n, result)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def testDenseMatrix(self):
        n = 3
        mat = [[0.96, 4.6, -3.7], [2.7, 4.3, -0.67], [0.9, 0., -5.]]
        b = [2.4, 3.6, -5.8]
        result = [0.0] * n
        gj_Solve(np.ravel(mat), b, n, result)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def testTriDiagonalMatrix(self):
        n = 4
        mat = [[-2., 1., 0., 0.], [1., -2., 1., 0.], [0., 1., -2., 0.],
               [0., 0., 1., -2.]]
        b = [-1., 0., 0., -5.]
        result = [0.0] * n
        gj_Solve(np.ravel(mat), b, n, result)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def testSymmetricMatrix(self):
        n = 3
        mat = [[0.96, 4.6, -3.7], [4.6, 4.3, -0.67], [-3.7, -0.67, -5.]]
        b = [2.4, 3.6, -5.8]
        result = [0.0] * n
        gj_Solve(np.ravel(mat), b, n, result)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))

    def testSymmetricPositiveDefiniteMatrix(self):
        n = 4
        mat = [[1., 1., 4., -1.], [1., 5., 0., -1.], [4., 0., 21., -4.],
               [-1., -1., -4., 10.]]
        b = [2.4, 3.6, -5.8, 0.5]
        result = [0.0] * n
        gj_Solve(np.ravel(mat), b, n, result)
        mat = np.matrix(mat)
        new_b = mat * np.transpose(np.matrix(result))
        new_b = np.ravel(np.array(new_b))
        assert np.allclose(new_b, np.array(b))


if __name__ == '__main__':
    unittest.main()
