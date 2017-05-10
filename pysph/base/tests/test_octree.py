"""Unittests for Octree

You can run the tests like so:

    $ pytest -v run_tests.py
"""
import numpy as np

# PySPH imports
from pysph.base.utils import get_particle_array
from pysph.base.octree import Octree, CompressedOctree

# Python testing framework
import unittest
from pytest import importorskip

def test_single_level_octree():
    N = 50
    x, y, z = np.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    h = np.ones_like(x)

    pa = get_particle_array(x=x, y=y, z=z, h=h)

    # For maximum leaf particles greater that total number
    # of particles
    tree = Octree(pa.get_number_of_particles() + 1)
    tree.build_tree(pa)
    # Test that depth of the tree is 1
    assert tree.depth == 1

def test_compressed_octree_has_lesser_depth_than_octree():
    N = 50
    x, y, z = np.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    h = np.ones_like(x)

    pa = get_particle_array(x=x, y=y, z=z, h=h)

    x1 = np.array([20])
    y1 = np.array([20])
    z1 = np.array([20])

    # For a dataset where one particle is far away from a
    # cluster of particles
    pa.add_particles(x=x1, y=y1, z=z1)
    tree = Octree(10)
    comp_tree = CompressedOctree(10)

    depth_tree = tree.build_tree(pa)
    depth_comp_tree = comp_tree.build_tree(pa)

    # Test that the depth of compressed octree for the same
    # leaf_max_particles is lesser than that of octree
    assert depth_comp_tree < depth_tree

def test_single_level_compressed_octree():
    N = 50
    x, y, z = np.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    h = np.ones_like(x)

    pa = get_particle_array(x=x, y=y, z=z, h=h)

    # For maximum leaf particles greater that total number
    # of particles
    tree = CompressedOctree(pa.get_number_of_particles() + 1)
    tree.build_tree(pa)
    # Test that depth of the tree is 1
    assert tree.depth == 1


class SimpleOctreeTestCase(unittest.TestCase):
    """Simple test case for Octree
    """
    def setUp(self):
        N = 50
        x, y, z = np.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
        self.x = x.ravel()
        self.y = y.ravel()
        self.z = z.ravel()
        self.h = np.ones_like(self.x)

        self.tree = Octree(10)

    def test_levels_in_tree(self):
        pa = get_particle_array(x=self.x, y=self.y, z=self.z, h=self.h)
        self.tree.build_tree(pa)
        root = self.tree.get_root()

        def _check_levels(node, level):
            # Test that levels for nodes are correctly set
            # starting from 0 at root
            self.assertTrue(node.level == level)
            children = node.get_children()
            for child in children:
                if child == None:
                    continue
                _check_levels(child, level + 1)

        _check_levels(root, 0)
        self.tree.delete_tree()

    def test_parent_for_node(self):
        pa = get_particle_array(x=self.x, y=self.y, z=self.z, h=self.h)
        self.tree.build_tree(pa)
        root = self.tree.get_root()
        # Test that parent of root is 'None'
        self.assertTrue(root.get_parent() == None)

        def _check_parent(node):
            children = node.get_children()
            for child in children:
                if child == None:
                    continue
                # Test that the parent is set correctly for all nodes
                self.assertTrue(child.get_parent() == node)
                _check_parent(child)

        _check_parent(root)
        self.tree.delete_tree()

    def test_sum_of_indices_lengths_equals_total_number_of_particles(self):
        pa = get_particle_array(x=self.x, y=self.y, z=self.z, h=self.h)
        self.tree.build_tree(pa)
        root = self.tree.get_root()
        sum_indices = [0]

        def _calculate_sum(node, sum_indices):
            indices = node.get_indices(self.tree)
            sum_indices[0] += indices.length
            children = node.get_children()
            for child in children:
                if child == None:
                    continue
                _calculate_sum(child, sum_indices)

        _calculate_sum(root, sum_indices)
        # Test that sum of lengths of all indices is equal to total
        # number of particles
        self.assertTrue(pa.get_number_of_particles() == sum_indices[0])
        self.tree.delete_tree()

    def test_plot_root(self):
        pa = get_particle_array(x=self.x, y=self.y, z=self.z, h=self.h)
        self.tree.build_tree(pa)
        root = self.tree.get_root()
        mpl = importorskip("matplotlib")
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)

        root.plot(ax)

        for line in ax.lines:
            xs = line.get_xdata()
            ys = line.get_ydata()
            line_length = (xs[0] - xs[1])**2 + (ys[0] - ys[1])**2
            # Test that the lengths of sides 2D projection of the node
            # is equal to the length of the side of the node
            self.assertTrue(line_length == root.length**2 or \
                    line_length == 0)

        self.tree.delete_tree()

class TestOctreeFor2DDataset(SimpleOctreeTestCase):
    """Test Octree for 2D dataset
    """
    def setUp(self):
        N = 500
        x, y = np.mgrid[0:1:N*1j, 0:1:N*1j]
        self.x = x.ravel()
        self.y = y.ravel()
        self.z = np.zeros_like(self.x)
        self.h = np.ones_like(self.x)

        self.tree = Octree(10)

class TestOctreeFor1DDataset(SimpleOctreeTestCase):
    """Test Octree for 1D dataset
    """
    def setUp(self):
        N = 1e5
        self.x = np.linspace(0, 1, num=N)
        self.y = np.zeros_like(self.x)
        self.z = np.zeros_like(self.x)
        self.h = np.ones_like(self.x)

        self.tree = Octree(100)

class TestOctreeForFloatingPointError(SimpleOctreeTestCase):
    """Test Octree for floating point error
    """
    def setUp(self):
        N = 50
        x, y, z = np.mgrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]
        self.x = x.ravel()
        self.y = y.ravel()
        self.z = z.ravel()

        x1 = np.array([-1e-20])
        y1 = np.array([1e-20])
        z1 = np.array([1e-20])

        self.x = np.concatenate([self.x, x1])
        self.y = np.concatenate([self.y, y1])
        self.z = np.concatenate([self.z, z1])

        self.h = np.ones_like(self.x)
        self.tree = Octree(10)

class SimpleCompressedOctreeTestCase(SimpleOctreeTestCase):
    """Simple test case for Compressed Octree
    """
    def setUp(self):
        N = 50
        x, y, z = np.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
        self.x = x.ravel()
        self.y = y.ravel()
        self.z = z.ravel()
        self.h = np.ones_like(self.x)

        self.tree = CompressedOctree(10)

class TestCompressedOctreeFor1DDataset(SimpleOctreeTestCase):
    """Test Octree for 1D dataset
    """
    def setUp(self):
        N = 1e5
        self.x = np.linspace(0, 1, num=N)
        self.y = np.zeros_like(self.x)
        self.z = np.zeros_like(self.x)
        self.h = np.ones_like(self.x)

        self.tree = CompressedOctree(100)

class TestCompressedOctreeFor2DDataset(SimpleOctreeTestCase):
    """Test Octree for 2D dataset
    """
    def setUp(self):
        N = 500
        x, y = np.mgrid[0:1:N*1j, 0:1:N*1j]
        self.x = x.ravel()
        self.y = y.ravel()
        self.z = np.zeros_like(self.x)
        self.h = np.ones_like(self.x)

        self.tree = CompressedOctree(10)

class TestCompressedOctreeForFloatingPointError(SimpleOctreeTestCase):
    """Test Octree for floating point error
    """
    def setUp(self):
        N = 50
        x, y, z = np.mgrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]
        self.x = x.ravel()
        self.y = y.ravel()
        self.z = z.ravel()

        x1 = np.array([-1e-20])
        y1 = np.array([1e-20])
        z1 = np.array([1e-20])

        self.x = np.concatenate([self.x, x1])
        self.y = np.concatenate([self.y, y1])
        self.z = np.concatenate([self.z, z1])

        self.h = np.ones_like(self.x)

        self.tree = CompressedOctree(10)

if __name__ == '__main__':
    unittest.main()
