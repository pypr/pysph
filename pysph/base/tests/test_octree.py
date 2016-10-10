"""Unittests for Octree

You can run the tests like so:

    $ nosetests -v test_octree.py
"""
import numpy
from numpy import random

# PySPH imports
from pysph.base.point import IntPoint, Point
from pysph.base.utils import get_particle_array
from pysph.base import nnps
from pysph.base.octree import Octree

# Carrays from PyZoltan
from pyzoltan.core.carray import UIntArray, IntArray

# Python testing framework
import unittest

class SimpleOctreeTestCase(unittest.TestCase):
    """Simple test case for Octree
    """
    def setUp(self):
        N = 100
        x, y, z = numpy.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        h = numpy.ones_like(x)

        self.pa = get_particle_array(x=x, y=y, z=z, h=h)
        self.tree = Octree(10, 1)
        self.tree.build_tree(self.pa)

    def test_single_level_tree(self):
        tree = Octree(self.pa.get_number_of_particles() + 1, 1)
        tree.build_tree(self.pa)
        self.assertTrue(tree.depth == 1)

    def test_levels_in_tree(self):
        root = self.tree.get_root()
        self.assertTrue(root.level == 0)
        current_level = 1
        children = root.get_children()
        while len(children) != 0:
            for child in children:
                self.assertTrue(child.level == current_level)
            current_level += 1
            children = child.get_children()

    def test_parent_for_node(self):
        root = self.tree.get_root()
        self.assertTrue(root.get_parent() == None)
        children = root.get_children()
        parent = root
        while len(children) != 0:
            for child in children:
                self.assertTrue(child.get_parent() == parent)
            parent = child
            children = child.get_children()


class TestOctreeForFloatingPointError(SimpleOctreeTestCase):
    """Test Octree for floating point error
    """
    def setUp(self):
        N = 100
        x, y, z = numpy.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        h = numpy.ones_like(x)

        self.pa = get_particle_array(x=x, y=y, z=z, h=h)

        x1 = numpy.array([1e-20])
        y1 = numpy.array([1e-20])
        z1 = numpy.array([1e-20])

        self.pa.add_particles(x=x1, y=y1, z=z1)
        self.tree = Octree(10, 1)
        self.tree.build_tree(self.pa)

