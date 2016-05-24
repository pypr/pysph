"""Tests for neighbor caching.
"""

import numpy as np
import unittest

from pysph.base.nnps import NeighborCache, LinkedListNNPS
from pysph.base.utils import get_particle_array
from pyzoltan.core.carray import UIntArray

class TestNeighborCache(unittest.TestCase):
    def _make_random_parray(self, name, nx=5):
        x, y, z = np.random.random((3, nx, nx, nx))
        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(z)
        h = np.ones_like(x)*0.2
        return get_particle_array(name=name, x=x, y=y, z=z, h=h)

    def test_neighbors_cached_properly(self):
        # Given
        pa1 = self._make_random_parray('pa1', 5)
        pa2 = self._make_random_parray('pa2', 4)
        particles = [pa1, pa2]
        nnps = LinkedListNNPS(dim=3, particles=particles)

        for dst_index in (0, 1):
            for src_idx in (0, 1):
                # When
                cache = NeighborCache(nnps, dst_index, src_idx)
                cache.update()
                nb_cached = UIntArray()
                nb_direct = UIntArray()

                # Then.
                for i in range(len(particles[dst_index].x)):
                    nnps.get_nearest_particles_no_cache(
                        src_idx, dst_index, i, nb_direct, False
                    )
                    cache.get_neighbors(src_idx, i, nb_cached)
                    nb_e = nb_direct.get_npy_array()
                    nb_c = nb_cached.get_npy_array()
                    self.assertTrue(np.all(nb_e == nb_c))

    def test_empty_neigbors_works_correctly(self):
        # Given
        pa1 = self._make_random_parray('pa1', 5)
        pa2 = self._make_random_parray('pa2', 2)
        pa2.x += 10.0
        particles = [pa1, pa2]

        # When
        nnps = LinkedListNNPS(dim=3, particles=particles)
        # Cache for neighbors of destination 0.
        cache = NeighborCache(nnps, dst_index=0, src_index=1)
        cache.update()

        # Then
        nb_cached = UIntArray()
        nb_direct = UIntArray()
        for i in range(len(particles[0].x)):
            nnps.get_nearest_particles_no_cache(1, 0, i,nb_direct, False)
            # Get neighbors from source 1 on destination 0.
            cache.get_neighbors(src_index=1, d_idx=i, nbrs=nb_cached)
            nb_e = nb_direct.get_npy_array()
            nb_c = nb_cached.get_npy_array()
            self.assertEqual(len(nb_e), 0)
            self.assertTrue(np.all(nb_e == nb_c))

    def test_cache_updates_with_changed_particles(self):
        # Given
        pa1 = self._make_random_parray('pa1', 5)
        particles = [pa1]
        nnps = LinkedListNNPS(dim=3, particles=particles)
        cache = NeighborCache(nnps, dst_index=0, src_index=0)
        cache.update()

        # When
        pa2 = self._make_random_parray('pa2', 2)
        pa1.add_particles(x=pa2.x, y=pa2.y, z=pa2.z)
        nnps.update()
        cache.update()
        nb_cached = UIntArray()
        nb_direct = UIntArray()
        for i in range(len(particles[0].x)):
            nnps.get_nearest_particles_no_cache(0, 0, i, nb_direct, False)
            cache.get_neighbors(0, i, nb_cached)
            nb_e = nb_direct.get_npy_array()
            nb_c = nb_cached.get_npy_array()
            self.assertTrue(np.all(nb_e == nb_c))


if __name__ == '__main__':
    unittest.main()
