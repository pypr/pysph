import pytest
import numpy as np
import setuptools  # noqa: F401

pytest.importorskip('warp')
pytest.importorskip('pysph.base.particle_array')

from compyle.config import get_config  # noqa: E402
from pysph.base.utils import get_particle_array  # noqa: E402
from pysph.base.warp_device_helper import WarpDeviceHelper  # noqa: E402


class TestWarpDeviceHelper(object):
    def setup_method(self):
        self.pa = get_particle_array(name='f', x=[0.0, 1.0], m=1.0, rho=2.0)

    def test_simple(self):
        pa = self.pa
        h = WarpDeviceHelper(pa)
        pa.set_device_helper(h)

        assert np.allclose(pa.x, h.x.get())
        assert np.allclose(pa.y, h.y.get())
        assert np.allclose(pa.m, h.m.get())
        assert np.allclose(pa.rho, h.rho.get())
        assert np.allclose(pa.tag, h.tag.get())

    def test_push_and_pull_selected_properties(self):
        pa = self.pa
        h = WarpDeviceHelper(pa)
        pa.set_device_helper(h)

        pa.x[:] = [2.0, 3.0]
        pa.rho[0] = 1.0
        pa.tag[:] = 1
        h.push('x', 'rho', 'tag')

        assert np.allclose(pa.x, h.x.get())
        assert np.allclose(pa.rho, h.rho.get())
        assert np.allclose(pa.tag, h.tag.get())

        h.x.set(np.array([4.0, 5.0], h.x.dtype))
        h.rho[1] = 7.0
        h.tag[:] = np.array([0, 1], h.tag.dtype)
        h.pull('x', 'rho', 'tag')

        assert np.allclose(pa.x, [4.0, 5.0])
        assert np.allclose(pa.rho, [1.0, 7.0])
        assert np.allclose(pa.tag, [0, 1])

    def test_push_and_pull_all_properties(self):
        pa = self.pa
        h = WarpDeviceHelper(pa)
        pa.set_device_helper(h)

        pa.x[:] = [2.0, 3.0]
        pa.y[:] = [4.0, 5.0]
        pa.rho[:] = [6.0, 7.0]
        pa.tag[:] = [1, 0]
        h.push()

        assert np.allclose(h.x.get(), [2.0, 3.0])
        assert np.allclose(h.y.get(), [4.0, 5.0])
        assert np.allclose(h.rho.get(), [6.0, 7.0])
        assert np.allclose(h.tag.get(), [1, 0])

        h.x[:] = 8.0
        h.y[:] = 9.0
        h.rho[:] = 10.0
        h.tag[:] = np.array([0, 1], h.tag.dtype)
        h.pull()

        assert np.allclose(pa.x, [8.0, 8.0])
        assert np.allclose(pa.y, [9.0, 9.0])
        assert np.allclose(pa.rho, [10.0, 10.0])
        assert np.allclose(pa.tag, [0, 1])

    def test_float_precision_follows_config(self):
        cfg = get_config()
        old_use_double = cfg.use_double
        try:
            cfg.use_double = False
            pa = get_particle_array(name='f', x=[0.0, 1.0],
                                    backend='warp')

            assert pa.gpu.x.dtype == np.dtype(np.float32)
        finally:
            cfg.use_double = old_use_double

    def test_align_particles(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                y=[10.0, 11.0, 12.0, 13.0])
        pa.add_property('force', data=[0.0, 0.1, 0.2,
                                       1.0, 1.1, 1.2,
                                       2.0, 2.1, 2.2,
                                       3.0, 3.1, 3.2],
                        stride=3)
        pa.tag[:] = [1, 0, 2, 0]

        h = WarpDeviceHelper(pa)
        pa.set_device_helper(h)
        h.align_particles()
        h.pull()

        assert pa.get_number_of_particles(real=True) == 2
        assert np.allclose(pa.x, [1.0, 3.0])
        assert np.allclose(pa.y, [11.0, 13.0])
        assert np.allclose(pa.get('force'), [1.0, 1.1, 1.2,
                                             3.0, 3.1, 3.2])

    def test_particle_array_can_create_warp_backend(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0], backend='warp')

        assert isinstance(pa.gpu, WarpDeviceHelper)
        assert np.allclose(pa.x, pa.gpu.x.get())

    def test_property_and_constant_updates_are_mirrored(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0], backend='warp')

        pa.add_property('test', data=[3.0, 4.0])
        assert np.allclose(pa.test, pa.gpu.test.get())

        pa.add_constant('alpha', [0.25, 0.5])
        assert np.allclose(pa.constants['alpha'].get_npy_array(),
                           pa.gpu.alpha.get())

        pa.remove_property('test')
        assert not hasattr(pa.gpu, 'test')
        assert 'test' not in pa.gpu._data
        assert 'test' not in pa.gpu.properties

    def test_particle_array_align_and_property_readback(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                y=[10.0, 11.0, 12.0, 13.0],
                                backend='warp')
        pa.add_property('force', data=[0.0, 0.1, 0.2,
                                       1.0, 1.1, 1.2,
                                       2.0, 2.1, 2.2,
                                       3.0, 3.1, 3.2],
                        stride=3)
        pa.tag[:] = [1, 0, 2, 0]
        pa.gpu.push('tag')

        pa.align_particles()
        props = pa.get_property_arrays(all=True, only_real=False)

        assert pa.get_number_of_particles(real=True) == 2
        assert np.allclose(props['x'], [1.0, 3.0, 0.0, 2.0])
        assert np.allclose(props['y'], [11.0, 13.0, 10.0, 12.0])
        assert np.allclose(props['force'], [1.0, 1.1, 1.2,
                                            3.0, 3.1, 3.2,
                                            0.0, 0.1, 0.2,
                                            2.0, 2.1, 2.2])

    def test_particle_array_only_real_readback(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                backend='warp')
        pa.tag[:] = [1, 0, 2, 0]
        pa.gpu.push('tag')

        pa.align_particles()
        props = pa.get_property_arrays(all=True, only_real=True)

        assert np.allclose(props['x'], [1.0, 3.0])
        assert np.allclose(props['tag'], [0, 0])

    def test_particle_array_remove_particles(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                backend='warp')
        pa.tag[:] = [0, 1, 2, 0]
        pa.gpu.push('tag')

        pa.remove_particles([1], align=False)
        props = pa.get_property_arrays(all=True, only_real=False)

        assert np.allclose(props['x'], [0.0, 2.0, 3.0])
        assert np.allclose(props['tag'], [0, 2, 0])

    def test_remove_particles_raises_for_too_many_indices(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0], backend='warp')

        with pytest.raises(ValueError):
            pa.remove_particles([0, 1, 2])

    def test_particle_array_remove_tagged_particles(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                backend='warp')
        pa.tag[:] = [0, 1, 2, 0]
        pa.gpu.push('tag')

        pa.remove_tagged_particles(2, align=True)
        props = pa.get_property_arrays(all=True, only_real=False)

        assert pa.get_number_of_particles(real=True) == 2
        assert np.allclose(props['x'][:2], [0.0, 3.0])
        assert np.allclose(props['tag'][:2], [0, 0])

    def test_particle_array_add_particles(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0], y=[5.0, 6.0],
                                backend='warp')

        pa.add_particles(x=[2.0, 3.0], tag=[1, 0], align=True)
        props = pa.get_property_arrays(all=True, only_real=False)

        assert pa.get_number_of_particles(real=True) == 3
        assert np.allclose(props['x'], [0.0, 1.0, 3.0, 2.0])
        assert np.allclose(props['y'], [5.0, 6.0, 0.0, 0.0])
        assert np.allclose(props['tag'], [0, 0, 0, 1])

    def test_particle_array_extend_and_resize(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0], backend='warp')

        pa.extend(2)
        assert pa.gpu.get_number_of_particles() == 4
        assert np.allclose(pa.gpu.x.get(), [0.0, 1.0, 0.0, 0.0])
        assert np.allclose(pa.gpu.tag.get(), [0, 0, 0, 0])

        pa.gpu.resize(2)
        assert pa.gpu.get_number_of_particles() == 2
        assert np.allclose(pa.gpu.x.get(), [0.0, 1.0])

    def test_particle_array_append_parray(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0], backend='warp')
        other = get_particle_array(name='g', x=[2.0, 3.0])
        other.tag[:] = [1, 0]

        pa.append_parray(other, align=True)
        props = pa.get_property_arrays(all=True, only_real=False)

        assert pa.get_number_of_particles(real=True) == 3
        assert np.allclose(props['x'], [0.0, 1.0, 3.0, 2.0])
        assert np.allclose(props['tag'], [0, 0, 0, 1])

    def test_append_parray_adds_missing_properties_and_constants(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0], backend='warp')
        other = get_particle_array(name='g', x=[2.0, 3.0],
                                   temp=[20.0, 30.0],
                                   constants={'alpha': [0.25, 0.5]})

        pa.append_parray(other, align=False, update_constants=True)
        props = pa.get_property_arrays(all=True, only_real=False)

        assert np.allclose(props['x'], [0.0, 1.0, 2.0, 3.0])
        assert np.allclose(props['temp'], [0.0, 0.0, 20.0, 30.0])
        assert np.allclose(pa.gpu.alpha.get(), [0.25, 0.5])

    def test_empty_clone_preserves_schema_and_constants(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0], backend='warp',
                                constants={'alpha': [0.25, 0.5]})
        pa.add_property('force', stride=3)
        pa.set_output_arrays(['x', 'force'])

        clone = pa.gpu.empty_clone()

        assert clone.name == 'f'
        assert clone.gpu.get_number_of_particles() == 0
        assert clone.stride['force'] == 3
        assert 'alpha' in clone.gpu.constants
        assert np.allclose(clone.gpu.alpha.get(), [0.25, 0.5])
        assert set(clone.output_property_arrays) == set(['x', 'force'])

    def test_particle_array_extract_particles(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                backend='warp')
        pa.tag[:] = [1, 0, 2, 0]
        pa.gpu.push('tag')

        result = pa.extract_particles([1, 3], align=True)
        props = result.get_property_arrays(all=True, only_real=False)

        assert result.get_number_of_particles(real=True) == 2
        assert np.allclose(props['x'], [1.0, 3.0])
        assert np.allclose(props['tag'], [0, 0])

    def test_extract_particles_with_specific_strided_props(self):
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                backend='warp')
        pa.add_property('force', data=[0.0, 0.1, 0.2,
                                       1.0, 1.1, 1.2,
                                       2.0, 2.1, 2.2,
                                       3.0, 3.1, 3.2],
                        stride=3)

        result = pa.extract_particles([1, 3], align=False,
                                      props=['x', 'force'])
        props = result.get_property_arrays(all=True, only_real=False)

        assert 'y' not in props
        assert 'x' in props
        assert 'force' in props
        assert np.allclose(props['x'], [1.0, 3.0])
        assert np.allclose(props['force'], [1.0, 1.1, 1.2,
                                            3.0, 3.1, 3.2])

    def test_max_reports_device_value(self):
        pa = get_particle_array(name='f', x=[0.0, 3.0, 2.0],
                                backend='warp')

        assert pa.gpu.max('x') == 3.0
