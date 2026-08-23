Using Warp-backed particle arrays
=================================

PySPH can create a ParticleArray with a NVIDIA Warp-backed device mirror by
passing ``backend='warp'`` to :func:`pysph.base.utils.get_particle_array`.
The host ParticleArray remains the public object you use from Python and
Cython.  The Warp mirror stores property and constant arrays on the selected
Warp device and is available as ``pa.gpu``.

This is an experimental backend intended for ParticleArray storage and mutation
work.  It does not migrate SPH equations, integrators, or NNPS kernels to Warp.

Creating a Warp-backed ParticleArray
------------------------------------

Create the array as usual, adding ``backend='warp'``:

.. code-block:: python

    from pysph.base.utils import get_particle_array

    pa = get_particle_array(
        name='fluid',
        x=[0.0, 1.0, 2.0],
        y=[0.0, 0.0, 0.0],
        h=0.1,
        m=1.0,
        rho=1000.0,
        backend='warp',
    )

    print(pa.backend)       # warp
    print(pa.gpu.x.get())   # Warp device mirror copied back to NumPy

The object returned by ``get_particle_array`` is still a regular
ParticleArray.  The Warp-specific part is the device helper attached to
``pa.gpu``.

Synchronizing host and device data
----------------------------------

The host arrays and the Warp mirror are intentionally separate.  If you mutate
host arrays directly, push the changed properties to the Warp mirror:

.. code-block:: python

    pa.x[:] = [3.0, 4.0, 5.0]
    pa.tag[:] = [0, 1, 0]

    pa.gpu.push('x', 'tag')

Use ``push()`` with no arguments to copy all mirrored properties and constants.

If you mutate the Warp mirror directly, pull the changed arrays back to the
host:

.. code-block:: python

    pa.gpu.x[:] = 10.0
    pa.gpu.pull('x')

    assert (pa.x == [10.0, 10.0, 10.0]).all()

Use ``pull()`` with no arguments to copy all mirrored properties and constants
back to the host ParticleArray.

Working with local, remote, and ghost particles
-----------------------------------------------

The ``tag`` property has the same meaning as in a normal ParticleArray:
``0`` is local, ``1`` is remote, and ``2`` is ghost.  Calling
``align_particles`` partitions local particles first and updates the real
particle count:

.. code-block:: python

    pa.tag[:] = [1, 0, 2, 0]
    pa.gpu.push('tag')

    pa.align_particles()

    print(pa.get_number_of_particles(real=True))  # 2
    print(pa.get('x'))                            # local particles only

The Warp backend supports strided properties during alignment:

.. code-block:: python

    pa.add_property(
        'force',
        data=[0.0, 0.1, 0.2,
              1.0, 1.1, 1.2,
              2.0, 2.1, 2.2],
        stride=3,
    )

    pa.align_particles()

Particle mutations
------------------

The Warp mirror supports the same first-level ParticleArray mutation methods
for particle-table work:

.. code-block:: python

    pa.add_particles(x=[6.0, 7.0], tag=[1, 0])
    pa.remove_tagged_particles(1)
    pa.remove_particles([0], align=False)

Appending and extracting ParticleArrays also works:

.. code-block:: python

    wall = get_particle_array(name='wall', x=[8.0, 9.0], backend='warp')
    pa.append_parray(wall)

    subset = pa.extract_particles([0, 2], props=['x'])

When you request specific properties during extraction, PySPH still preserves
its baseline system properties such as ``tag``, ``pid``, and ``gid``.

Constants and additional properties
-----------------------------------

Constants are mirrored to Warp and remain fixed-size arrays:

.. code-block:: python

    pa.add_constant('gravity', [0.0, -9.81, 0.0])
    print(pa.gpu.gravity.get())

New properties are mirrored when they are added:

.. code-block:: python

    pa.add_property('temperature', data=[300.0] * pa.get_number_of_particles())
    print(pa.gpu.temperature.get())

Precision
---------

Floating-point properties follow Compyle's ``use_double`` configuration when
the Warp mirror is created.  With ``use_double=True``, float properties are
mirrored as ``float64``.  With ``use_double=False``, they are mirrored as
``float32``.

Example:

.. code-block:: python

    from compyle.config import get_config

    cfg = get_config()
    cfg.use_double = False

    pa = get_particle_array(name='fluid', x=[0.0, 1.0], backend='warp')
    print(pa.gpu.x.dtype)  # float32

Current limitations
-------------------

The Warp backend is currently a ParticleArray device mirror.  Host
``BaseArray`` storage remains authoritative for the public ParticleArray API.
The first implementation focuses on correctness for property storage,
synchronization, alignment, constants, and particle-table mutations.

Equation evaluation, integrators, NNPS kernels, and full solver execution are
not migrated to Warp by this backend.
