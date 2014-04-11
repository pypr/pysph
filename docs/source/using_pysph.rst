.. _introduction:

Using the PySPH framework
==========================

In this document, we describe the fundamental data structures for
working with particles in PySPH. Take a look at the :ref:`tutorials`
if you want to try out some of the examples. For the experienced user,
take a look at :ref:`design_overview` if you want to extend PySPH for
your application.


Working With Particles
-----------------------

As an object oriented framework for particle methods, PySPH provides
convenient data structures to store and manipulate collections of
particles. These can be constructed from within Python and are fully
compatible with NumPy arrays. We begin with a brief description for
the basic data structures for arrays

.. py:currentmodule:: pyzoltan.core.carray

C-arrays
---------

The :py:class:`BaseArray` class provides a typed array data structure called
**BaseArray**. These are used throughout PySPH and are fundamentally very
similar to NumPy arrays. The following named types are supported:

    - :py:class:`UIntArray`    (32 bit unsigned integers)
    - :py:class:`IntArray`     (32 bit signed integers)
    - :py:class:`LongArray`    (64 bit signed integers)
    - :py:class:`DoubleArray`  (64 bit floating point numbers

Some simple commands to work with **BaseArrays** from the interactive
shell are given below

.. code-block:: python

    >>> import numpy
    >>> from pyzoltan.core.carray import DoubleArray
    >>> array = DoubleArray(10)                      # array of doubles of length 10
    >>> array.set_data( numpy.arange(10) )           # set the data from a NumPy array
    >>> array.get(3)                                 # get the value at a given index
    >>> array.set(5, -1.0)                           # set the value at an index to a value


.. py:currentmodule:: pysph.base.particle_array

ParticleArray
--------------

In PySPH, a collection of **BaseArrays** make up what is called a
:py:class:`ParticleArray`. This is the main data structure that is used to
represent particles and can be created from NumPy arrays like so:

.. code-block:: python

   >>> import numpy
   >>> from pysph.base.utils import get_particle_array
   >>> x, y = numpy.mgrid[0:1:0.1, 0:1:0.1]             # create some data
   >>> x = x.ravel(); y = y.ravel()                     # flatten the arrays
   >>> pa = get_particle_array(name='array', x=x, y=y)  # create the particle array

In the above, the helper function :py:func:`get_particle_array` will
instantiate and return a :py:class:`ParticleArray` with properties `x` and `y`
set from given NumPy arrays. In general, a :py:class:`ParticleArray` can be
instantiated with an arbitrary number of properties. Each property is stored
internally as a :py:class:`pyzoltan.core.carray.BaseArray` of the appropriate
type.

By default, every **ParticleArray** will have the following properties:

    - `x, y, z`   : Position coordinates (doubles)
    - `u, v, w`   : Velocity (doubles)
    - `h, m, rho` : Smoothing length, mass and density (doubles)
    - `au, av, aw`: Accelerations (doubles)
    - `p`         : Pressure (doubles)
    - `gid`       : Unique global index (unsigned int)
    - `pid`       : Processor id (int)
    - `tag`       : Tag (int)

The role of the particle properties like positions, velocities and
other variables is clear.

PySPH introduces a global identifier for a particle which is required
to be *unique* for that particle. This is represented with the
property *gid* which is of type *unsigned int*. This property is
used in the parallel load balancing algorithm with Zoltan.

The property *pid* for a particle is an *integer* that is used to
identify the processor to which the particle is currently assigned.

The property *tag* is an *integer* that is used for any other
identification. For example, we might want to mark all boundary
particles with the tag 100. Using the *tag*, we can delete all such
particles as

.. code-block:: python

   >>> pa.remove_tagged_particles(tag=100)

This gives us a very flexible way to work with particles. Another way
of deleting/extracting particles is by providing the indices for the
particles in a **LongArray**:

.. code-block:: python

   >>> indices = numpy.array([1,3,5,7])
   >>> la = LongArray(indices.size); la.set_data(indices)
   >>> pa.remove_particles( la )
   >>> extracted = pa.extract_particles(la, props=['rho', 'x', 'y'])

:py:class:`ParticleArrays` can be concatenated:

.. code-block:: python

   >>> pa.append_parray(another_array)

To set a given list of properties to zero:

.. code-block:: python

   >>> props = ['au', 'av', 'aw']
   >>> pa.set_to_zero(props)


.. py:currentmodule:: pysph.base.nnps

Nearest Neighbour Particle Searching
-------------------------------------

To carry out pairwise interactions for SPH, we need to find the nearest
neighbours for a given particle within a specified interaction radius. The
:py:class:`NNPS` object is responsible for handling these nearest neighbour
queries for a *list* of particle arrays:

.. code-block:: python

   >>> from pysph.base import nnps
   >>> pa1 = get_particle_array(...)                    # create one particle array
   >>> pa2 = get_particle_array(...)                    # create another particle array
   >>> particles = [pa1, pa2]
   >>> nps = nnps.LinkedListNNPS(dim=3, particles=particles, radius_scale=3)

The above will create an :py:class:`NNPS` object that uses the classical
*linked-list* algorithm for nearest neighbour searches. The radius of
interaction is determined by the argument `radius_scale`. The book-keeping
cells have a length of :math:`\text{radius_scale} \times h_{\text{max}}`,
where :math:`h_{\text{max}}` is the maximum smoothing length of *all*
particles assigned to the local processor.

Since we allow a list of particle arrays, we need to distinguish
between *source* and *destination* particle arrays in the neighbor
queries.

.. note::

   A **destination** particle is a particle belonging to that species
   for which the neighbors are sought.

   A **source** particle is a particle belonging to that species which
   contributes to a given destination particle.

With these definitions, we can query for nearest neighbors like so:

.. code-block:: python

   >>> nbrs = UIntArray()
   >>> nps.get_nearest_particles(src_index, dst_index, d_idx, nbrs)

where `src_index`, `dst_index` and `d_idx` are integers. This will
return, for the *d_idx* particle of the *dst_index* particle array
(species), nearest neighbors from the *src_index* particle array
(species).

If we want to re-compute the data structure for a new distribution of
particles, we can call the :py:meth:`NNPS.update` method:

.. code-block:: python

   >>> nps.update()
