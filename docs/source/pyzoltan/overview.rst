.. _pyzoltan-docs:

==============
Introduction
==============

.. py:currentmodule:: pyzoltan.core.zoltan

PyZoltan is as the name suggests, is a Python wrapper for the Zoltan_
data management library. Although it's primary purpose is a tool for
dynamic load balancing for PySPH, the features are general enough to
warrant a separate discussion. In :ref:`introduction`, we touched upon
how to perform nearest neighbor queries in a distributed
environment. In this document, we will introduce the PyZoltan
interface in it's native non-SPH format.

In PyZoltan, we wrap the specific routines and objects that we wish to
use. The following features of Zoltan are currently supported:

 - Dynamic load balancing using geometric algorithms
 - Unstructured point-to-point communication
 - Distributed data directories

PyZoltan interfaces with Zoltan, which is itself a library that is
called to perform specific tasks on the application data. Information
about the application data is provided to Zoltan through the `method
of callbacks
<http://www.cs.sandia.gov/Zoltan/ug_html/ug_query.html>`_, whereby,
query functions are registered with Zoltan. These query functions are
called internally by Zoltan and are responsible to provide the correct
information about the application data to the library. The user is
responsible to make available this data in consonance with the
application requirement.

-----------------
A simple example
-----------------

The following simple example demonstrates the partitioning of a random
collection of points in the unit cube :math:`[0,1]^3`. The objects to
be partitioned in this case is therefore the points themselves which
can be thought of as particles in an SPH simulation.

.. code-block:: python

    # Imports
    import mpi4py.MPI as mpi
    from pyzoltan.core.carray import UIntArray, DoubleArray
    from pyzoltan.core import zoltan

    from numpy import random
    import numpy as np

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    colors = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood']

    # CREATE THE PARTICLE DATA LOCALLY ON EACH PROCESSOR
    numPoints = 1<<12

    x = random.random( numPoints )
    y = random.random( numPoints )
    z = random.random( numPoints )
    gid = np.arange( numPoints*size, dtype=np.uint32 )[rank*numPoints:(rank+1)*numPoints]

    # GATHER THE DISTRIBUTED DATA ON THE ROOT FOR PLOTTING
    X = np.zeros( size * numPoints )
    Y = np.zeros( size * numPoints )
    Z = np.zeros( size * numPoints )
    GID = np.arange( numPoints*size, dtype=np.uint32 )

    comm.Gather( sendbuf=x, recvbuf=X, root=0 )
    comm.Gather( sendbuf=y, recvbuf=Y, root=0 )
    comm.Gather( sendbuf=z, recvbuf=Z, root=0 )

    # ROOT PLOTS THE DATA
    if rank == 0:
	fig = plt.figure()
	s1 = fig.add_subplot(111)
	s1.axes = Axes3D(fig)
	for i in range(size):
	    s1.axes.plot3D( X[i*numPoints:(i+1)*numPoints],
			    Y[i*numPoints:(i+1)*numPoints],
			    Z[i*numPoints:(i+1)*numPoints],
			    c=colors[i], marker='o', linestyle='None',
			    alpha=0.8)

	s1.axes.set_xlabel( 'X' )
	s1.axes.set_ylabel( 'Y' )
	s1.axes.set_zlabel( 'Z' )

	plt.title('Initital Distribution')
	plt.savefig( 'initial.pdf' )

    # PARTITION THE POINT SET USING PyZoltan
    xa = DoubleArray(numPoints); xa.set_data(x)
    ya = DoubleArray(numPoints); ya.set_data(y)
    za = DoubleArray(numPoints); za.set_data(z)
    gida = UIntArray(numPoints); gida.set_data(gid)

    # CREATE THE GEOMETRIC LOAD BALANCER
    pz = zoltan.ZoltanGeometricPartitioner(
	dim=3, comm=comm, x=xa, y=ya, z=za, gid=gida)

    # CALL THE LOAD BALANCING FUNCTION
    pz.set_lb_method('RCB') # valid options RCB, RIB, HSFC
    pz.Zoltan_Set_Param('DEBUG_LEVEL', '1')
    pz.Zoltan_LB_Balance()

    # get the new assignments
    my_global_ids = list( gid )

    # REMOVE POINTS TO BE EXPORTED
    for i in range(pz.numExport):
	my_global_ids.remove( pz.exportGlobalids[i] )

    # ADD POINTS TO BE IMPORTED
    for i in range(pz.numImport):
	my_global_ids.append( pz.importGlobalids[i] )


    # GATHER THE NEW DATA ON ROOT
    new_gids = np.array( my_global_ids, dtype=np.uint32 )

    # gather the new gids on root as a list
    NEW_GIDS = comm.gather( new_gids, root=0 )

    # PLOT THE NEW ASSIGNMENTS
    if rank == 0:
	fig = plt.figure()
	s1 = fig.add_subplot(111)
	s1.axes = Axes3D(fig)
	for i in range(size):
	    s1.axes.plot3D( X[ NEW_GIDS[i] ],
			    Y[ NEW_GIDS[i] ],
			    Z[ NEW_GIDS[i] ],
			    c=colors[i], marker='o', linestyle='None',
			    alpha=0.8 )

	s1.axes.set_xlabel( 'X' )
	s1.axes.set_ylabel( 'Y' )
	s1.axes.set_zlabel( 'Z' )

	plt.title('Final Distribution')
	plt.savefig( 'final.pdf' )
	plt.show()

Although the code seems lengthy, a lot of it is concerned with setting
up the initial data and plotting on the root node. After the initial
imports, we define the local data on each processor and broadcast this
to the root node for plotting the initial assignment:

.. code-block:: python

    numPoints = 1<<12

    x = random.random( numPoints )
    y = random.random( numPoints )
    z = random.random( numPoints )
    gid = np.arange( numPoints*size, dtype=np.uint32 )[rank*numPoints:(rank+1)*numPoints]

    X = np.zeros( size * numPoints )
    Y = np.zeros( size * numPoints )
    Z = np.zeros( size * numPoints )
    GID = np.arange( numPoints*size, dtype=np.uint32 )

    comm.Gather( sendbuf=x, recvbuf=X, root=0 )
    comm.Gather( sendbuf=y, recvbuf=Y, root=0 )
    comm.Gather( sendbuf=z, recvbuf=Z, root=0 )

.. note::

   Each object (point) is assigned a *unique* global identifier (the
   `gid` array). The identifiers must be unique for a load balancing
   cycle.

.. note::
 
   The data type of the global identifiers is of type `ZOLTAN_ID_TYPE`
   (default uint32). This is set at the time of building the Zoltan
   library.

After the data has been initialized, we instantiate the
:py:class:`ZoltanGeometricPartitioner` object and set some parameters:

.. code-block:: python

    xa = DoubleArray(numPoints); xa.set_data(x)
    ya = DoubleArray(numPoints); ya.set_data(y)
    za = DoubleArray(numPoints); za.set_data(z)
    gida = UIntArray(numPoints); gida.set_data(gid)

    pz = zoltan.ZoltanGeometricPartitioner(
	dim=3, comm=comm, x=xa, y=ya, z=za, gid=gida)

    pz.set_lb_method('RCB')
    pz.Zoltan_Set_Param('DEBUG_LEVEL', '1')  

.. note::

   We use CArrays internally to represent the data in PyZoltan. This
   is done mainly for compatibility with the PySPH particle data
   structure.

The complete list of parameters can be found in the Zoltan reference
manual. All parameters are supported through the wrapper
:py:meth:`PyZoltan.Zoltan_Set_Param` method. In this example, we set
the desired load balancing algorithm (Recursive Coordinate Bisection)
and the output debug level. Once all the parameters are appropriately
set-up, we can ask Zoltan to provide new assignments for the particles:

.. code-block:: python

    pz.Zoltan_LB_Balance()

This will call the chosen load balancing function internally and upon
return, set a number of lists (arrays) indicating which objects need
to be exported and which objects need to be imported. The data
attributes for the export lists are:

 - *numExport* : Number of objects to be exported
 - *exportLocalids* : Local indices of the objects to be exported
 - *exportGlobalids* : Global indices of the objects to be exported
 - *exportProcs* : A list of size `numExport` indicating to which processor each object is exported

And similar arrays for the import lists. The import/export lists
returned by Zoltan give an application all the information required to
initiate the data transfer.

.. note:: 

   Zoltan does **not** perform the data transfer. The data transfer
   must be done by the application or using the Unstructured
   communication utilities provided by Zoltan.

Given the new assignments, we once again broadcast this to the root to
plot the final partition. The partition generated by this approach is
shown below.

.. figure:: ../../Images/point-partition.png
   :scale: 50
   :align: center

   Point assignment to 4 processors where color indicates
   assignment.

We can see that the `RCB` method has resulted in cuts orthogonal to
the domain axes. Each processor has exactly one fourth of the total
number of particles. 

The code for this example can be found in
`pyzoltan/core/tests/3d_partition.py`

.. _Zoltan: http://www.cs.sandia.gov/Zoltan/
