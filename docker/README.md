## Docker related files

The docker files are available at https://hub.docker.com/u/pysph/

The `base` sub-directory contains a Dockerfile that is used to make the base
image that can be easily used to test PySPH on both Python-2.7 and Python-3.5.
This is the base image for any other PySPH related docker images.

The base image only contains the necessary packages so as to run *all* the
tests.  It therefore include all the dependencies like mpi4py, Zoltan, Mayavi,
and h5py so as to exercise all the tests.
