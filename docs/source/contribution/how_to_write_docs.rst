.. _how_to_write_docs:

Contribute to docs
==================

How to build the docs locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build the docs, clone the repository::

   $ git clone https://github.com/pypr/pysph

Make sure to work in an ``pysph`` environment. I will proceed with the further
instructions assuming that the repository is cloned in home directory. Change to
the ``docs`` directory and run ``make html``. ::

   $ cd ~/pysph/docs/
   $ make html


Possible error one might get is::

   $ sphinx-build: Command not found

Which means you don't a have `sphinx-build` in your system. To install across
the system do::

   $ sudo apt-get install python3-sphinx


or to install in an environment locally do::

   $ pip install sphinx

run ``make html`` again. The documentation is built locally at
``~/pysph/docs/build/html`` directory. Open ```index.html`` file by running ::

   $ cd ~/pysph/docs/build/html
   $ xdg-open index.html



How to add the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a starting point one can add documentation to one of the examples in
``~/pysph/pysph/examples`` folder. There is a dedicated
``~/pysph/docs/source/examples`` directory to add documentation to examples.
Choose an example to write documentation for, ::

   $ cd ~/pysph/docs/source/examples
   $ touch your_example.rst

We will write all the documentation in ``rst`` file format. The ``index.rst``
file in the examples directory should know about our newly created file, add a
reference next to the last written example.::

   * :ref:`Some_example`:
   * :ref:`Other_example`:
   * :ref:`taylor_green`: the Taylor-Green Vortex problem in 2D.
   * :ref:`sphere_in_vessel`: A sphere floating in a hydrostatic tank example.
   * :ref:`your_example_file`: Description of the example.

and at the top of the example file add the reference, for example in
``your_example_file.rst``, you should add,::

   .. _your_example_file


That's it, add the documentation and send a pull request.
