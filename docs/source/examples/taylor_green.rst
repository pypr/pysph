.. _taylor_green:

The Taylor-Green Vortex
------------------------

This example solves the classic Taylor-Green Vortex problem in two-dimensions.
To run it one may do::

   $ pysph run taylor_green

There are many command line options that this example provides, check them out with::

   $ pysph run taylor_green -h

The example source can be seen at `taylor_green.py
<https://github.com/pypr/pysph/tree/master/pysph/examples/taylor_green.py>`_.


This example demonstrates several useful features:

* user defined command line arguments and how they can be used.
* running the problem with multiple schemes.
* periodicity in both dimensions.
* post processing of generated data.
* using the :py:class:`pysph.tools.sph_evaluator.SPHEvaluator` class for post-processing.

We discuss each of these below.

User command line arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user defined command line arguments are easy to add. The following code
snippet demonstrates how one adds this.

.. code-block:: python

  class TaylorGreen(Application):
      def add_user_options(self, group):
          group.add_argument(
              "--init", action="store", type=str, default=None,
              help="Initialize particle positions from given file."
          )
          group.add_argument(
              "--perturb", action="store", type=float, dest="perturb", default=0,
              help="Random perturbation of initial particles as a fraction "\
                  "of dx (setting it to zero disables it, the default)."
          )
          # ...

This code is straight-forward Python code to add options using the `argparse
API <https://docs.python.org/3/library/argparse.html>`_. It is important to
note that the options are then available in the application's ``options``
attribute and can be accessed as ``self.options`` from the application's
methods.  The ``consume_user_options`` method highlights this.

.. code-block:: python

    def consume_user_options(self):
        nx = self.options.nx
        re = self.options.re

        self.nu = nu = U*L/re
        # ...

This method is called after the command line arguments are passed. To refresh
your memory on the order of invocation of the various methods of the
application, see the documentation of the
:py:class:`pysph.solver.application.Application` class. This shows that once
the application is run using the ``run`` method, the command line arguments
are parsed and the following methods are called (this means that at this
point, the application has a valid ``self.options``):

- ``consume_user_options()``
- ``configure_scheme()``

The ``configure_scheme`` is important as this example allows the user to
change the Reynolds number which changes the viscosity as well as the
resolution via ``--nx`` and ``--hdx``.  The code for the configuration looks like:

.. code-block:: python

    def configure_scheme(self):
        scheme = self.scheme
        h0 = self.hdx * self.dx
        if self.options.scheme == 'tvf':
            scheme.configure(pb=self.options.pb_factor*p0, nu=self.nu, h0=h0)
        elif self.options.scheme == 'wcsph':
            scheme.configure(hdx=self.hdx, nu=self.nu, h0=h0)
        elif self.options.scheme == 'edac':
            scheme.configure(h=h0, nu=self.nu, pb=self.options.pb_factor*p0)
        kernel = QuinticSpline(dim=2)
        scheme.configure_solver(kernel=kernel, tf=self.tf, dt=self.dt)

Note the use of the ``self.options.scheme`` and the use of the
``scheme.configure`` method. Furthermore, the method also calls the scheme's
``configure_solver`` method.


Using multiple schemes
~~~~~~~~~~~~~~~~~~~~~~

This is relatively easy, this is achieved by using the
:py:class:`pysph.sph.scheme.SchemeChooser` scheme as follows:

.. code-block:: python

    def create_scheme(self):
        wcsph = WCSPHScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, h0=h0,
            hdx=hdx, nu=None, gamma=7.0, alpha=0.0, beta=0.0
        )
        tvf = TVFScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, nu=None,
            p0=p0, pb=None, h0=h0
        )
        edac = EDACScheme(
            ['fluid'], [], dim=2, rho0=rho0, c0=c0, nu=None,
            pb=p0, h=h0
        )
        s = SchemeChooser(default='tvf', wcsph=wcsph, tvf=tvf, edac=edac)
        return s

When using multiple schemes it is important to recall that each scheme needs
different particle properties. The schemes set these extra properties for you.
In this example, the ``create_particles`` method has the following code:

.. code-block:: python

    def create_particles(self):
        # ...
        fluid = get_particle_array(name='fluid', x=x, y=y, h=h)

        self.scheme.setup_properties([fluid])

The line tht calls ``setup_properties`` passes a list of the particle arrays
to the scheme so the scheme can configure/setup any additional properties.


Periodicity
~~~~~~~~~~~

This is rather easily done with the code in the ``create_domain`` method:

.. code-block:: python

    def create_domain(self):
        return DomainManager(
            xmin=0, xmax=L, ymin=0, ymax=L, periodic_in_x=True,
            periodic_in_y=True
        )

See also :ref:`simulating_periodicity`.


Post-processing
~~~~~~~~~~~~~~~

The code has a significant chunk of code for post-processing the results. This
is in the ``post_process`` method. This demonstrates how to iterate over the
files and read the file data to calculate various quantities. In particular it
also demonstrates the use of the
:py:class:`pysph.tools.sph_evaluator.SPHEvaluator` class. For example consider
the method:

.. code-block:: python

    def _get_sph_evaluator(self, array):
        if not hasattr(self, '_sph_eval'):
            from pysph.tools.sph_evaluator import SPHEvaluator
            equations = [
                ComputeAveragePressure(dest='fluid', sources=['fluid'])
            ]
            dm = self.create_domain()
            sph_eval = SPHEvaluator(
                arrays=[array], equations=equations, dim=2,
                kernel=QuinticSpline(dim=2), domain_manager=dm
            )
            self._sph_eval = sph_eval
        return self._sph_eval

This code, creates the evaluator, note that it just takes the particle arrays
of interest, a set of equations (this can be as complex as the normal SPH
equations, with groups and everything), the kernel, and a domain manager. The
evaluator has two important methods:

 - `update_particle_arrays(...)`: this allows a user to update the arrays
   to a new set of values efficiently.
 - `evaluate`: this actually performs the evaluation of the equations.

The example has this code which demonstrates these:

.. code-block:: python

    def _get_post_process_props(self, array):
            # ...
            sph_eval = self._get_sph_evaluator(array)
            sph_eval.update_particle_arrays([array])
            sph_eval.evaluate()
            # ...

Note the use of the above methods.
