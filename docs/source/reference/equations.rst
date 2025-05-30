SPH equations
===============

.. autoclass:: pysph.sph.equation.Equation
   :members:

.. automodule:: pysph.sph.basic_equations
   :members:
   :undoc-members:

.. automodule:: pysph.sph.wc.basic
   :members:
   :undoc-members:

.. automodule:: pysph.sph.wc.viscosity
   :members:
   :undoc-members:

.. automodule:: pysph.sph.wc.transport_velocity
   :members:
   :undoc-members:

.. automodule:: pysph.sph.wc.gtvf
   :members:
   :undoc-members:

.. automodule:: pysph.sph.wc.density_correction
   :members:
   :undoc-members:

.. automodule:: pysph.sph.wc.kernel_correction
   :members:
   :undoc-members:

.. automodule:: pysph.sph.wc.crksph
   :members:
   :undoc-members:

.. automodule:: pysph.sph.wc.pcisph
   :members:
   :undoc-members:

.. automodule:: pysph.sph.boundary_equations
   :members:
   :undoc-members:

.. automodule:: pysph.sph.solid_mech.basic
   :members:
   :undoc-members:

.. automodule:: pysph.sph.solid_mech.hvi
   :members:
   :undoc-members:

Gas Dynamics
-------------

.. automodule:: pysph.sph.gas_dynamics.basic
   :members:
   :undoc-members:

.. automodule:: pysph.sph.gas_dynamics.boundary_equations
   :members:
   :undoc-members:
   
Surface tension
----------------

.. automodule:: pysph.sph.surface_tension
   :members:
   :undoc-members:

Implicit Incompressible SPH
----------------------------

.. automodule:: pysph.sph.iisph
   :members:
   :undoc-members:

Hopkins' ‘Traditional’ SPH (TSPH)
---------------------------------

.. automodule:: pysph.sph.gas_dynamics.tsph
   :members: TSPHScheme, SummationDensity, IdealGasEOS, VelocityGradDivC1,
             BalsaraSwitch, WallBoundary, UpdateGhostProps, MomentumAndEnergy
   :undoc-members:
   :member-order: bysource

Hopkins' ‘Modern’ SPH (PSPH)
----------------------------

.. automodule:: pysph.sph.gas_dynamics.psph
   :members: PSPHScheme, PSPHSummationDensityAndPressure, GradientKinsfolkC1,
             LimiterAndAlphas, WallBoundary,
             UpdateGhostProps, MomentumAndEnergy
   :undoc-members:
   :member-order: bysource

MAGMA2
--------------

.. automodule:: pysph.sph.gas_dynamics.magma2
   :members: IncreaseSmoothingLength, UpdateSmoothingLength,
             SummationDensityMPMStyle, IdealGasEOS, AuxiliaryGradient,
             CorrectionMatrix, FirstGradient, SecondGradient,
             EntropyBasedDissipationTrigger, WallBoundary, 
             MomentumAndEnergyMI1, MomentumAndEnergyMI2,
             MomentumAndEnergyStdGrad, EvaluateTildeMu,
             SettleByArtificialPressure
   :undoc-members:
   :member-order: bysource

Rigid body motion
-----------------

.. automodule:: pysph.sph.rigid_body
   :members:
   :undoc-members:

Miscellaneous
--------------

.. automodule:: pysph.sph.misc.advection
   :members:
   :undoc-members:

.. automodule:: pysph.base.reduce_array
   :members:
   :undoc-members:

Group of equations
-------------------

.. autoclass:: pysph.sph.equation.Group
   :special-members:

.. autoclass:: pysph.sph.equation.MultiStageEquations
   :special-members: