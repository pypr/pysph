from pysph.sph.scheme import Scheme
import numpy

class ADKEScheme(Scheme):
    def __init__(self, fluids, solids, dim, gamma=1.4, alpha=1.0, beta=2.0,
            k=1.0, eps=0.0, g1=0, g2=0):
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.eps = eps
        self.g1 = g1
        self.g2 = g2

    """
    def consume_user_options(self, options):
        vars = ['gamma', 'alpha', 'beta', 'g1', 'g2', 'k', 'eps']
        for var in vars:
            setattr(self, var, getattr(options, var))
    """

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.basic_equations import SummationDensity
        from pysph.sph.gas_dynamics.basic import ( IdealGasEOS,
                ADKEAccelerations, SummationDensityADKE)
        from pysph.sph.gas_dynamics.boundary_equations import (
                TransmissiveBoundary)

        equations = []
        g1 = []
        for solid in self.solids:
            g1.append(TransmissiveBoundary(solid, sources=self.fluids))
        equations.append(Group(g1, update_nnps=True,iterate=False))

        g2 = []
        for fluid in self.fluids:
            g2.append(SummationDensityADKE(
                fluid,  sources=self.fluids + self.solids, k = self.k,
                eps= self.eps
                ))
        equations.append(Group(g2, update_nnps=True, iterate=False))

        g5 = []
        for solid in self.solids:
            g5.append(TransmissiveBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g5))




        g3 = []
        for fluid in self.fluids:
            g3.append(SummationDensity(fluid, self.fluids+self.solids))
        equations.append(Group(g3, update_nnps=True,iterate=False))



        g7 = []
        for solid in self.solids:
            g7.append(TransmissiveBoundary(solid, sources=self.fluids))
        equations.append(Group(equations=g7))



        g4 = []
        for elem in self.fluids+self.solids:
            g4.append(IdealGasEOS(elem, sources=None, gamma=self.gamma))
        equations.append(Group(equations=g4))
        g6 = []

        for fluid in self.fluids:
            g6.append(ADKEAccelerations(
                dest=fluid, sources = self.fluids + self.solids,
                alpha=self.alpha, beta=self.beta, g1=self.g1, g2=self.g2,
                k= self.k, eps=self.eps
                ))

        equations.append(Group(equations=g6))
        return equations

    def configure_solver(self, kernel=None, integrator_cls=None,
                           extra_steppers=None, **kw):
        """Configure the solver to be generated.

        Parameters
        ----------

        kernel : Kernel instance.
            Kernel to use, if none is passed a default one is used.
        integrator_cls : pysph.sph.integrator.Integrator
            Integrator class to use, use sensible default if none is
            passed.
        extra_steppers : dict
            Additional integration stepper instances as a dict.
        **kw : extra arguments
            Any additional keyword args are passed to the solver instance.
        """
        from pysph.base.kernels import Gaussian
        if kernel is None:
            kernel = Gaussian(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import PECIntegrator
        from pysph.sph.integrator_step import ADKEStep

        cls = integrator_cls if integrator_cls is not None else PECIntegrator
        step_cls = ADKEStep
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def setup_properties(self, particles, clean=True):
        from pysph.base.utils import get_particle_array
        particle_arrays = dict([(p.name, p) for p in particles])
        lng = numpy.zeros(1, dtype=float)
        consts ={ 'lng': lng}
        required_props = [
                'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'cs', 'p',
                'e', 'au', 'av', 'aw', 'arho', 'ae', 'am', 'ah', 'x0', 'y0',
                'z0', 'u0', 'v0', 'w0', 'rho0', 'e0', 'h0', 'div',  'h0',
                'wij', 'htmp']


        dummy = get_particle_array(constants=consts,
                additional_props=required_props,
                name='junk')
        dummy.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm', 'h',
            'cs', 'p', 'e','au', 'av', 'ae', 'pid', 'gid', 'tag'] )

        props = list(dummy.properties.keys())
        output_props = dummy.output_property_arrays
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, props, clean)
            pa.lng[0] = 0
            pa.set_output_arrays(output_props)

        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            #pa.lng[0] = 0

