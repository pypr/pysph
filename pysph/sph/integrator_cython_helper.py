"""Basic code for the templated integrators.

Currently we only support two-step integrators.

These classes are used to generate the code for the actual integrators
from the `sph_eval` module.
"""

import inspect
from os.path import join, dirname
from textwrap import dedent
from mako.template import Template

# Local imports.
from pysph.sph.equation import get_array_names
from .acceleration_eval_cython_helper import get_helper_code
from compyle.api import CythonGenerator, get_func_definition


getfullargspec = getattr(
    inspect, 'getfullargspec', inspect.getargspec
)


class IntegratorCythonHelper(object):
    """A helper that generates Cython code for the Integrator class.
    """
    def __init__(self, integrator, acceleration_eval_helper):
        self.object = integrator
        self.acceleration_eval_helper = acceleration_eval_helper
        pas = acceleration_eval_helper.object.particle_arrays
        self._particle_arrays = dict((x.name, x) for x in pas)
        if self.object is not None:
            self._check_integrator_steppers()

    def get_code(self):
        if self.object is not None:
            path = join(dirname(__file__), 'integrator_cython.mako')
            template = Template(filename=path)
            return template.render(helper=self)
        else:
            return ''

    def setup_compiled_module(self, module, acceleration_eval):
        # Create the compiled module.
        cython_integrator = module.Integrator(
            self.object, acceleration_eval, self.object.steppers
        )
        # Setup the integrator to use this compiled module.
        self.object.set_compiled_object(cython_integrator)

    ##########################################################################
    # Mako interface.
    ##########################################################################
    def get_particle_array_names(self):
        return ', '.join(sorted(self.object.steppers.keys()))

    def get_helper_code(self):
        helpers = []
        for stepper in self.object.steppers.values():
            if hasattr(stepper, '_get_helpers_'):
                for helper in stepper._get_helpers_():
                    if helper not in helpers:
                        helpers.append(helper)

        code = get_helper_code(helpers)
        return '\n'.join(code)

    def get_stepper_code(self):
        classes = {}
        for dest, stepper in self.object.steppers.items():
            cls = stepper.__class__.__name__
            classes[cls] = stepper

        known_types = dict(self.acceleration_eval_helper.known_types)
        known_types.update(dict(t=0.0, dt=0.0))
        code_gen = CythonGenerator(known_types=known_types)

        wrappers = []
        for cls in sorted(classes.keys()):
            code_gen.parse(classes[cls])
            wrappers.append(code_gen.get_code())
        return '\n'.join(wrappers)

    def get_stepper_defs(self):
        lines = []
        for dest, stepper in self.object.steppers.items():
            cls_name = stepper.__class__.__name__
            code = 'cdef public {cls} {name}'.format(cls=cls_name,
                                                     name=dest+'_stepper')
            lines.append(code)
        return '\n'.join(lines)

    def get_stepper_init(self):
        lines = []
        for dest, stepper in self.object.steppers.items():
            cls_name = stepper.__class__.__name__
            code = (
                'self.{name} = {cls}(**steppers["{dest}"].__dict__)'
                .format(name=dest+'_stepper', cls=cls_name, dest=dest)
            )
            lines.append(code)
        return '\n'.join(lines)

    def get_args(self, dest, method):
        stepper = self.object.steppers[dest]
        meth = getattr(stepper, method, None)
        if meth is None:
            return []
        else:
            return getfullargspec(meth).args

    def get_array_declarations(self, method):
        arrays = set()
        for dest in self.object.steppers:
            s, d = get_array_names(self.get_args(dest, method))
            self._check_arrays_for_properties(dest, s | d)
            arrays.update(s | d)

        known_types = self.acceleration_eval_helper.known_types
        decl = []
        for arr in sorted(arrays):
            decl.append('cdef {type} {arr}'.format(
                type=known_types[arr].type, arr=arr
            ))

        return '\n'.join(decl)

    def get_array_setup(self, dest, method):
        s, d = get_array_names(self.get_args(dest, method))
        lines = ['%s = dst.%s.data' % (n, n[2:]) for n in sorted(s | d)]
        return '\n'.join(lines)

    def get_stepper_loop(self, dest, method):
        args = self.get_args(dest, method)
        if 'self' in args:
            args.remove('self')
        call_args = ', '.join(args)
        c = 'self.{obj}.{method}({args})'.format(
            obj=dest+'_stepper', method=method, args=call_args
        )
        return c

    def get_py_stage_code(self, dest, method):
        stepper = self.object.steppers[dest]
        method = 'py_' + method
        if hasattr(stepper, method):
            return 'self.steppers["{dest}"].{method}(dst.array, t, dt)'.format(
                dest=dest, method=method
            )
        else:
            return ''

    def has_stepper_loop(self, dest, method):
        return hasattr(self.object.steppers[dest], method)

    def get_stepper_method_wrapper_names(self):
        """Returns the names of the methods we should wrap.  For a 2 stage
        method this will return ('initialize', 'stage1', 'stage2')
        """
        methods = set()
        for stepper in self.object.steppers.values():
            stages = []
            for x in dir(stepper):
                if x.startswith('py_stage'):
                    stages.append(x[3:])
                elif x.startswith('stage') or x == 'initialize':
                    stages.append(x)
            methods.update(stages)
        return list(sorted(methods))

    def get_timestep_code(self):
        method = self.object.one_timestep
        sourcelines = inspect.getsourcelines(method)[0]
        defn, lines = get_func_definition(sourcelines)
        return dedent(''.join(lines))

    ##########################################################################
    # Private interface.
    ##########################################################################

    def _check_arrays_for_properties(self, dest, args):
        """Given a particle array name and a set of arguments used by an
        integrator stepper method, check if the particle array has the
        required props.
        """

        pa = self._particle_arrays[dest]
        # Remove the 's_' or 'd_'
        props = set([x[2:] for x in args])
        available_props = set(pa.properties.keys()).union(pa.constants.keys())
        if not props.issubset(available_props):
            diff = props.difference(available_props)
            integrator_name = self.object.steppers[dest].__class__.__name__
            names = ', '.join([x for x in sorted(diff)])
            msg = "ERROR: {integrator_name} requires the following "\
                  "properties:\n\t{names}\n"\
                  "Please add them to the particle array '{dest}'.".format(
                      integrator_name=integrator_name, names=names, dest=dest
                  )
            self._runtime_error(msg)

    def _check_integrator_steppers(self):
        for name, stepper in self.object.steppers.items():
            if name not in self._particle_arrays:
                msg = "ERROR: Integrator keyword arguments must correspond "\
                      "to particle array names.\n"\
                      "Given keyword: '{name}' not a valid particle array.\n"\
                      "Valid particle array names: {valid}".format(
                          name=name, valid=sorted(self._particle_arrays.keys())
                      )
                self._runtime_error(msg)

    def _runtime_error(self, msg):
        print('*'*70)
        print(msg)
        print('*'*70)
        raise RuntimeError(msg)
