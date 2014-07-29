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
from pysph.base.cython_generator import CythonGenerator, get_func_definition


class IntegratorCythonHelper(object):
    """A helper that generates Cython code for the Integrator class.
    """
    def __init__(self, integrator):
        self.object = integrator

    def get_code(self):
        if self.object is not None:
            path = join(dirname(__file__), 'integrator.mako')
            template = Template(filename=path)
            return template.render(helper=self)
        else:
            return ''

    def setup_compiled_module(self, module, acceleration_eval):
        # Create the compiled module.
        cython_integrator = module.Integrator(
            acceleration_eval, self.object.steppers
        )
        # Setup the integrator to use this compiled module.
        self.object.set_compiled_object(cython_integrator)

    ##########################################################################
    # Mako interface.
    ##########################################################################
    def get_particle_array_names(self):
        return ', '.join(sorted(self.object.steppers.keys()))

    def get_stepper_code(self):
        classes = {}
        for dest, stepper in self.object.steppers.iteritems():
            cls = stepper.__class__.__name__
            classes[cls] = stepper

        wrappers = []
        code_gen = CythonGenerator()
        for cls in sorted(classes.keys()):
            code_gen.parse(classes[cls])
            wrappers.append(code_gen.get_code())
        return '\n'.join(wrappers)

    def get_stepper_defs(self):
        lines = []
        for dest, stepper in self.object.steppers.iteritems():
            cls_name = stepper.__class__.__name__
            code = 'cdef public {cls} {name}'.format(cls=cls_name,
                                                     name=dest+'_stepper')
            lines.append(code)
        return '\n'.join(lines)

    def get_stepper_init(self):
        lines = []
        for dest, stepper in self.object.steppers.iteritems():
            cls_name = stepper.__class__.__name__
            code = 'self.{name} = {cls}(**steppers["{dest}"].__dict__)'\
                        .format(name=dest+'_stepper', cls=cls_name,
                                dest=dest)
            lines.append(code)
        return '\n'.join(lines)

    def get_args(self, dest, method):
        stepper = self.object.steppers[dest]
        meth = getattr(stepper, method)
        return inspect.getargspec(meth).args

    def get_array_declarations(self, method):
        arrays = set()
        for dest in self.object.steppers:
            s, d = get_array_names(self.get_args(dest, method))
            arrays.update(s | d)

        decl = []
        for arr in sorted(arrays):
            decl.append('cdef double* %s'%arr)
        return '\n'.join(decl)

    def get_array_setup(self, dest, method):
        s, d = get_array_names(self.get_args(dest, method))
        lines = ['%s = dst.%s.data'%(n, n[2:]) for n in s|d]
        return '\n'.join(lines)

    def get_stepper_loop(self, dest, method):
        args = self.get_args(dest, method)
        if 'self' in args:
            args.remove('self')
        call_args = ', '.join(args)
        c = 'self.{obj}.{method}({args})'\
                .format(obj=dest+'_stepper', method=method, args=call_args)
        return c

    def get_stepper_method_wrapper_names(self):
        """Returns the names of the methods we should wrap.  For a 2 stage
        method this will return ('initialize', 'stage1', 'stage2')
        """
        methods = set(['initialize'])
        for stepper in self.object.steppers.values():
            stages = [x for x in dir(stepper) if x.startswith('stage')]
            methods.update(stages)
        return list(sorted(methods))

    def get_timestep_code(self):
        method = self.object.one_timestep
        sourcelines = inspect.getsourcelines(method)[0]
        defn, lines = get_func_definition(sourcelines)
	return dedent(''.join(lines))
