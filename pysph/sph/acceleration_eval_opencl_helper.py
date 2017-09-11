"""
TODO:

- Add common place for the context/queue
- Add a GPUHelper to ParticleArray.

Basic support:
- Code to actually execute the kernels.
- Particle array to CPU/GPU
- Structs to GPU
- integrator support.

Advanced:
- support for doubles/floats.
- sub groups.
- Iterated groups.
- DT_ADAPT.
- Reduction.
- support get_code for helper functions etc.


"""
import inspect
from os.path import dirname, join

from mako.template import Template

from pysph.base.translator import OpenCLConverter, ocl_detect_type

from .acceleration_eval_cython_helper import (
    get_all_array_names, get_known_types_for_arrays
)


class OpenCLAccelerationEval(object):
    """Does the actual work of performing the evaluation.
    """
    def __init__(self, helper):
        self.helper = helper

    def compute(self, t, dt):
        pass

    def set_nnps(self, nnps):
        pass

    def update_particle_arrays(self, arrays):
        pass


def add_address_space(known_types):
    for v in known_types.values():
        if '__global' not in v.type:
            v.type = '__global ' + v.type


class AccelerationEvalOpenCLHelper(object):
    def __init__(self, acceleration_eval):
        self.object = acceleration_eval
        self.all_array_names = get_all_array_names(
            self.object.particle_arrays
        )
        self.known_types = get_known_types_for_arrays(
            self.all_array_names
        )
        add_address_space(self.known_types)
        self.data = []
        self.ctx = None
        self.queue = None
        self.program = None

    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_code(self):
        path = join(dirname(__file__), 'acceleration_eval_opencl.mako')
        template = Template(filename=path)
        main = template.render(helper=self)
        print(main)
        print(self.data)
        return main

    def setup_compiled_module(self, module):
        object = self.object
        acceleration_eval = OpenCLAccelerationEval(self)
        object.set_compiled_object(acceleration_eval)

    def compile(self, code):
        import pyopencl as cl
        if self.ctx is None:
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, code.encode('ascii')).build()
        return self.program

    ##########################################################################
    # Mako interface.
    ##########################################################################
    def get_header(self):
        # FIXME
        # Write the equivalent for get_code in cython where any extra code,
        # helpers are suitably wrapped as well.
        object = self.object

        headers = []
        transpiler = OpenCLConverter(known_types=self.known_types)
        headers.append(transpiler.parse_instance(object.kernel))

        headers.append(object.all_group.get_equation_wrappers(
            self.known_types
        ))

        return '\n'.join(headers)

    def _get_typed_args(self, args):
        code = []
        for arg in args:
            if arg in self.known_types:
                type = self.known_types[arg].type
            else:
                type = ocl_detect_type(arg, None)

            code.append('{type} {arg}'.format(
                type=type, arg=arg
            ))

        return code

    def _clean_kernel_args(self, args):
        remove = ('d_idx', 's_idx')
        for a in remove:
            if a in args:
                args.remove(a)

    def _get_simple_kernel(self, g_idx, dest, all_eqs, kind):
        assert kind in ('initialize', 'post_loop', 'loop')
        kernel = 'g{g_idx}_{kind}'.format(g_idx=g_idx, kind=kind)
        all_args = []
        py_args = []
        code = ['int d_idx = get_global_id(0);']
        for eq in all_eqs.equations:
            method = getattr(eq, kind, None)
            if method is not None:
                cls = eq.__class__.__name__
                arg = '__global {cls}* {name}'.format(
                    cls=cls, name=eq.var_name
                )
                all_args.append(arg)
                py_args.append(eq.var_name)
                args = inspect.getargspec(method).args
                if 'self' in args:
                    args.remove('self')
                call_args = list(args)
                self._clean_kernel_args(args)
                for x in self._get_typed_args(args):
                    if x not in all_args:
                        all_args.append(x)
                for x in args:
                    if x not in py_args:
                        py_args.append(x)

                call_args.insert(0, eq.var_name)
                code.append(
                    '{cls}_{kind}({args});'.format(
                        cls=cls, kind=kind, args=', '.join(call_args)
                    )
                )

        body = '\n'.join([' '*4 + x for x in code])
        self.data.append(
            dict(kernel=kernel, args=py_args, dest=dest, loop=False)
        )

        return (
            '__kernel void\n{kernel}\n({args})\n{{\n{body}\n}}'.format(
                kernel=kernel, args=', '.join(all_args),
                body=body
            )
        )

    def _declare_precomp_vars(self, context):
        decl = []
        names = list(context.keys())
        names.sort()
        for var in names:
            value = context[var]
            if isinstance(value, int):
                declare = 'long '
                decl.append('{declare}{var} = {value};'.format(
                    declare=declare, var=var, value=value
                ))
            elif isinstance(value, float):
                declare = 'double '
                decl.append('{declare}{var} = {value};'.format(
                    declare=declare, var=var, value=value
                ))
            elif isinstance(value, (list, tuple)):
                decl.append(
                    'double[{size}] {var};'.format(
                        var=var, size=len(value)
                    )
                )
        return decl

    def _set_kernel(self, code, kernel):
        if kernel is not None:
            name = kernel.__class__.__name__
            kern = '%s_kernel(kern, ' % name
            grad = '%s_gradient(kern, ' % name
            grad_h = '%s_gradient_h(kern, ' % name
            deltap = '%s_deltap(kern)' % name

            code = code.replace('DELTAP', deltap).replace('GRADIENT(', grad)
            return code.replace('KERNEL(', kern).replace('GRADH(', grad_h)
        else:
            return code

    def get_initialize_kernel(self, g_idx, dest, all_eqs):
        return self._get_simple_kernel(g_idx, dest, all_eqs, kind='initialize')

    def get_simple_loop_kernel(self, g_idx, dest, all_eqs):
        return self._get_simple_kernel(g_idx, dest, all_eqs, kind='loop')

    def get_post_loop_kernel(self, g_idx, dest, all_eqs):
        return self._get_simple_kernel(g_idx, dest, all_eqs, kind='post_loop')

    def get_loop_kernel(self, g_idx, dest, source, eq_group):
        kind = 'loop'
        kernel = 'g{g_idx}_{source}_on_{dest}_loop'.format(
            g_idx=g_idx, source=source, dest=dest
        )
        context = eq_group.context
        code = self._declare_precomp_vars(context)
        code.append('int d_idx = get_global_id(0);')
        code.append('int s_idx, i;')
        code.append('int n = nbr_length[d_idx];')
        code.append('for (i=start_idx[d_idx]; i<n; i++) {')
        code.append('    s_idx = neighbors[i];')
        pre = []
        for p, cb in eq_group.precomputed.items():
            src = cb.code.strip().splitlines()
            pre.append('\n'.join([' '*4 + x + ';' for x in src]))
        if len(pre) > 0:
            pre.append('')
        code.extend(pre)

        all_args = []
        py_args = []
        for eq in eq_group.equations:
            method = getattr(eq, kind, None)
            if method is not None:
                cls = eq.__class__.__name__
                arg = '__global {cls}* {name}'.format(
                    cls=cls, name=eq.var_name
                )
                all_args.append(arg)
                py_args.append(eq.var_name)
                args = inspect.getargspec(method).args
                if 'self' in args:
                    args.remove('self')
                call_args = list(args)
                self._clean_kernel_args(args)
                for x in self._get_typed_args(args):
                    if x not in all_args and x not in context:
                        all_args.append(x)
                for x in args:
                    if x not in py_args and x not in context:
                        py_args.append(x)

                call_args.insert(0, eq.var_name)
                code.append(
                    '    {cls}_{kind}({args});'.format(
                        cls=cls, kind=kind, args=', '.join(call_args)
                    )
                )

        body = '\n'.join([' '*4 + x for x in code])
        self._set_kernel(body, self.object.kernel)
        k_name = self.object.kernel.__class__.__name__
        all_args.extend(
            ['__global {kernel}* kern'.format(kernel=k_name),
             '__global unsigned int *nbr_length',
             '__global unsigned int *start_idx',
             '__global unsigned int *neighbors']
        )
        self.data.append(
            dict(kernel=kernel, args=py_args, dest=dest,
                 source=source, loop=True)
        )

        return (
            '__kernel void\n{kernel}\n({args})\n'
            '{{\n{body}\n    }}\n}}\n'.format(
                kernel=kernel, args=', '.join(all_args),
                body=body
            )
        )
