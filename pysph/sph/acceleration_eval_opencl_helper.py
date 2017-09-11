"""
TODO:

Basic support:
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
import pyopencl as cl
import pyopencl.array  # noqa: 401

from pysph.base.opencl import get_context, get_queue, DeviceHelper
from pysph.base.translator import (CStructHelper, OpenCLConverter,
                                   ocl_detect_type)

from .acceleration_eval_cython_helper import (
    get_all_array_names, get_known_types_for_arrays
)


class OpenCLAccelerationEval(object):
    """Does the actual work of performing the evaluation.
    """
    def __init__(self, helper):
        self.helper = helper
        self.nnps = None

    def compute(self, t, dt):
        helper = self.helper
        nnps = self.nnps
        for call, args, loop_info in helper.calls:
            if loop_info[0]:
                nnps.set_context(loop_info[1], loop_info[2])
                cache = nnps.current_cache
                cache.get_neighbors_gpu()
                args = list(args) + [
                    cache._nbr_lengths_gpu.data,
                    cache._start_idx_gpu.data,
                    cache._neighbors_gpu.data
                ]
                call(*args)
            else:
                call(*args)

    def set_nnps(self, nnps):
        self.nnps = nnps

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
        self._ctx = get_context()
        self._queue = get_queue()
        self._array_map = None
        self._array_index = None
        self._cpu_structs = {}
        self._gpu_structs = {}
        self.calls = []
        self.program = None

    def _setup_arrays_on_device(self):
        pas = self.object.particle_arrays
        array_map = {}
        array_index = {}
        for idx, pa in enumerate(pas):
            pa.set_device_helper(DeviceHelper(pa))
            array_map[pa.name] = pa
            array_index[pa.name] = idx

        self._array_map = array_map
        self._array_index = array_index

        gpu = self._gpu_structs
        cpu = self._cpu_structs
        for k, v in cpu.items():
            if v is None:
                gpu[k] = v
            else:
                gpu[k] = cl.array.to_device(self._queue, v)

    def _get_argument(self, arg, dest, src=None):
        ary_map = self._array_map
        structs = self._gpu_structs
        if arg.startswith('d_'):
            return getattr(ary_map[dest].gpu, arg[2:]).data
        elif arg.startswith('s_'):
            return getattr(ary_map[src].gpu, arg[2:]).data
        else:
            return structs[arg].data

    def _setup_calls(self):
        calls = []
        prg = self.program
        array_index = self._array_index
        for info in self.data:
            method = getattr(prg, info.get('kernel'))
            dest = info['dest']
            src = info.get('source', None)
            np = self._array_map[dest].get_number_of_particles()
            args = [self._queue, (np,), None]
            for arg in info['args']:
                args.append(self._get_argument(arg, dest, src))
            loop = info['loop']
            if loop:
                loop_info = (loop, array_index[src], array_index[dest])
                args.append(self._get_argument('kern', dest, src))
            else:
                loop_info = (loop, None, None)
            calls.append((method, args, loop_info))
        return calls

    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_code(self):
        path = join(dirname(__file__), 'acceleration_eval_opencl.mako')
        template = Template(filename=path)
        main = template.render(helper=self)
        return main

    def setup_compiled_module(self, module):
        object = self.object
        self._setup_arrays_on_device()
        self.calls = self._setup_calls()
        acceleration_eval = OpenCLAccelerationEval(self)
        object.set_compiled_object(acceleration_eval)

    def compile(self, code):
        self.program = cl.Program(self._ctx, code.encode('ascii')).build()
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

        # This is to be done after the above as the equation names are assigned
        # only at this point.
        cpu_structs = self._cpu_structs
        h = CStructHelper(object.kernel)
        cpu_structs['kern'] = h.get_array()
        for eq in object.all_group.equations:
            h.parse(eq)
            cpu_structs[eq.var_name] = h.get_array()

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
        code.append('int start = start_idx[d_idx];')
        code.append('int end = start + nbr_length[d_idx];')
        code.append('for (i=start; i<end; i++) {')
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
