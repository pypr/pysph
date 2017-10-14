from functools import partial
import inspect
import os
import re
import sys
from textwrap import wrap

import numpy as np
from mako.template import Template
import pyopencl as cl
import pyopencl.array  # noqa: 401
import pyopencl.tools  # noqa: 401

from pysph.base.config import get_config
from pysph.base.utils import is_overloaded_method
from pysph.base.ext_module import get_platform_dir
from pysph.base.opencl import (profile_kernel, get_context, get_queue,
                               DeviceHelper)
from pysph.base.translator import (CStructHelper, OpenCLConverter,
                                   ocl_detect_type)

from .equation import get_predefined_types
from .acceleration_eval_cython_helper import (
    get_all_array_names, get_known_types_for_arrays
)


def get_kernel_definition(kernel, arg_list):
    sig = '__kernel void\n{kernel}\n({args})'.format(
        kernel=kernel, args=', '.join(arg_list),
    )
    return '\n'.join(wrap(sig, width=78, subsequent_indent=' ' * 4,
                          break_long_words=False))


def wrap_code(code, indent=' ' * 4):
    return wrap(
        code, width=74, initial_indent=indent,
        subsequent_indent=indent + ' ' * 4, break_long_words=False
    )


def get_code(obj, transpiler=None):
    """This function looks at the object and gets any additional code to
    wrap from the `_get_helpers_` method.
    """
    result = []
    if hasattr(obj, '_get_helpers_'):
        if transpiler is None:
            transpiler = OpenCLConverter()
        doc = '\n// Helpers from %s' % obj.__class__.__name__
        result.append(doc)
        for helper in obj._get_helpers_():
            result.append(transpiler.parse_function(helper))
    return result


class OpenCLAccelerationEval(object):
    """Does the actual work of performing the evaluation.
    """

    def __init__(self, helper):
        self.helper = helper
        self.particle_arrays = helper.object.particle_arrays
        self.nnps = None
        self._queue = helper._queue
        self._use_double = get_config().use_double

    def _call_kernel(self, info, extra_args):
        nnps = self.nnps
        call = info.get('method')
        args = list(info.get('args'))
        dest = info['dest']
        n = dest.get_number_of_particles(info.get('real', True))
        args[1] = (n,)
        args[3:] = [x() for x in args[3:]]
        if info.get('loop'):
            nnps.set_context(info['src_idx'], info['dst_idx'])
            cache = nnps.current_cache
            cache.get_neighbors_gpu()
            self._queue.finish()
            args = args + [
                cache._nbr_lengths_gpu.array.data,
                cache._start_idx_gpu.array.data,
                cache._neighbors_gpu.array.data
            ] + extra_args
            call(*args)
        else:
            call(*(args + extra_args))
        self._queue.finish()

    def _sync_from_gpu(self, eq):
        ary = eq._gpu.get()
        for i, name in enumerate(ary.dtype.names):
            setattr(eq, name, ary[0][i])

    def _converged(self, equations):
        for eq in equations:
            self._sync_from_gpu(eq)
            if not (eq.converged() > 0):
                return False
        return True

    def compute(self, t, dt):
        helper = self.helper
        dtype = np.float64 if self._use_double else np.float32
        extra_args = [np.asarray(t, dtype=dtype), np.asarray(dt, dtype=dtype)]
        i = 0
        iter_count = 0
        iter_start = 0
        while i < len(helper.calls):
            info = helper.calls[i]
            type = info['type']
            if type == 'method':
                method = getattr(self, info.get('method'))
                method(*info.get('args'))
            elif type == 'kernel':
                self._call_kernel(info, extra_args)
            elif type == 'start_iteration':
                iter_count = 0
                iter_start = i
            elif type == 'stop_iteration':
                eqs = info['equations']
                group = info['group']
                iter_count += 1
                if ((iter_count >= group.min_iterations) and
                    (iter_count == group.max_iterations or
                     self._converged(eqs))):
                    pass
                else:
                    i = iter_start
            i += 1

    def set_nnps(self, nnps):
        self.nnps = nnps

    def update_particle_arrays(self, arrays):
        raise NotImplementedError('OpenCL backend is incomplete')

    def update_nnps(self):
        self.nnps.update_domain()
        self.nnps.update()

    def do_reduce(self, eqs, dest):
        for eq in eqs:
            eq.reduce(dest)


def add_address_space(known_types):
    for v in known_types.values():
        if '__global' not in v.type:
            v.type = '__global ' + v.type


def get_equations_with_converged(group):
    def _get_eqs(g):
        if g.has_subgroups:
            res = []
            for x in g.equations:
                res.extend(_get_eqs(x))
            return res
        else:
            return g.equations
    eqs = [x for x in _get_eqs(group)
           if is_overloaded_method(getattr(x, 'converged'))]
    return eqs


def convert_to_float_if_needed(code):
    use_double = get_config().use_double
    if not use_double:
        code = re.sub(r'\bdouble\b', 'float', code)
    return code


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
        predefined = dict(get_predefined_types(
            self.object.all_group.pre_comp
        ))
        self.known_types.update(predefined)
        self.data = []
        self._ctx = get_context()
        self._queue = get_queue()
        self._array_map = None
        self._array_index = None
        self._equations = {}
        self._cpu_structs = {}
        self._gpu_structs = {}
        self.calls = []
        self.program = None

    def _setup_arrays_on_device(self):
        pas = self.object.particle_arrays
        array_map = {}
        array_index = {}
        for idx, pa in enumerate(pas):
            if pa.gpu is None:
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
                g_struct, code = cl.tools.match_dtype_to_c_struct(
                    self._ctx.devices[0], "dummy", v.dtype
                )
                g_v = v.astype(g_struct)
                gpu[k] = cl.array.to_device(self._queue, g_v)
                if k in self._equations:
                    self._equations[k]._gpu = gpu[k]

    def _get_argument(self, arg, dest, src=None):
        ary_map = self._array_map
        structs = self._gpu_structs

        # This is needed for late binding on the device helper's attributes
        # which may change at each iteration when particles are added/removed.
        def _get_array(gpu_helper, attr):
            return getattr(gpu_helper, attr).data

        def _get_struct(obj):
            return obj

        if arg.startswith('d_'):
            return partial(_get_array, ary_map[dest].gpu, arg[2:])
        elif arg.startswith('s_'):
            return partial(_get_array, ary_map[src].gpu, arg[2:])
        else:
            return partial(_get_struct, structs[arg].data)

    def _setup_calls(self):
        calls = []
        prg = self.program
        array_index = self._array_index
        for item in self.data:
            type = item.get('type')
            if type == 'kernel':
                kernel = item.get('kernel')
                method = getattr(prg, kernel)
                method = profile_kernel(method, method.function_name)
                dest = item['dest']
                src = item.get('source', dest)
                args = [self._queue, None, None]
                for arg in item['args']:
                    args.append(self._get_argument(arg, dest, src))
                loop = item['loop']
                if loop:
                    args.append(self._get_argument('kern', dest, src))
                info = dict(
                    method=method, dest=self._array_map[dest],
                    src=self._array_map[src], args=args,
                    loop=loop, src_idx=array_index[src],
                    dst_idx=array_index[dest], type='kernel'
                )
            elif type == 'method':
                info = dict(item)
                if info.get('method') == 'do_reduce':
                    args = info.get('args')
                    grp = args[0]
                    args[0] = [x for x in grp.equations
                               if hasattr(x, 'reduce')]
                    args[1] = self._array_map[args[1]]

            elif 'iteration' in type:
                group = item['group']
                equations = get_equations_with_converged(group._orig_group)
                info = dict(type=type, equations=equations, group=group)
            else:
                raise RuntimeError('Unknown type %s' % type)
            calls.append(info)
        return calls

    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_code(self):
        path = os.path.join(os.path.dirname(__file__),
                            'acceleration_eval_opencl.mako')
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
        code = convert_to_float_if_needed(code)
        path = os.path.expanduser(os.path.join(
            '~', '.pysph', 'source', get_platform_dir()
        ))
        if not os.path.exists(path):
            os.makedirs(path)
        fname = os.path.join(path, 'generated.cl')
        with open(fname, 'w') as fp:
            fp.write(code)
            print("OpenCL code written to %s" % fname)
        code = code.encode('ascii') if sys.version_info.major < 3 else code
        self.program = cl.Program(self._ctx, code).build(
            options=['-w']
        )
        return self.program

    ##########################################################################
    # Mako interface.
    ##########################################################################
    def get_header(self):
        object = self.object

        transpiler = OpenCLConverter(known_types=self.known_types)

        headers = []
        headers.extend(get_code(object.kernel, transpiler))
        for equation in object.all_group.equations:
            headers.extend(get_code(equation, transpiler))

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
            self._equations[eq.var_name] = eq
            h.parse(eq)
            cpu_structs[eq.var_name] = h.get_array()

        return '\n'.join(headers)

    def _get_typed_args(self, args):
        code = []
        for arg in args:
            type = ocl_detect_type(arg, self.known_types.get(arg))
            code.append('{type} {arg}'.format(
                type=type, arg=arg
            ))

        return code

    def _clean_kernel_args(self, args):
        remove = ('d_idx', 's_idx')
        for a in remove:
            if a in args:
                args.remove(a)

    def _get_simple_kernel(self, g_idx, sg_idx, group, dest, all_eqs, kind):
        assert kind in ('initialize', 'post_loop', 'loop')
        sub_grp = '' if sg_idx == -1 else 's{idx}'.format(idx=sg_idx)
        kernel = 'g{g_idx}{sub}_{dest}_{kind}'.format(
            g_idx=g_idx, sub=sub_grp, dest=dest, kind=kind
        )
        all_args = []
        py_args = []
        code = [
            'int d_idx = get_global_id(0);'
        ]
        for eq in all_eqs.equations:
            method = getattr(eq, kind, None)
            if method is not None:
                cls = eq.__class__.__name__
                arg = '__global {cls}* {name}'.format(
                    cls=cls, name=eq.var_name
                )
                all_args.append(arg)
                py_args.append(eq.var_name)

                call_args = list(inspect.getargspec(method).args)
                if 'self' in call_args:
                    call_args.remove('self')
                call_args.insert(0, eq.var_name)
                code.extend(
                    wrap_code(
                        '{cls}_{kind}({args});'.format(
                            cls=cls, kind=kind, args=', '.join(call_args)
                        ),
                        indent=''
                    )
                )
        s_ary, d_ary = all_eqs.get_array_names()
        # We only need the dest arrays here as these are simple kernels
        # without a loop so there is no "source".
        _args = list(d_ary)
        py_args.extend(_args)
        all_args.extend(self._get_typed_args(_args + ['t', 'dt']))

        body = '\n'.join([' ' * 4 + x for x in code])
        self.data.append(dict(
            kernel=kernel, args=py_args, dest=dest, loop=False,
            real=group.real, type='kernel'
        ))

        sig = get_kernel_definition(kernel, all_args)
        return (
            '{sig}\n{{\n{body}\n}}'.format(
                sig=sig, body=body
            )
        )

    def _declare_precomp_vars(self, context):
        decl = []
        names = sorted(context.keys())
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
                    'double {var}[{size}];'.format(
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
            deltap = '%s_get_deltap(kern)' % name

            code = code.replace('DELTAP', deltap).replace('GRADIENT(', grad)
            return code.replace('KERNEL(', kern).replace('GRADH(', grad_h)
        else:
            return code

    def call_reduce(self, all_eq_group, dest):
        self.data.append(dict(method='do_reduce', type='method',
                              args=[all_eq_group, dest]))

    def call_update_nnps(self, group):
        self.data.append(dict(method='update_nnps',
                              type='method', args=[]))

    def get_initialize_kernel(self, g_idx, sg_idx, group, dest, all_eqs):
        return self._get_simple_kernel(
            g_idx, sg_idx, group, dest, all_eqs, kind='initialize'
        )

    def get_simple_loop_kernel(self, g_idx, sg_idx, group, dest, all_eqs):
        return self._get_simple_kernel(
            g_idx, sg_idx, group, dest, all_eqs, kind='loop'
        )

    def get_post_loop_kernel(self, g_idx, sg_idx, group, dest, all_eqs):
        return self._get_simple_kernel(
            g_idx, sg_idx, group, dest, all_eqs, kind='post_loop'
        )

    def get_loop_kernel(self, g_idx, sg_idx, group, dest, source, eq_group):
        kind = 'loop'
        sub_grp = '' if sg_idx == -1 else 's{idx}'.format(idx=sg_idx)
        kernel = 'g{g_idx}{sg}_{source}_on_{dest}_loop'.format(
            g_idx=g_idx, sg=sub_grp, source=source, dest=dest
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
            pre.extend([' ' * 4 + x + ';' for x in src])
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
                call_args = list(inspect.getargspec(method).args)
                if 'self' in call_args:
                    call_args.remove('self')
                call_args.insert(0, eq.var_name)

                code.extend(
                    wrap_code(
                        '{cls}_{kind}({args});'.format(
                            cls=cls, kind=kind, args=', '.join(call_args)
                        ),
                        indent='    '
                    )
                )

        s_ary, d_ary = eq_group.get_array_names()
        s_ary.update(d_ary)
        _args = list(s_ary)
        py_args.extend(_args)
        all_args.extend(self._get_typed_args(_args))

        body = '\n'.join([' ' * 4 + x for x in code])
        body = self._set_kernel(body, self.object.kernel)
        k_name = self.object.kernel.__class__.__name__
        all_args.extend(
            ['__global {kernel}* kern'.format(kernel=k_name),
             '__global unsigned int *nbr_length',
             '__global unsigned int *start_idx',
             '__global unsigned int *neighbors',
             'double t', 'double dt']
        )
        self.data.append(dict(
            kernel=kernel, args=py_args, dest=dest, source=source, loop=True,
            real=group.real, type='kernel'
        ))

        sig = get_kernel_definition(kernel, all_args)
        return (
            '{sig}\n{{\n{body}\n    }}\n}}\n'.format(
                sig=sig, body=body
            )
        )

    def start_iteration(self, group):
        self.data.append(dict(
            type='start_iteration', group=group
        ))

    def stop_iteration(self, group):
        self.data.append(dict(
            type='stop_iteration', group=group,
        ))
