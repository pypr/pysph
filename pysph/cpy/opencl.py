"""Common OpenCL related functionality.
"""
from __future__ import print_function
import pyopencl as cl
from collections import defaultdict
from operator import itemgetter

from .config import get_config

_ctx = None
_queue = None
_profile_info = defaultdict(float)


class DeviceWGSException(Exception):
    pass


def get_context():
    global _ctx
    if _ctx is None:
        _ctx = cl.create_some_context()
    return _ctx


def set_context(ctx):
    global _ctx
    _ctx = ctx


def get_queue():
    global _queue
    if _queue is None:
        properties = None
        if get_config().profile:
            properties = cl.command_queue_properties.PROFILING_ENABLE
        _queue = cl.CommandQueue(get_context(), properties=properties)
    return _queue


def set_queue(q):
    global _queue
    _queue = q


def profile(name, event):
    global _profile_info
    event.wait()
    time = (event.profile.end - event.profile.start) * 1e-9
    _profile_info[name] += time


def print_profile():
    global _profile_info
    _profile_info = sorted(_profile_info.items(), key=itemgetter(1),
                           reverse=True)
    if len(_profile_info) == 0:
        print("No profile information available")
        return
    print("{:<30} {:<30}".format('Kernel', 'Time'))
    tot_time = 0
    for kernel, time in _profile_info:
        print("{:<30} {:<30}".format(kernel, time))
        tot_time += time
    print("Total profiled time: %g secs" % tot_time)


def profile_kernel(kernel, name):
    def _profile_knl(*args, **kwargs):
        event = kernel(*args, **kwargs)
        profile(name, event)
        return event

    if get_config().profile:
        return _profile_knl
    else:
        return kernel


def named_profile(name):
    def _decorator(f):
        if name is None:
            n = f.__name__
        else:
            n = name

        def _profiled_kernel_generator(*args, **kwargs):
            kernel = f(*args, **kwargs)
            return profile_kernel(kernel, n)

        return _profiled_kernel_generator

    return _decorator


class SimpleKernel(object):
    """ElementwiseKernel substitute that supports a custom work group size.
    """

    def __init__(self, ctx, args, operation, wgs,
                 name="", preamble="", options=[]):
        self.args = args
        self.operation = operation
        self.name = name
        self.preamble = preamble
        self.options = options

        self.prg = cl.Program(ctx, self._generate()).build(options)
        self.knl = getattr(self.prg, name)

        if self.get_max_wgs() < wgs:
            raise DeviceWGSException("")

    def _massage_arg(self, arg):
        if '*' in arg:
            return "__global " + arg
        return arg

    def _generate(self):
        args = [self._massage_arg(arg) for arg in self.args.split(",")]

        source = r"""
        %(preamble)s

        __kernel void %(name)s(%(args)s)
        {
          int lid = get_local_id(0);
          int gsize = get_global_size(0);
          int work_group_start = get_local_size(0)*get_group_id(0);
          long i = get_global_id(0);

          %(body)s
        }
        """ % {
            "args": ",".join(args),
            "name": self.name,
            "preamble": self.preamble,
            "body": self.operation
        }

        return source

    def get_max_wgs(self):
        return self.knl.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE,
            get_queue().device
        )

    def __call__(self, *args, **kwargs):
        wait_for = kwargs.pop("wait_for", None)
        queue = kwargs.pop("queue", None)
        gs = kwargs.pop("gs", None)
        ls = kwargs.pop("ls", None)

        if queue is None or gs is None or ls is None:
            raise ValueError("queue, gs and ls can not be empty")

        if kwargs:
            raise TypeError("unknown keyword arguments: '%s'"
                            % ", ".join(kwargs))

        def unwrap(arg):
            return arg.data if isinstance(arg, cl.array.Array) else arg

        self.knl.set_args(*[unwrap(arg) for arg in args])
        return cl.enqueue_nd_range_kernel(queue, self.knl, gs, ls,
                                          wait_for=wait_for)
