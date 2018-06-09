
from pyopencl.elementwise import ElementwiseKernel
from mako.template import Template
import os
import sys

from pysph.base.opencl import profile_kernel, get_elwise_kernel, \
    get_simple_kernel


class GPUNNPSHelper(object):
    def __init__(self, ctx, tpl_filename, use_double=False, c_type=None):
        """

        Parameters
        ----------
        ctx
        tpl_filename
        use_double
        c_type:
            c_type to use. Overrides use_double
        """
        disable_unicode = False if sys.version_info.major > 2 else True
        self.src_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                tpl_filename),
            disable_unicode=disable_unicode
        )

        self.data_t = "double" if use_double else "float"

        if c_type is not None:
            self.data_t = c_type

        helper_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "gpu_helper_functions.mako"),
            disable_unicode=disable_unicode
        )

        helper_preamble = helper_tpl.get_def("get_helpers").render(
            data_t=self.data_t
        )
        preamble = self.src_tpl.get_def("preamble").render(
            data_t=self.data_t
        )
        self.preamble = "\n".join([helper_preamble, preamble])
        self.ctx = ctx
        self.cache = {}

    def _get_code(self, kernel_name, **kwargs):
        arguments = self.src_tpl.get_def("%s_args" % kernel_name).render(
            data_t=self.data_t, **kwargs)

        src = self.src_tpl.get_def("%s_src" % kernel_name).render(
            data_t=self.data_t, **kwargs)

        return arguments, src

    def get_kernel(self, kernel_name, **kwargs):
        key = kernel_name, tuple(kwargs.items())
        wgs = kwargs.get('wgs', None)

        if key in self.cache:
            return self.cache[key]
        else:
            args, src = self._get_code(kernel_name, **kwargs)

            if wgs is None:
                knl = get_elwise_kernel(kernel_name, args, src,
                                        preamble=self.preamble)
            else:
                knl = get_simple_kernel(kernel_name, args, src, wgs,
                                        preamble=self.preamble)

            self.cache[key] = knl
            return knl
