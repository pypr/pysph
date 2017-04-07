#!/usr/bin/env python

from pyopencl.elementwise import ElementwiseKernel
from mako.template import Template
import os

class GPUNNPSHelper(object):
    def __init__(self, ctx, tpl_filename, use_double=True):
        self.src_tpl = Template(filename = \
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    tpl_filename), disable_unicode=True)

        self.data_t = "double" if use_double else "float"
        self.preamble = self.src_tpl.get_def("preamble").render()
        self.ctx = ctx

    def _get_code(self, kernel_name, **kwargs):
        arguments = self.src_tpl.get_def("%s_args" % kernel_name).render(
                data_t=self.data_t, **kwargs)

        src = self.src_tpl.get_def("%s_src" % kernel_name).render(
                data_t=self.data_t, **kwargs)

        return arguments, src

    def get_kernel(self, kernel_name, **kwargs):
        args, src = self._get_code(kernel_name, **kwargs)
        knl = ElementwiseKernel(self.ctx, args, src,
                kernel_name, preamble=self.preamble)
        return knl

