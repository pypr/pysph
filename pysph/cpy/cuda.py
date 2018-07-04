"""Common CUDA related functionality.
"""
from __future__ import print_function
from collections import defaultdict
from operator import itemgetter

from .config import get_config

_cuda_ctx = False


def set_context():
    global _cuda_ctx
    if not _cuda_ctx:
        import pycuda.autoinit
        _cuda_ctx = True


