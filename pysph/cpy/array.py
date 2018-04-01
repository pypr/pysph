from pysph.base.config import get_config


def get_backend(backend=None):
    if not backend:
        cfg = get_config()
        if cfg.use_opencl:
            return 'opencl'
        else:
            return 'cython'
    else:
        return backend


def wrap(*args, **kw):
    '''
    Parameters
    ----------

    *args: any numpy arrays to be wrapped.

    **kw: only one keyword arg called `backend` is supported.

    backend: str: use appropriate backend for arrays.
    '''
    backend = get_backend(kw.get('backend'))
    return [Array(x, backend=backend) for x in args]


class Array(object):
    """A simple wrapper for numpy arrays.

    It has two attributes,

    `data` is the raw numpy array.

    `dev` is the device array if needed.

    Use the `pull()` method to get the data from device.
    Use `push()` to push the data to the device.

    """
    def __init__(self, ary, backend=None):
        self.backend = get_backend(backend)
        self.data = ary
        if self.backend == 'opencl':
            from pyopencl.array import to_device
            from pysph.base.opencl import get_queue
            self.q = get_queue()
            self.dev = to_device(self.q, self.data)
        else:
            self.dev = self.data

    def pull(self):
        if self.backend == 'opencl':
            self.data = self.dev.get()

    def push(self):
        if self.backend == 'opencl':
            self.dev.set(self.data)
