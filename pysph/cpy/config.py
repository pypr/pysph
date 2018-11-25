"""Simple configuration options for PySPH.

Do not import any PySPH specific extensions here, if you must, do the import
inside the function/method.
"""

from contextlib import contextmanager


class Config(object):
    def __init__(self):
        self._use_openmp = None
        self._use_opencl = None
        self._use_cuda = None
        self._use_double = None
        self._omp_schedule = None
        self._profile = None
        self._use_local_memory = None
        self._wgs = None

    @property
    def use_openmp(self):
        if self._use_openmp is None:
            self._use_openmp = self._use_openmp_default()
        return self._use_openmp

    @use_openmp.setter
    def use_openmp(self, value):
        self._use_openmp = value

    def _use_openmp_default(self):
        return False

    @property
    def omp_schedule(self):
        if self._omp_schedule is None:
            self._omp_schedule = self._omp_schedule_default()
        return self._omp_schedule

    @omp_schedule.setter
    def omp_schedule(self, value):
        if len(value) != 2 or \
                value[0].lower() not in ("static", "dynamic", "guided"):
            raise ValueError("Invalid OpenMP Schedule: {}".format(value))

        self._omp_schedule = value

    def set_omp_schedule(self, omp_schedule):
        """
        Expects input to be in the format used by OMP_SCHEDULE
        i.e. "schedule_type, chunk_size"
        """
        temp = omp_schedule.split(",")
        if len(temp) == 2:
            self.omp_schedule = (temp[0], int(temp[1]))
        else:
            self.omp_schedule = (temp[0], None)

    def _omp_schedule_default(self):
        return ("dynamic", 64)

    @property
    def use_opencl(self):
        if self._use_opencl is None:
            self._use_opencl = self._use_opencl_default()
        return self._use_opencl

    @use_opencl.setter
    def use_opencl(self, value):
        self._use_opencl = value

    def _use_opencl_default(self):
        return False

    @property
    def use_cuda(self):
        if self._use_cuda is None:
            self._use_cuda = self._use_cuda_default()
        return self._use_cuda

    @use_opencl.setter
    def use_cuda(self, value):
        self._use_cuda = value

    def _use_cuda_default(self):
        return False

    @property
    def use_double(self):
        """This is only used by OpenCL code.
        """
        if self._use_double is None:
            self._use_double = self._use_double_default()
        return self._use_double

    @use_double.setter
    def use_double(self, value):
        """This is only used by OpenCL code.
        """
        self._use_double = value

    def _use_double_default(self):
        return False

    @property
    def profile(self):
        if self._profile is None:
            self._profile = self._profile_default()
        return self._profile

    @profile.setter
    def profile(self, value):
        self._profile = value

    def _profile_default(self):
        return False

    @property
    def use_local_memory(self):
        if self._use_local_memory is None:
            self._use_local_memory = self._use_local_memory_default()
        return self._use_local_memory

    @use_local_memory.setter
    def use_local_memory(self, value):
        self._use_local_memory = value

    def _use_local_memory_default(self):
        return False

    @property
    def wgs(self):
        if self._wgs is None:
            self._wgs = self._wgs_default()
        return self._wgs

    @wgs.setter
    def wgs(self, value):
        self._wgs = value

    def _wgs_default(self):
        return 32


_config = None


def get_config():
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config):
    global _config
    _config = config


@contextmanager
def use_config(**kw):
    """A context manager for the configuration.

    One can do the following::

        with use_config(use_openmp=True) as cfg:
            do_something()
            cfg.use_opencl = True
            do_something_else()

    The configuration will be restored to the original when one exits the
    context. Inside the scope of the with statement the configuration ``cfg``
    is the one operational and so can be changed.
    """
    orig_cfg = get_config()
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg, k, v)

    set_config(cfg)

    try:
        yield cfg
    finally:
        set_config(orig_cfg)
