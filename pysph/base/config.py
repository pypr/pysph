"""Simple configuration options for PySPH.

Do not import any PySPH specific extensions here, if you must, do the import
inside the function/method.
"""


class Config(object):
    def __init__(self):
        self._use_openmp = None
        self._use_opencl = None
        self._use_double = None
        self._profile = None

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


_config = None


def get_config():
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config):
    global _config
    _config = config
