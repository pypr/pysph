"""Simple configuration options for PySPH.

Do not import any PySPH specific extensions here, if you must, do the import
inside the function/method.
"""

import sys


class Config(object):
    def __init__(self):
        self._use_openmp = None

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


_config = None

def get_config():
    global _config
    if _config is None:
        _config = Config()
    return _config

def set_config(config):
    global _config
    _config = config
