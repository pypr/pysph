# Python3 specific code for some tests.

from pysph.cpy.types import int_, declare


def py3_f(x: int_) -> int_:
    y = declare('int')
    y = x + 1
    return x*y
