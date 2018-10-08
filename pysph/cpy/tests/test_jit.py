from math import sin
import unittest
import numpy as np

from pytest import importorskip

from ..config import get_config, use_config
from ..array import wrap
from ..jit import get_binop_return_type, AnnotationHelper
from ..types import annotate
from ..parallel import Elementwise, Reduction, Scan


@annotate
def g(x):
    return x


@annotate
def h(a, b):
    return g(a) * g(b)


class TestAnnotationHelper(unittest.TestCase):
    def test_const_as_call_arg(self):
        # Given
        @annotate
        def int_f(a):
            return g(1)

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(int_f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

        # Given
        @annotate
        def long_f(a):
            return g(10000000000)

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(long_f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'long'

        # Given
        @annotate
        def double_f(a):
            return g(1.)

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(double_f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'double'

    def test_variable_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            x = declare('int')
            x = a + b
            return g(x)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_subscript_as_call_arg(self):
        # Given
        @annotate
        def f(i, a):
            return g(a[i])

        # When
        types = {'i': 'int', 'a': 'intp'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_binop_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(a + b)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_compare_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(a == b)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_call_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(h(a, b))

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_binop_with_call_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(h(a, b) + h(b, a))

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'int'

    def test_non_jit_call_as_call_arg(self):
        # Given
        @annotate
        def f(a, b):
            return g(sin(a))

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.external_funcs['g'].arg_types['x'] == 'double'

    def test_variable_in_return(self):
        # Given
        @annotate
        def f(a):
            return a

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

    def test_subscript_in_return(self):
        # Given
        @annotate
        def f(i, a):
            return a[i]

        # When
        types = {'i': 'int', 'a': 'intp'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

    def test_const_in_return(self):
        # Given
        @annotate
        def int_f(a, b):
            return 1

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(int_f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

        # Given
        @annotate
        def long_f(a, b):
            return 10000000000

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(long_f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'long'

        # Given
        @annotate
        def double_f(a, b):
            return 1.

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(double_f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'double'

    def test_binop_in_return(self):
        # Given
        @annotate
        def f(a, b):
            return a + b

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

    def test_call_in_return(self):
        # Given
        @annotate
        def f(a, b):
            return g(a)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert 'g' in helper.external_funcs
        assert helper.arg_types['return_'] == 'int'

    def test_binop_with_call_in_return(self):
        # Given
        @annotate
        def f(a, b):
            return g(a) + g(b)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

    def test_multi_level_call_in_return(self):
        # Given
        @annotate
        def f(a, b):
            return h(a, b)

        # When
        types = {'a': 'int', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert 'h' in helper.external_funcs
        assert 'g' in helper.external_funcs['h'].external_funcs
        assert helper.arg_types['return_'] == 'int'

    def test_non_jit_call_in_return(self):
        # Given
        @annotate
        def f(a):
            return sin(a)

        # When
        types = {'a': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'double'

    def test_binop_return_type(self):
        # Given
        @annotate
        def f(a, b):
            return a + b

        # When
        types = {'a': 'long', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'long'

        # When
        types = {'a': 'int', 'b': 'double'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'double'

        # When
        types = {'a': 'uint', 'b': 'int'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'int'

        # When
        types = {'a': 'uint', 'b': 'ulong'}
        helper = AnnotationHelper(f, types)
        helper.annotate()

        # Then
        assert helper.arg_types['return_'] == 'ulong'
