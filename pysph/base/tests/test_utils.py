from unittest import TestCase, main

from ..utils import is_overloaded_method


class TestUtils(TestCase):
    def test_is_overloaded_method_works_for_simple_overloads(self):
        # Given
        class A(object):
            def f(self): pass

        class B(A):
            pass

        # When/Then
        b = B()
        self.assertFalse(is_overloaded_method(b.f))

        class C(A):
            def f(self): pass

        # When/Then
        c = C()
        self.assertTrue(is_overloaded_method(c.f))

    def test_is_overloaded_method_works_for_parent_overloads(self):
        # Given
        class A(object):
            def f(self): pass

        class B(A):
            def f(self): pass

        class C(B):
            pass

        # When/Then
        c = C()
        self.assertTrue(is_overloaded_method(c.f))


if __name__ == '__main__':
    main()
