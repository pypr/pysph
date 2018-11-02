from textwrap import dedent
import pytest
import numpy as np
import sys

from ..config import get_config
from ..types import annotate, declare
from ..translator import (
    CConverter, CodeGenerationError, CStructHelper, KnownType,
    OpenCLConverter, CUDAConverter, py2c
)


@annotate(i='int', y='floatp', return_='float')
def annotated_f(i, y):
    x = declare('LOCAL_MEM matrix(64)')
    return y[i]


def test_simple_assignment_expression():
    # Given
    src = dedent('''
    b = (2*a + 1)*(-a/1.5)%2
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double a;
    double b;
    b = ((((2 * a) + 1) * (-a / 1.5)) % 2);
    ''')
    assert code == expect.strip()


def test_multiple_assignment_expressions():
    # Given
    src = dedent('''
    a = 21.5
    b = (2*a + 1)*(a/1.5)%2
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double a;
    double b;
    a = 21.5;
    b = ((((2 * a) + 1) * (a / 1.5)) % 2);
    ''')
    assert code == expect.strip()


def test_if_block():
    # Given
    src = dedent('''
    a = 21.5
    if a > 20:
        b = a - 1
    elif a < 20:
        b = a + 1
    else:
        b = a
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double a;
    double b;
    a = 21.5;
    if ((a > 20)) {
        b = (a - 1);
    }
    else {
        if ((a < 20)) {
            b = (a + 1);
        }
        else {
            b = a;
        }
    }
    ''')
    assert code.strip() == expect.strip()


def test_conditionals():
    # Given
    src = dedent('''
    if (x > 10 and x < 20) or not (x >= 10 and x <= 20):
        y
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    double y;
    if ((((x > 10) && (x < 20)) || !((x >= 10) && (x <= 20)))) {
        y;
    }
    ''')
    assert code.strip() == expect.strip()

    # Given
    src = dedent('''
    if x != 10 and x is 100 or (x == 20 and x is not 1):
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    if ((((x != 10) && (x == 100)) || ((x == 20) && (x != 1)))) {
        ;
    }
    ''')
    assert code.strip() == expect.strip()

    # Given
    src = dedent('''
    if x != 10 and x is 100 or (x == 20 and x is not 1):
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    if ((((x != 10) && (x == 100)) || ((x == 20) && (x != 1)))) {
        ;
    }
    ''')
    assert code.strip() == expect.strip()


def test_ternary_operator():
    # Given
    src = dedent('''
    y = 2.0
    x = 1.0 if y >= 2.0 else 0.0
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    double y;
    y = 2.0;
    x = (y >= 2.0) ? 1.0 : 0.0;
    ''')
    assert code.strip() == expect.strip()


def test_multiple_boolops():
    # Given
    src = dedent('''
    if x % 2 == 0 or x % 2 == 1 or x > 0:
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    if ((((x % 2) == 0) || ((x % 2) == 1) || (x > 0))) {
        ;
    }
    ''')
    assert code.strip() == expect.strip()


def test_power():
    # Given
    src = dedent('''
    1.5*x**2
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    (1.5 * pow(x, 2));
    ''')
    assert code.strip() == expect.strip()


def test_only_two_operands_supported_for_comparisons():
    # Given
    src = dedent('''
    if 10 < x < 20:
        pass
    ''')

    # When
    with pytest.raises(NotImplementedError):
        py2c(src)


def test_calling_function():
    # Given
    src = dedent('''
    sin(23.2 + 1)
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    sin((23.2 + 1));
    ''')
    assert code == expect.strip()


def test_calling_printf_with_string():
    # Given
    src = dedent(r'''
    printf('%s %d %f\n', 'hello', 1, 2.0)
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    printf("%s %d %f\n", "hello", 1, 2.0);
    ''')
    assert code == expect.strip()


def test_subscript():
    # Given
    src = dedent('''
    x[1]
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    x[1];
    ''')
    assert code == expect.strip()


def test_known_math_constants():
    # Given
    src = dedent('''
    x = M_E + M_LOG2E + M_LOG10E + M_LN2 + M_LN10
    x += M_PI + M_PI_2 + M_PI_4 + M_1_PI * M_2_PI
    x += M_2_SQRTPI * M_SQRT2 * M_SQRT1_2 * pi
    x = INFINITY
    x = NAN
    x = HUGE_VALF
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    x = ((((M_E + M_LOG2E) + M_LOG10E) + M_LN2) + M_LN10);
    x += (((M_PI + M_PI_2) + M_PI_4) + (M_1_PI * M_2_PI));
    x += (((M_2_SQRTPI * M_SQRT2) * M_SQRT1_2) * pi);
    x = INFINITY;
    x = NAN;
    x = HUGE_VALF;
    ''')
    assert code == expect.strip()


def test_simple_function_with_return():
    # Given
    src = dedent('''
    def f(x=0.0):
        'docstring'
        y = x + 1
        return y
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double f(double x)
    {
        double y;
        y = (x + 1);
        return y;
    }
    ''')
    assert code.strip() == expect.strip()


def test_simple_function_without_return():
    # Given
    src = dedent('''
    def f(y=0.0, x=0.0):
        z = y + x
        y = z
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    void f(double y, double x)
    {
        double z;
        z = (y + x);
        y = z;
    }
    ''')
    assert code.strip() == expect.strip()


def test_function_argument_types():
    # Given
    src = dedent('''
    def f(s_idx, s_p, d_idx, d_p, J=0, t=0.0, l=[0,0], xx=(0, 0)):
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
void f(long s_idx, double* s_p, long d_idx, double* d_p, long J, double t,
    double* l, double* xx)
{
    ;
}
    ''')
    assert code.strip() == expect.strip()


def test_known_types_in_funcargs():
    # Given
    src = dedent('''
    def f(x, xx, cond=True):
        pass
    ''')

    # When
    known_types = {'xx': KnownType('foo*'), 'x': KnownType('float32')}
    code = py2c(src, known_types=known_types)

    # Then
    expect = dedent('''
    void f(float32 x, foo* xx, int cond)
    {
        ;
    }
    ''')
    assert code.strip() == expect.strip()


def test_annotated_function():
    # Given/When
    t = CConverter()
    code = t.parse_function(annotated_f)

    # Then
    expect = dedent('''
    float annotated_f(int i, float* y)
    {
        LOCAL_MEM double x[64];
        return y[i];
    }
    ''')
    assert code.strip() == expect.strip()


@pytest.mark.skipif(sys.version_info < (3, 4), reason='Requires Python3')
def test_py3_annotations():
    # Given/When
    from .py3_code import py3_f
    t = CConverter()
    code = t.parse_function(py3_f)

    # Then
    expect = dedent('''
    int py3_f(int x)
    {
        int y;
        y = (x + 1);
        return (x * y);
    }
    ''')
    assert code.strip() == expect.strip()


def test_calling_method_of_known_type():
    # Given
    src = dedent('''
    obj.method(1, 2)
    obj.meth()
    ''')
    known = {'obj': KnownType('SomeClass*', base_type='SomeClass')}

    # When
    code = py2c(src, known_types=known)

    # Then
    expect = dedent('''
    SomeClass_method(obj, 1, 2);
    SomeClass_meth(obj);
    ''')
    assert code.strip() == expect.strip()


def test_calling_method_of_known_type_in_method():
    # Given
    src = dedent('''
    class Foo(object):
        def g(self):
            pass
        def f(self, obj):
            obj.method(1, 2)
            self.g()
    ''')

    # When
    known = {'obj': KnownType('SomeClass*', base_type='SomeClass')}
    code = py2c(src, known_types=known)

    # Then
    expect = dedent('''
    void Foo_g(Foo* self)
    {
        ;
    }

    void Foo_f(Foo* self, SomeClass* obj)
    {
        SomeClass_method(obj, 1, 2);
        Foo_g(self);
    }
    ''')
    assert code.strip() == expect.strip()


def test_raises_error_when_unknown_args_are_given():
    # Given
    src = dedent('''
    def f(x):
        pass
    ''')

    # When/Then
    with pytest.raises(CodeGenerationError):
        py2c(src)

    # Given
    # Unsupported default arg.
    src = dedent('''
    def f(x=''):
        pass
    ''')

    # When/Then
    with pytest.raises(CodeGenerationError):
        py2c(src)

    # Given
    # Unsupported default arg list.
    src = dedent('''
    def f(x=(1, '')):
        pass
    ''')

    # When/Then
    with pytest.raises(CodeGenerationError):
        py2c(src)


def test_user_supplied_detect_type():
    # Given
    src = dedent('''
    def f(x, xx=[1,2,3], cond=True):
        pass
    ''')

    # When
    def dt(name, value):
        return 'double'
    code = py2c(src, detect_type=dt)

    # Then
    expect = dedent('''
    void f(double x, double xx, double cond)
    {
        ;
    }
    ''')
    assert code.strip() == expect.strip()


def test_while():
    # Given
    src = dedent('''
    while x < 21:
        do(x)
        do1(x)
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    while ((x < 21)) {
        do(x);
        do1(x);
    }
    ''')
    assert code.strip() == expect.strip()


def test_bool_true_false_and_none():
    # Given
    src = dedent('''
    while True:
        pass
    if False:
        pass
    if x is None or x is not None:
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x;
    while (1) {
        ;
    }

    if (0) {
        ;
    }

    if (((x == NULL) || (x != NULL))) {
        ;
    }
    ''')
    assert code.strip() == expect.strip()


def test_for():
    # Given
    src = dedent('''
    for i in range(5):
        do(i)
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    for (long i=0; i<5; i+=1) {
        do(i);
    }
    ''')
    assert code.strip() == expect.strip()

    # Given
    src = dedent('''
    for i in range(2, 5):
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    for (long i=2; i<5; i+=1) {
        ;
    }
    ''')
    assert code.strip() == expect.strip()

    # Given
    src = dedent('''
    for i in range(2, 10, 2):
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    for (long i=2; i<10; i+=2) {
        ;
    }
    ''')
    assert code.strip() == expect.strip()


def test_for_with_decreasing_range():
    # Given
    src = dedent('''
    for i in range(10, -1, -1):
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    for (long i=10; i>-1; i+=-1) {
        ;
    }
    ''')
    assert code.strip() == expect.strip()


def test_for_with_declare():
    # Given
    src = dedent('''
    i = declare('int')
    for i in range(5):
        do(i)
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    int i;
    for (i=0; i<5; i+=1) {
        do(i);
    }
    ''')
    assert code.strip() == expect.strip()


def test_two_fors():
    # Given
    src = dedent('''
    for i in range(5):
        do(i)
    for i in range(5):
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    for (long i=0; i<5; i+=1) {
        do(i);
    }

    for (long i=0; i<5; i+=1) {
        ;
    }
    ''')
    assert code.strip() == expect.strip()


def test_for_with_symbols():
    # Given
    src = dedent('''
    n = declare('int')
    n = 25
    for i in range(n):
        pass
    for i in range(0, n+1, step()):
        pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    int n;
    n = 25;
    long __cpy_stop_0 = n;
    for (long i=0; i<__cpy_stop_0; i+=1) {
        ;
    }

    __cpy_stop_0 = (n + 1);
    long __cpy_step_0 = step();
    if (__cpy_step_0 < 0) {
        for (long i=0; i>__cpy_stop_0; i+=__cpy_step_0) {
            ;
        }
    }
    else {
        for (long i=0; i<__cpy_stop_0; i+=__cpy_step_0) {
            ;
        }
    }
    ''')
    assert code.strip() == expect.strip()


def test_nested_for_with_symbols():
    # Given
    src = dedent('''
    n = declare('int')
    n = 25
    for i in range(n):
        for j in range(0, n+1, step()):
            pass
    for i in range(n+1):
        for j in range(0, n+2, step()):
            pass
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    int n;
    n = 25;
    long __cpy_stop_0 = n;
    for (long i=0; i<__cpy_stop_0; i+=1) {
        long __cpy_stop_1 = (n + 1);
        long __cpy_step_1 = step();
        if (__cpy_step_1 < 0) {
            for (long j=0; j>__cpy_stop_1; j+=__cpy_step_1) {
                ;
            }
        }
        else {
            for (long j=0; j<__cpy_stop_1; j+=__cpy_step_1) {
                ;
            }
        }
    }

    __cpy_stop_0 = (n + 1);
    for (long i=0; i<__cpy_stop_0; i+=1) {
        long __cpy_stop_1 = (n + 2);
        long __cpy_step_1 = step();
        if (__cpy_step_1 < 0) {
            for (long j=0; j>__cpy_stop_1; j+=__cpy_step_1) {
                ;
            }
        }
        else {
            for (long j=0; j<__cpy_stop_1; j+=__cpy_step_1) {
                ;
            }
        }
    }
    ''')
    assert code.strip() == expect.strip()


def test_for_with_break_continue():
    # Given
    src = dedent('''
    for i in range(10):
        if i%7 == 0:
            break
        if i%2 == 0:
            continue
        do(i)
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    for (long i=0; i<10; i+=1) {
        if (((i % 7) == 0)) {
            break;
        }
        if (((i % 2) == 0)) {
            continue;
        }
        do(i);
    }
    ''')
    assert code.strip() == expect.strip()


def test_for_not_range_and_else_fails():
    # Given
    src = dedent('''
    for i in something():
        pass
    ''')

    # When/Then
    with pytest.raises(NotImplementedError):
        py2c(src)

    # Given
    src = dedent('''
    for i in range(5):
        pass
    else:
        pass
    ''')

    # When/Then
    with pytest.raises(NotImplementedError):
        py2c(src)

    # Given
    src = dedent('''
    for i in range(0, 5, 2, 3):
        pass
    ''')

    # When/Then
    with pytest.raises(NotImplementedError):
        py2c(src)


def test_while_else_raises_error():
    # Given
    src = dedent('''
    while 1:
        do()
    else:
        do()
    ''')

    # When/Then
    with pytest.raises(NotImplementedError):
        py2c(src)


def test_try_block_raises_error():
    # Given
    src = dedent('''
    try:
        do()
    except ImportError:
        pass
    ''')

    # When/Then
    with pytest.raises(NotImplementedError):
        py2c(src)


def test_attribute_access():
    # Given
    src = dedent('''
    self.x = 1
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double self;
    self->x = 1;
    ''')

    assert code.strip() == expect.strip()


def test_declare_call_declares_variable():
    # Given
    src = dedent('''
    x = declare('int')
    x += 1
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    int x;
    x += 1;
    ''')
    assert code.strip() == expect.strip()


def test_declare_matrix():
    # Given
    src = dedent('''
    x = declare('matrix((3,))')
    do(x[0])
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x[3];
    do(x[0]);
    ''')
    assert code.strip() == expect.strip()

    # Given
    src = dedent('''
    x = declare('matrix((2, 3))')
    do(x[0][1])
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double x[2][3];
    do(x[0][1]);
    ''')
    assert code.strip() == expect.strip()

    # Given
    src = dedent('''
    x = declare('matrix((2, 3), "int")')
    do(x[0][1])
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    int x[2][3];
    do(x[0][1]);
    ''')
    assert code.strip() == expect.strip()


def test_declare_call_declares_multiple_variables():
    # Given
    src = dedent('''
    x, y = declare('int', 2)
    u, v = declare('matrix(3)', 2)
    A = declare('matrix((2,2), "long")')
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    int x, y;
    double u[3], v[3];
    long A[2][2];
    ''')
    assert code.strip() == expect.strip()


def test_class():
    # Given
    src = dedent('''
    class Foo(object):
        def g(self, x=0.0):
            return x*2.0
        def f(self, x=0.0):
            y = x + 1
            do(self.a, x)
            z = self.g(y)
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double Foo_g(Foo* self, double x)
    {
        return (x * 2.0);
    }

    void Foo_f(Foo* self, double x)
    {
        double y;
        double z;
        y = (x + 1);
        do(self->a, x);
        z = Foo_g(self, y);
    }
    ''')
    assert code.strip() == expect.strip()


def test_unsupported_method():
    # Given
    src = dedent('''
    np.identity(25)
    ''')

    # When
    with pytest.raises(NotImplementedError):
        py2c(src)


def test_c_struct_helper():
    # Given
    class Fruit(object):
        pass

    f = Fruit()
    f.apple = 1
    f.banana = 2.0
    f.pear = 1.5
    h = CStructHelper(f)

    # When
    result = h.get_code()

    # Then
    expect = dedent('''
    typedef struct Fruit {
        int apple;
        double banana;
        double pear;
    } Fruit;
    ''')
    assert result.strip() == expect.strip()

    # When/Then
    array = h.get_array()
    use_double = get_config().use_double
    fdtype = np.float64 if use_double else np.float32
    expect = np.dtype([('apple', np.int32),
                       ('banana', fdtype), ('pear', fdtype)])

    assert array.dtype == expect
    assert array['apple'] == 1
    assert array['banana'] == 2.0
    assert array['pear'] == 1.5


def test_c_struct_helper_empty_object():
    # Given
    class Fruit(object):
        pass

    f = Fruit()
    h = CStructHelper(f)

    # When
    result = h.get_code()

    # Then
    expect = dedent('''
    typedef struct Fruit {
    } Fruit;
    ''')
    assert result.strip() == expect.strip()

    # When/Then
    assert h.get_array() is None


def test_wrapping_class():
    # Given
    class Dummy(object):
        '''Class Docstring'''
        def __init__(self, x=0, f=0.0, s=''):
            "Constructor docstring"
            self.x = x
            self.f = f
            self.s = s
            self._private = 1

        def method(self):
            '''Method docstring.
            '''
            pass

    obj = Dummy()

    # When
    c = CConverter()
    result = c.parse_instance(obj)

    # Then
    expect = dedent('''
    typedef struct Dummy {
        double f;
        int x;
    } Dummy;


    void Dummy_method(Dummy* self)
    {
        ;
    }
    ''')
    assert result.strip() == expect.strip()

    # When
    h = CStructHelper(obj)
    use_double = get_config().use_double
    fdtype = np.float64 if use_double else np.float32
    dtype = np.dtype([('f', fdtype), ('x', np.int32)])
    expect = np.zeros(1, dtype)
    assert h.get_array() == expect


def test_wrapping_class_with_ignore_methods():
    # Given
    class Dummy1(object):
        '''Class Docstring'''
        def f(self):
            pass

        def not_me(self):
            pass

    obj = Dummy1()

    # When
    c = CConverter()
    result = c.parse_instance(obj, ignore_methods=['not_me'])

    # Then
    expect = dedent('''
    typedef struct Dummy1 {
    } Dummy1;

    void Dummy1_f(Dummy1* self)
    {
        ;
    }
    ''')
    assert result.strip() == expect.strip()


def check_opencl_cuda_conversion(converter_obj):
    # Note that LID_0 etc. are predefined symbols when we include the CLUDA
    # preamble, therefore should be known.
    src = dedent('''
    def f(s_idx, s_p, d_idx, d_p, J=0, t=0.0, l=[0,0], xx=(0, 0)):
        s_p[s_idx] = LID_0*GID_0
    ''')

    # When
    known_types = {'d_p': KnownType('GLOBAL_MEM int*')}
    converter = converter_obj(known_types=known_types)
    code = converter.convert(src)

    # Then
    expect = dedent('''
WITHIN_KERNEL void f(long s_idx, GLOBAL_MEM double* s_p, long d_idx,
    GLOBAL_MEM int* d_p, long J, double t, double* l, double* xx)
{
    s_p[s_idx] = (LID_0 * GID_0);
}
    ''')
    assert code.strip() == expect.strip()


def test_cuda_conversion():
    check_opencl_cuda_conversion(CUDAConverter)


def test_opencl_conversion():
    check_opencl_cuda_conversion(OpenCLConverter)


def test_opencl_class():
    src = dedent('''
    class Foo(object):
        def g(self, x=0.0):
            pass
    ''')

    # When
    converter = OpenCLConverter()
    code = converter.convert(src)

    # Then
    expect = dedent('''
    WITHIN_KERNEL void Foo_g(GLOBAL_MEM Foo* self, double x)
    {
        ;
    }
    ''')
    assert code.strip() == expect.strip()


def test_cuda_local_conversion():
    @annotate(xc='ldoublep', yc='lintp')
    def knl(xc, yc):
        xc[LID_0] = 1
        yc[LID_0] = 1

    # When
    converter = CUDAConverter()
    code = converter.parse(knl)

    # Then
    expect_1 = dedent('''
WITHIN_KERNEL void knl(int size_xc, int size_yc)
{
    extern LOCAL_MEM float shared_buff[];
    double* xc = (double*) shared_buff;
    int* yc = (int*) &xc[size_xc];
    xc[LID_0] = 1;
    yc[LID_0] = 1;
}
    ''')

    expect_2 = dedent('''
WITHIN_KERNEL void knl(int size_xc, int size_yc)
{
    extern LOCAL_MEM float shared_buff[];
    int* yc = (int*) shared_buff;
    double* xc = (double*) &yc[size_yc];
    xc[LID_0] = 1;
    yc[LID_0] = 1;
}
    ''')

    assert code.strip() == expect_1.strip() or code.strip() == expect_2.strip()


def test_handles_parsing_functions():
    # Given
    def f(x=1.0):
        return x + 1

    # When
    t = CConverter()
    code = t.parse_function(f)

    # Then
    expect = dedent('''
    double f(double x)
    {
        return (x + 1);
    }
    ''')
    assert code.strip() == expect.strip()

    # Given
    class A(object):
        def f(self, x=1.0):
            return x + 1.0

    # When
    t = CConverter()
    code = t.parse_function(A)

    # Then
    expect = dedent('''
    double A_f(A* self, double x)
    {
        return (x + 1.0);
    }
    ''')
    assert code.strip() == expect.strip()
