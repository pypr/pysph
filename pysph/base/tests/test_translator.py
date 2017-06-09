from textwrap import dedent
import pytest

from pysph.base.translator import py2c


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
    b = ((((2 * a) + 1) * ((- a) / 1.5)) % 2);
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
    if ((((x > 10) && (x < 20)) || (! ((x >= 10) && (x <= 20))))) {
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
    (1.5 * pow(x, 2))
    ''')


def test_only_two_operands_supported_for_comparisons():
    # Given
    src = dedent('''
    if 10 < x < 20:
        pass
    ''')

    # When
    with pytest.raises(NotImplementedError):
        code = py2c(src)


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


def test_simple_function_with_return():
    # Given
    src = dedent('''
    def f(x):
        return x+1
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double f(double x) {
        return (x + 1);
    }
    ''')
    assert code == expect.strip()


def test_simple_function_without_return():
    # Given
    src = dedent('''
    def f(y, x):
        y += x
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    void f(double y, double x) {
        y += x;
    }
    ''')
    assert code == expect.strip()


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
        code = py2c(src)

    # Given
    src = dedent('''
    for i in range(5):
        pass
    else:
        pass
    ''')

    # When/Then
    with pytest.raises(NotImplementedError):
        code = py2c(src)

    # Given
    src = dedent('''
    for i in range(0, 5, 2, 3):
        pass
    ''')

    # When/Then
    with pytest.raises(NotImplementedError):
        code = py2c(src)


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
        code = py2c(src)


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
        code = py2c(src)


def test_attribute():
    # Given
    src = dedent('''
    self.x = 1
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double self;
    self->x = 1
    ''')
