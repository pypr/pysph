from textwrap import dedent

from pysph.base.translator import py2c


def test_simple_assignment_expression():
    # Given
    src = dedent('''
    b = (2*a + 1)*(a/1.5)%2
    ''')

    # When
    code = py2c(src)

    # Then
    expect = dedent('''
    double a;
    double b;
    b = ((((2 * a) + 1) * (a / 1.5)) % 2);
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
