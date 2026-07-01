import pytest

from ..constraints import parse_constraint


def test_inequality_greater_than():
    c = parse_constraint('a > 0', ['a', 'b'])
    assert c['type'] == 'ineq'
    assert c['fun']([2, -100]) == pytest.approx(2)
    assert c['jac']([2, -100]) == pytest.approx([1, 0])


def test_inequality_less_than():
    c = parse_constraint('a < 5', ['a', 'b'])
    assert c['type'] == 'ineq'
    # scipy's 'ineq' convention is satisfied when fun(x) >= 0.
    assert c['fun']([1, 999]) == pytest.approx(4)
    assert c['fun']([10, 999]) == pytest.approx(-5)


def test_equality():
    c = parse_constraint('a = b', ['a', 'b'])
    assert c['type'] == 'eq'
    assert c['fun']([2, 2]) == pytest.approx(0)
    assert c['fun']([2, 3]) == pytest.approx(-1)
    assert c['jac']([2, 3]) == pytest.approx([1, -1])


def test_missing_operator_raises_syntax_error():
    with pytest.raises(SyntaxError):
        parse_constraint('a + b', ['a', 'b'])


def test_unknown_parameter_raises():
    with pytest.raises(Exception):
        parse_constraint('c > 0', ['a', 'b'])
