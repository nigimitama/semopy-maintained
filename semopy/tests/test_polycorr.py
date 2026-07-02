import numpy as np
import pandas as pd
import pytest

from ..polycorr import bivariate_cdf, polyserial_corr, polychoric_corr, hetcor


def test_bivariate_cdf():
    assert bivariate_cdf(lower=[-100, -100], upper=[100, 100], corr=1) == \
        pytest.approx(1, abs=1e-3)


def _simulate_correlated(rho, n=3000, seed=0):
    rs = np.random.RandomState(seed)
    cov = [[1, rho], [rho, 1]]
    return rs.multivariate_normal([0, 0], cov, size=n)


@pytest.mark.parametrize('rho', [-0.6, 0.0, 0.5])
def test_polyserial_corr_recovers_true_correlation(rho):
    xy = _simulate_correlated(rho)
    x, y_cont = xy[:, 0], xy[:, 1]
    y_ord = pd.qcut(y_cont, q=4, labels=False).astype(float)
    est = polyserial_corr(x, y_ord)
    assert abs(est - rho) < 0.1


@pytest.mark.parametrize('rho', [-0.6, 0.3])
def test_polychoric_corr_recovers_true_correlation(rho):
    xy = _simulate_correlated(rho)
    x_ord = pd.qcut(xy[:, 0], q=4, labels=False).astype(float)
    y_ord = pd.qcut(xy[:, 1], q=4, labels=False).astype(float)
    est = polychoric_corr(x_ord, y_ord)
    assert abs(est - rho) < 0.1


def test_hetcor_recovers_true_correlation_dataframe():
    rho = 0.5
    xy = _simulate_correlated(rho)
    data = pd.DataFrame({'x_cont': xy[:, 0],
                        'y_ord': pd.qcut(xy[:, 1], q=4, labels=False)})
    cor = hetcor(data, ords=['y_ord'])
    assert cor.loc['x_cont', 'x_cont'] == pytest.approx(1.0, abs=1e-6)
    assert cor.loc['x_cont', 'y_ord'] == pytest.approx(rho, abs=0.1)
    assert cor.loc['y_ord', 'x_cont'] == cor.loc['x_cont', 'y_ord']


def test_hetcor_recovers_true_correlation_ndarray():
    rho = 0.5
    xy = _simulate_correlated(rho)
    xy[:, 1] = pd.qcut(xy[:, 1], q=4, labels=False)
    cor = hetcor(xy, ords=[1])
    assert cor[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert cor[0, 1] == pytest.approx(rho, abs=0.1)
    assert cor[0, 1] == cor[1, 0]
