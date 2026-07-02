#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from ..model import Model
from ..examples import univariate_regression, multivariate_regression, political_democracy
from .conftest import assert_estimates_close, assert_fit_success

CONTINUOUS_OBJECTIVES = ('MLW', 'ULS', 'GLS')


def fit_and_check(desc, data, true, obj, **kwargs):
    m = Model(desc)
    r = m.fit(data, obj=obj)
    assert_fit_success(r, obj)
    assert_estimates_close(m.inspect(), true, obj, **kwargs)


@pytest.mark.parametrize('obj', CONTINUOUS_OBJECTIVES)
def test_univariate_regression(obj):
    desc = univariate_regression.get_model()
    data = univariate_regression.get_data()
    true = univariate_regression.get_params()
    fit_and_check(desc, data, true, obj)


def test_univariate_regression_fiml():
    desc = univariate_regression.get_model()
    data = univariate_regression.get_data()
    true = univariate_regression.get_params()
    fit_and_check(desc, data - data.mean(), true, 'FIML')


@pytest.mark.parametrize('obj', CONTINUOUS_OBJECTIVES)
def test_multivariate_regression(obj):
    desc = multivariate_regression.get_model()
    data = multivariate_regression.get_data()
    true = multivariate_regression.get_params()
    fit_and_check(desc, data, true, obj)


def test_multivariate_regression_fiml():
    desc = multivariate_regression.get_model()
    data = multivariate_regression.get_data()
    true = multivariate_regression.get_params()
    fit_and_check(desc, data - data.mean(), true, 'FIML')


@pytest.mark.parametrize('obj', CONTINUOUS_OBJECTIVES)
def test_political_democracy(obj):
    pdem = political_democracy
    model_desc, pdem_data, ref_params = pdem.get_model(), pdem.get_data(), pdem.get_params()
    # p-values are not checked here: some residual covariances in this
    # particular model are not expected to be significant.
    fit_and_check(model_desc, pdem_data, ref_params, obj,
                  pval_thresh=None, max_rel_err=.3)


@pytest.mark.parametrize('obj', CONTINUOUS_OBJECTIVES)
def test_random_model(obj, two_factor_model):
    desc, data, params = two_factor_model
    fit_and_check(desc, data, params, obj)


def test_random_model_fiml(two_factor_model):
    desc, data, params = two_factor_model
    fit_and_check(desc, data - data.mean(), params, 'FIML')


@pytest.mark.parametrize('obj', ('WLS', 'DWLS'))
def test_random_model_wls(obj, two_factor_model):
    """WLS/DWLS estimation on discretized (ordinal-like) indicators."""
    desc, data, params = two_factor_model
    ordinal_data = data.copy()
    for col in ('y1', 'y2', 'y3', 'y4', 'y5', 'y6'):
        ordinal_data[col] = pd.qcut(data[col], q=5, labels=False, duplicates='drop')
    m = Model(desc)
    r = m.fit(ordinal_data.drop(columns=['eta1', 'eta2']), obj=obj)
    assert_fit_success(r, obj)
    # Discretization is lossy, so only check that the sign/rough magnitude
    # of loadings and the regression path is recovered, not tight precision.
    assert_estimates_close(m.inspect(), params, obj, pval_thresh=None, max_rel_err=0.6)
