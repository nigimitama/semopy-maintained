#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

from ..model_means import Model
from ..means import estimate_means
from ..examples import univariate_regression, multivariate_regression, political_democracy
from .conftest import assert_estimates_close, assert_fit_success


def fit_and_check(desc, data, true, obj='MLW'):
    m = Model(desc)
    r = m.fit(data, obj=obj)
    assert_fit_success(r, obj)
    ins = pd.concat([m.inspect(), estimate_means(m)], axis=0)
    # estimate_means() does not provide p-values for the mean/intercept
    # terms, so only estimation accuracy is checked here.
    assert_estimates_close(ins, true, obj, pval_thresh=None)


def test_univariate_regression():
    desc = univariate_regression.get_model()
    data = univariate_regression.get_data()
    true = univariate_regression.get_params()
    fit_and_check(desc, data, true, 'MLW')


def test_multivariate_regression():
    desc = multivariate_regression.get_model()
    data = multivariate_regression.get_data()
    true = multivariate_regression.get_params()
    fit_and_check(desc, data, true, 'MLW')


def test_political_democracy():
    fit_and_check(political_democracy.get_model(),
                  political_democracy.get_data(),
                  political_democracy.get_params())


def test_random_model(two_factor_model_with_means):
    desc, data, params = two_factor_model_with_means
    fit_and_check(desc, data, params, 'MLW')
