#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ..model_effects import ModelEffects
from ..examples import multivariate_regression
from .conftest import assert_estimates_close, assert_fit_success


def fit_and_check(desc, data, true, obj='ML', k=None):
    m = ModelEffects(desc)
    r = m.fit(data, obj=obj, group='group', k=k)
    assert_fit_success(r, obj)
    assert_estimates_close(m.inspect(), true, obj)


def test_multivariate_regression():
    desc = multivariate_regression.get_model()
    data = multivariate_regression.get_data().copy()
    data['group'] = 1
    true = multivariate_regression.get_params()
    fit_and_check(desc, data, true, 'ML')


def test_random_model(two_factor_model_with_effects):
    desc, data, params, k = two_factor_model_with_effects
    fit_and_check(desc, data, params, 'ML', k)
