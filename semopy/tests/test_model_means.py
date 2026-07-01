#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest

from ..model_means import ModelMeans
from ..examples import univariate_regression, multivariate_regression
from .conftest import assert_estimates_close, assert_fit_success

OBJECTIVES = ('ML', 'REML')


def fit_and_check(desc, data, true, obj='ML'):
    m = ModelMeans(desc)
    r = m.fit(data, obj=obj)
    assert_fit_success(r, obj)
    assert_estimates_close(m.inspect(), true, obj)


@pytest.mark.parametrize('obj', OBJECTIVES)
def test_univariate_regression(obj):
    desc = univariate_regression.get_model()
    data = univariate_regression.get_data()
    true = univariate_regression.get_params()
    fit_and_check(desc, data, true, obj)


@pytest.mark.parametrize('obj', OBJECTIVES)
def test_multivariate_regression(obj):
    desc = multivariate_regression.get_model()
    data = multivariate_regression.get_data()
    true = multivariate_regression.get_params()
    fit_and_check(desc, data, true, obj)


@pytest.mark.parametrize('obj', OBJECTIVES)
def test_random_model(obj, two_factor_model_with_means):
    desc, data, params = two_factor_model_with_means
    fit_and_check(desc, data, params, obj)
