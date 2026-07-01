#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Model.predict(), semopy's conditional-expectation imputation.

Note: the separate, SLSQP-optimization-based Imputer/predict_general path in
imputer.py has its own pre-existing issues under recent pandas/numpy versions
(read-only arrays returned by DataFrame.values) that go beyond fixing a
single call site, so it is not covered here.
"""
import numpy as np

from ..model import Model
from ..examples import multivariate_regression


def test_predict_imputes_missing_values_close_to_truth():
    desc = multivariate_regression.get_model()
    data = multivariate_regression.get_data()
    data = data - data.mean()

    m = Model(desc)
    assert m.fit(data).success

    col = 'y1'
    rows = data.index[:10]
    true_vals = data.loc[rows, col].copy()

    x = data.copy()
    x.loc[rows, col] = np.nan

    imputed = m.predict(x)

    assert not imputed.loc[rows, col].isna().any()
    err = np.abs(imputed.loc[rows, col].values - true_vals.values)
    # A model-based prediction should beat the naive "predict the mean"
    # baseline, whose expected absolute error is the column's std.
    assert err.mean() < data[col].std()


def test_predict_returns_input_unchanged_when_nothing_missing():
    desc = multivariate_regression.get_model()
    data = multivariate_regression.get_data()
    data = data - data.mean()

    m = Model(desc)
    assert m.fit(data).success

    result = m.predict(data)
    obs = m.vars['observed']
    assert np.allclose(result[obs].values, data[obs].values)
