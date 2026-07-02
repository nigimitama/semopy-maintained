#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import combinations

import numpy as np

from ..model import Model
from ..model_means import ModelMeans
from ..model_effects import ModelEffects
from ..examples import multivariate_regression
from .conftest import assert_fit_success


def std_errors(m, information):
    ins = m.inspect(information=information)
    ins = ins[ins['op'] == '~'].sort_values(['lval', 'rval'], axis=0)
    return ins['Std. Err'].values


def test_multivariate_regression():
    desc = multivariate_regression.get_model()
    data = multivariate_regression.get_data() - multivariate_regression.get_data().mean()
    data['group'] = 1

    model = Model(desc)
    model_means = ModelMeans(desc, intercepts=False)
    model_effects = ModelEffects(desc, intercepts=False)

    assert_fit_success(model.fit(data))
    assert_fit_success(model_means.fit(data))
    assert_fit_success(model_effects.fit(data, group='group'))

    se = [std_errors(model, 'expected'), std_errors(model, 'observed'),
          std_errors(model_means, 'expected'), std_errors(model_means, 'observed'),
          std_errors(model_effects, 'expected'), std_errors(model_effects, 'observed')]

    for a, b in combinations(se, 2):
        both_negligible = (a < 1e-4) & (b < 1e-4)
        rel_diff = np.max(np.abs(a - b) / b * both_negligible)
        assert rel_diff < 1e-2, f"Standard errors diverge {rel_diff:.3f}."
