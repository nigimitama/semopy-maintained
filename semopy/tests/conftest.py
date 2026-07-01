#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared fixtures and assertion helpers for semopy's test suite.

The synthetic two-factor model used across several test modules used to be
copy-pasted (with minor variations) at the top of each test file. It is now
built once here and exposed as fixtures so that every test file starts from
the same well-defined, reproducible data.
"""
import numpy as np
import pandas as pd
import pytest


def build_two_factor_data(n=100, p=3, seed=2021, means=False, effects=False):
    """Simulate data from a 2-factor CFA model with a regression eta2 ~ eta1.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of indicators per factor.
    seed : int
        Seed for a local (non-global) random state, so tests stay isolated
        from each other regardless of execution order.
    means : bool
        If True, adds random intercepts to the indicators.
    effects : bool
        If True, additionally adds a random-effects-like noise term ``u`` and
        returns its outer product ``k = u @ u.T`` (e.g. for ModelEffects).

    Returns
    -------
    desc : str
        Model description in semopy syntax.
    data : pd.DataFrame
        Simulated dataset.
    params : pd.DataFrame
        True parameter values in the same shape as Model.inspect() output.
    k : np.ndarray or None
        Random effects covariance matrix (only if ``effects`` is True).
    """
    rs = np.random.RandomState(seed)
    loadings = [rs.uniform(0.2, 1.2, size=(p - 1, 1)),
                rs.uniform(0.2, 1.2, size=(p - 1, 1))]
    loadings = [np.append([1], ld) for ld in loadings]

    y = rs.normal(size=(n, 2 * p))
    eta1 = rs.normal(scale=1, size=(n, 1))
    eta2 = rs.normal(scale=1, size=(n, 1)) + 3 * eta1
    y[:, :p] += np.kron(loadings[0], eta1)
    y[:, p:] += np.kron(loadings[1], eta2)

    row_means = None
    if means or effects:
        row_means = rs.normal(scale=3, size=2 * p)
        y = y + row_means

    k = None
    if effects:
        u = rs.normal(size=y.shape)
        y = y + u
        k = u @ u.T

    records = []
    y_names = []
    d = {'eta1': [], 'eta2': []}
    for i in range(1, p + 1):
        name = f'y{i}'
        y_names.append(name)
        d['eta1'].append(name)
        records.append((name, '~', 'eta1', loadings[0][i - 1]))
        if row_means is not None:
            records.append((name, '~', '1', row_means[i - 1]))
    for j in range(1, p + 1):
        name = f'y{p + j}'
        y_names.append(name)
        d['eta2'].append(name)
        records.append((name, '~', 'eta2', loadings[1][j - 1]))
        if row_means is not None:
            records.append((name, '~', '1', row_means[p + j - 1]))

    desc = '\n'.join(f"{eta} =~ {' + '.join(ys)}" for eta, ys in d.items())
    desc += '\neta2 ~ eta1'

    params = pd.DataFrame.from_records(
        records, columns=['lval', 'op', 'rval', 'Estimate'])
    data = pd.DataFrame(np.column_stack([y, eta1, eta2]),
                        columns=y_names + ['eta1', 'eta2'])
    return desc, data, params, k


@pytest.fixture
def two_factor_model():
    """Synthetic 2-factor model without an explicit mean structure."""
    desc, data, params, _ = build_two_factor_data()
    return desc, data, params


@pytest.fixture
def two_factor_model_with_means():
    """Synthetic 2-factor model with random intercepts."""
    desc, data, params, _ = build_two_factor_data(means=True)
    return desc, data, params


@pytest.fixture
def two_factor_model_with_effects():
    """Synthetic 2-factor model with intercepts and a random-effects term.

    An individual-level 'group' column is added (one observation per group),
    matching how ModelEffects is exercised elsewhere in the test suite.
    """
    desc, data, params, k = build_two_factor_data(means=True, effects=True)
    data = data.copy()
    data['group'] = data.index
    return desc, data, params, k


def assert_fit_success(result, obj=''):
    """Assert that a Model.fit() result (or tuple of results) succeeded."""
    if isinstance(result, tuple):
        assert all(r.success for r in result), \
            f"Optimization routine failed. [{obj}]"
    else:
        assert result.success, f"Optimization routine failed. [{obj}]"


def assert_estimates_close(inspection: pd.DataFrame, true: pd.DataFrame,
                           obj='', pval_thresh=0.05, max_rel_err=0.1):
    """Compare a fitted model's inspect() table against known true values.

    Checks that (1) parameters present in ``true`` are statistically
    significant (p-value below ``pval_thresh``, when a p-value is available)
    and (2) the average relative estimation error is below ``max_rel_err``.
    """
    errs = []
    for _, row in true.iterrows():
        mask = (inspection['op'] == row['op']) & \
            (inspection['lval'] == row['lval']) & \
            (inspection['rval'] == row['rval'])
        if not mask.any():
            continue
        matched = inspection[mask]
        pval = matched['p-value'].values[0]
        if pval_thresh is not None and isinstance(pval, (int, float, np.floating)) \
                and not np.isnan(pval):
            assert pval < pval_thresh, \
                f"Incorrect p-value estimate for {row['lval']} {row['op']} " \
                f"{row['rval']}: {pval} [{obj}]"
        est = matched['Estimate'].values[0]
        errs.append(abs((est - row['Estimate']) / row['Estimate']))
    err = np.mean(errs)
    assert err < max_rel_err, \
        f"Parameter estimation quality is too low: {err} [{obj}]"
