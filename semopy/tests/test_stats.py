import numpy as np
import pytest
from ..model import Model
from ..model_means import ModelMeans
from ..stats import calc_stats
from ..examples import political_democracy, holzinger39

# Each entry is a (model description, data, reference values) triple where
# the reference values were read off an actual `summary(fit, fit.measures =
# TRUE)` run in R/lavaan for the exact same dataset/model (see the module
# docstring below for the equivalent lavaan code). Add a new tuple here to
# extend the cross-check to another example.
LAVAAN_REFERENCE_CASES = {
    'political_democracy': dict(
        desc=political_democracy.get_model(),
        data=political_democracy.get_data(),
        dof=35, dof_baseline=55,
        loglik=-1547.791, aic=3157.582, bic=3229.424,
    ),
    'holzinger_swineford_1939': dict(
        desc=holzinger39.get_model(),
        data=holzinger39.get_data(),
        dof=24, dof_baseline=36,
        loglik=-3737.745, aic=7517.490, bic=7595.339,
    ),
}


@pytest.mark.parametrize('case', LAVAAN_REFERENCE_CASES.values(),
                         ids=LAVAAN_REFERENCE_CASES.keys())
def test_calc_stats_matches_lavaan(case):
    m = Model(case['desc'])
    m.fit(case['data'])
    stats = calc_stats(m).loc['Value']

    assert stats['DoF'] == case['dof']
    assert stats['DoF Baseline'] == case['dof_baseline']
    assert 0 < stats['chi2 p-value'] <= 1
    assert stats['chi2'] > 0
    assert stats['chi2 Baseline'] > stats['chi2']

    for index in ('CFI', 'GFI', 'AGFI', 'NFI', 'TLI'):
        assert 0 <= stats[index] <= 1.01
    assert stats['RMSEA'] >= 0

    assert stats['LogLik'] == pytest.approx(case['loglik'], abs=1e-2)
    assert stats['AIC'] == pytest.approx(case['aic'], abs=1e-2)
    assert stats['BIC'] == pytest.approx(case['bic'], abs=1e-2)


def test_calc_stats_non_mlw_objective_yields_nan_aic_bic():
    # AIC/BIC/LogLik require the 'MLW' objective (as in lavaan, which only
    # reports them for ML estimation). ModelMeans always fits with 'FIML'
    # internally, so these should come back as NaN rather than a silently
    # wrong number, while the rest of the fit indices still compute fine.
    m = ModelMeans(political_democracy.get_model())
    m.fit(political_democracy.get_data())
    stats = calc_stats(m).loc['Value']

    assert np.isnan(stats['LogLik'])
    assert np.isnan(stats['AIC'])
    assert np.isnan(stats['BIC'])
    assert stats['DoF'] > 0
    assert stats['chi2'] > 0
