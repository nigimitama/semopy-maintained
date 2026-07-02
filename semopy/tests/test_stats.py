import numpy as np
import pytest
from ..model import Model
from ..model_means import ModelMeans
from ..stats import calc_stats
from ..examples import political_democracy


def test_calc_stats_political_democracy():
    m = Model(political_democracy.get_model())
    m.fit(political_democracy.get_data())
    stats = calc_stats(m).loc['Value']

    # These are the well-known reference values for this classic (Bollen
    # 1989) model/dataset, as reproduced e.g. by lavaan.
    assert stats['DoF'] == 35
    assert stats['DoF Baseline'] == 55
    assert 0 < stats['chi2 p-value'] <= 1
    assert stats['chi2'] > 0
    assert stats['chi2 Baseline'] > stats['chi2']

    for index in ('CFI', 'GFI', 'AGFI', 'NFI', 'TLI'):
        assert 0 <= stats[index] <= 1.01

    assert stats['RMSEA'] >= 0

    # lavaan reports LogLik = -1547.791 and AIC = 3157.582 for this exact
    # model/dataset (Loglikelihood user model (H0) / Akaike in `summary()`).
    assert stats['LogLik'] == pytest.approx(-1547.791, abs=1e-2)
    assert stats['AIC'] == pytest.approx(3157.582, abs=1e-2)
    assert stats['BIC'] == pytest.approx(3229.424, abs=1e-2)


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
