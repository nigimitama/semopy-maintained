from ..model import Model
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
