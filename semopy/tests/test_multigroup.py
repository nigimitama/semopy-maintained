from ..multigroup import multigroup
from ..examples import political_democracy


def test_multigroup_political_democracy():
    desc = political_democracy.get_model()
    data = political_democracy.get_data().copy()
    data['group'] = ['a' if i % 2 == 0 else 'b' for i in range(len(data))]

    res = multigroup(desc, data, 'group')

    assert set(res.groups) == {'a', 'b'}
    assert res.n_obs['a'] + res.n_obs['b'] == len(data)
    for g in res.groups:
        assert res.runs[g].success
        assert not res.estimates[g].empty
        assert res.stats[g].loc['Value', 'DoF'] == 35

    # __str__ should not raise and should mention both groups.
    report = str(res)
    assert 'a' in report and 'b' in report
