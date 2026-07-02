from ..model import Model
from ..efa import explore_cfa_model
from ..examples import political_democracy

INDICATORS = ['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4',
              'y5', 'y6', 'y7', 'y8']


def test_explore_cfa_model_discovers_valid_structure():
    data = political_democracy.get_data()[INDICATORS]
    min_loadings = 2

    # mode='optics' avoids SparsePCA's non-deterministic random_state,
    # keeping this test reproducible.
    desc = explore_cfa_model(data, min_loadings=min_loadings, mode='optics')

    assert '=~' in desc
    for line in desc.splitlines():
        indicators = line.split('=~')[1].split('+')
        assert len(indicators) >= min_loadings

    m = Model(desc)
    assert m.fit(data).success
