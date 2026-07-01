from ..model import Model
from ..regularization import create_regularization
from ..examples import political_democracy


def _sum_abs_residual_covariances(model):
    ins = model.inspect()
    cov = ins[(ins['op'] == '~~') & (ins['lval'] != ins['rval'])]
    return cov['Estimate'].abs().sum()


def test_l1_thresh_shrinks_residual_covariances():
    desc = political_democracy.get_model()
    data = political_democracy.get_data()

    baseline = Model(desc)
    assert baseline.fit(data, obj='MLW').success
    baseline_sum = _sum_abs_residual_covariances(baseline)

    regularized = Model(desc)
    reg = create_regularization(regularized, regularization='l1-thresh',
                                c=200.0, mx_names={'psi'})
    r = regularized.fit(data, obj='MLW', regularization=reg)
    assert r.success
    regularized_sum = _sum_abs_residual_covariances(regularized)

    assert regularized_sum < baseline_sum * 1e-4
