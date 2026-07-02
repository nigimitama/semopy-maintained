#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil

import pytest

from ..model import Model
from ..plot import semplot

try:
    import graphviz  # noqa: F401
    _GRAPHVIZ_AVAILABLE = True
except ModuleNotFoundError:
    _GRAPHVIZ_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _GRAPHVIZ_AVAILABLE or shutil.which('dot') is None,
    reason="Graphviz Python package or 'dot' executable is not installed."
)


@pytest.fixture
def fitted_model(two_factor_model):
    desc, data, _ = two_factor_model
    m = Model(desc)
    m.fit(data)
    return m


def test_semplot_default(fitted_model, tmp_path):
    g = semplot(fitted_model, str(tmp_path / 'model.png'))
    assert 'fontname' not in g.source


def test_semplot_fontname(fitted_model, tmp_path):
    g = semplot(fitted_model, str(tmp_path / 'model.png'),
                fontname='Noto Sans CJK JP')
    assert 'fontname=' in g.source
    assert 'Noto Sans CJK JP' in g.source
    # Applies to the graph itself, as well as node and edge defaults.
    assert g.source.count('Noto Sans CJK JP') >= 3


def test_semplot_node_and_edge_attrs(fitted_model, tmp_path):
    g = semplot(fitted_model, str(tmp_path / 'model.png'),
                node_attrs={'fontname': 'IPAGothic'},
                edge_attrs={'fontname': 'IPAGothic'})
    assert g.source.count('IPAGothic') >= 2


def test_semplot_node_attrs_override_fontname(fitted_model, tmp_path):
    g = semplot(fitted_model, str(tmp_path / 'model.png'),
                fontname='Noto Sans CJK JP',
                node_attrs={'fontname': 'IPAGothic'})
    assert 'IPAGothic' in g.source
