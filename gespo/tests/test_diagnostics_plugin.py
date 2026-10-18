from pathlib import Path

from gespo.training.components.endpoint_loader import load_boundary_plugin
from gespo.utils.diagnostics_plugins.adk_feature_kmeans import (
    adk_feature_kmeans_boundaries,
    plot_feature_pairs,
)


def test_legacy_adk_boundary_plugin_path_remains_loadable():
    legacy = Path(__file__).resolve().parents[1] / 'utils/path_boundary_plugins/adk_feature_kmeans.py'
    assert load_boundary_plugin(f'{legacy}:adk_feature_kmeans_boundaries') is adk_feature_kmeans_boundaries
    assert load_boundary_plugin(f'{legacy}:plot_feature_pairs') is plot_feature_pairs
