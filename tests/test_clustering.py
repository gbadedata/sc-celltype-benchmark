"""Tests for clustering module.

Tests neighbourhood graph construction, Leiden clustering,
and UMAP embedding using synthetic preprocessed data.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from src.clustering import (
    ClusteringReport,
    build_neighbor_graph,
    compute_umap,
    run_clustering,
    run_leiden,
)
from src.preprocessing import run_preprocessing
from src.quality_control import run_qc


@pytest.fixture
def preprocessed_adata(synthetic_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData that has been QC-filtered and preprocessed."""
    adata, _ = run_qc(synthetic_adata)
    adata, _ = run_preprocessing(adata)
    return adata


class TestBuildNeighborGraph:
    """Test kNN graph construction."""

    def test_connectivities_added(self, preprocessed_adata: ad.AnnData) -> None:
        """Connectivity matrix is added to .obsp after graph build."""
        result = build_neighbor_graph(preprocessed_adata)
        assert "connectivities" in result.obsp

    def test_distances_added(self, preprocessed_adata: ad.AnnData) -> None:
        """Distance matrix is added to .obsp after graph build."""
        result = build_neighbor_graph(preprocessed_adata)
        assert "distances" in result.obsp

    def test_neighbors_uns_added(self, preprocessed_adata: ad.AnnData) -> None:
        """Neighbor parameters are stored in .uns['neighbors']."""
        result = build_neighbor_graph(preprocessed_adata)
        assert "neighbors" in result.uns

    def test_cell_count_unchanged(self, preprocessed_adata: ad.AnnData) -> None:
        """Graph construction does not remove cells."""
        n_before = preprocessed_adata.n_obs
        result = build_neighbor_graph(preprocessed_adata)
        assert result.n_obs == n_before


class TestRunLeiden:
    """Test Leiden clustering."""

    def test_leiden_column_added(self, preprocessed_adata: ad.AnnData) -> None:
        """Leiden cluster labels added to .obs['leiden']."""
        adata = build_neighbor_graph(preprocessed_adata)
        result = run_leiden(adata)
        assert "leiden" in result.obs.columns

    def test_at_least_two_clusters(self, preprocessed_adata: ad.AnnData) -> None:
        """Leiden finds at least 2 clusters in multi-type data."""
        adata = build_neighbor_graph(preprocessed_adata)
        result = run_leiden(adata)
        n_clusters = result.obs["leiden"].nunique()
        assert n_clusters >= 2

    def test_all_cells_assigned(self, preprocessed_adata: ad.AnnData) -> None:
        """Every cell receives a cluster assignment (no NaN labels)."""
        adata = build_neighbor_graph(preprocessed_adata)
        result = run_leiden(adata)
        assert result.obs["leiden"].isna().sum() == 0

    def test_labels_are_string_categories(self, preprocessed_adata: ad.AnnData) -> None:
        """Leiden labels are categorical with string categories (scanpy convention)."""
        import pandas as pd
        adata = build_neighbor_graph(preprocessed_adata)
        result = run_leiden(adata)
        col = result.obs["leiden"]
        assert isinstance(col.dtype, pd.CategoricalDtype)
        # Underlying categories must be strings (e.g. '0', '1', '2')
        assert col.cat.categories.dtype == object


class TestComputeUmap:
    """Test UMAP embedding."""

    def test_umap_embedding_added(self, preprocessed_adata: ad.AnnData) -> None:
        """UMAP coordinates added to .obsm['X_umap']."""
        adata = build_neighbor_graph(preprocessed_adata)
        result = compute_umap(adata)
        assert "X_umap" in result.obsm

    def test_umap_shape(self, preprocessed_adata: ad.AnnData) -> None:
        """UMAP embedding has shape (n_cells, 2)."""
        adata = build_neighbor_graph(preprocessed_adata)
        result = compute_umap(adata)
        assert result.obsm["X_umap"].shape == (preprocessed_adata.n_obs, 2)

    def test_umap_values_finite(self, preprocessed_adata: ad.AnnData) -> None:
        """UMAP coordinates contain no NaN or Inf values."""
        adata = build_neighbor_graph(preprocessed_adata)
        result = compute_umap(adata)
        assert np.all(np.isfinite(result.obsm["X_umap"]))


class TestRunClustering:
    """Test the full clustering pipeline."""

    def test_returns_adata_and_report(self, preprocessed_adata: ad.AnnData) -> None:
        """run_clustering returns (AnnData, ClusteringReport)."""
        result, report = run_clustering(preprocessed_adata)
        assert isinstance(result, ad.AnnData)
        assert isinstance(report, ClusteringReport)

    def test_report_n_cells_consistent(self, preprocessed_adata: ad.AnnData) -> None:
        """Report cell count matches output AnnData."""
        adata, report = run_clustering(preprocessed_adata)
        assert report.n_cells == adata.n_obs

    def test_report_cluster_sizes_consistent(self, preprocessed_adata: ad.AnnData) -> None:
        """Min/max cluster sizes in report match actual cluster sizes."""
        adata, report = run_clustering(preprocessed_adata)
        actual_sizes = adata.obs["leiden"].value_counts()
        assert report.min_cluster_size == int(actual_sizes.min())
        assert report.max_cluster_size == int(actual_sizes.max())

    def test_report_n_clusters_positive(self, preprocessed_adata: ad.AnnData) -> None:
        """Report contains a positive cluster count."""
        _, report = run_clustering(preprocessed_adata)
        assert report.n_clusters > 0

    def test_all_embeddings_present(self, preprocessed_adata: ad.AnnData) -> None:
        """Output AnnData has both PCA and UMAP embeddings."""
        adata, _ = run_clustering(preprocessed_adata)
        assert "X_pca" in adata.obsm
        assert "X_umap" in adata.obsm
