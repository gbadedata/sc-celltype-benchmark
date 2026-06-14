"""Tests for markers module.

Tests Wilcoxon rank-sum marker detection, top-gene extraction,
and canonical marker overlap scoring using synthetic clustered data.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from src.clustering import run_clustering
from src.markers import (
    PBMC_MARKERS,
    detect_markers,
    get_top_markers,
    score_marker_overlap,
)
from src.preprocessing import run_preprocessing
from src.quality_control import run_qc


@pytest.fixture
def clustered_adata(synthetic_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData that has been QC-filtered, preprocessed and clustered."""
    adata, _ = run_qc(synthetic_adata)
    adata, _ = run_preprocessing(adata)
    adata, _ = run_clustering(adata)
    return adata


class TestPbmcMarkers:
    """Validate the canonical PBMC_MARKERS dictionary."""

    def test_has_expected_cell_types(self) -> None:
        """PBMC_MARKERS contains the 10 key immune cell types."""
        expected = {
            "CD4+ T cells", "CD8+ T cells", "NK cells", "B cells",
            "CD14+ Monocytes", "FCGR3A+ Monocytes", "Dendritic cells",
            "Plasmacytoid DCs", "Platelets", "HSPCs",
        }
        assert set(PBMC_MARKERS.keys()) == expected

    def test_all_types_have_markers(self) -> None:
        """Every cell type has at least 2 canonical markers."""
        for cell_type, genes in PBMC_MARKERS.items():
            assert len(genes) >= 2, f"{cell_type} has fewer than 2 canonical markers"

    def test_no_duplicate_markers_within_type(self) -> None:
        """No duplicate genes within a single cell type's marker list."""
        for cell_type, genes in PBMC_MARKERS.items():
            assert len(genes) == len(set(genes)), f"{cell_type} has duplicate markers"

    def test_t_cell_markers_distinct_from_b_cells(self) -> None:
        """T cell and B cell marker sets are mutually exclusive."""
        t_markers = set(PBMC_MARKERS["CD4+ T cells"])
        b_markers = set(PBMC_MARKERS["B cells"])
        assert len(t_markers & b_markers) == 0


class TestDetectMarkersUseRaw:
    """Test that detect_markers correctly uses raw log-normalised counts."""

    def test_uses_raw_when_available(self, clustered_adata: ad.AnnData) -> None:
        """detect_markers runs on .raw when it is set."""
        assert clustered_adata.raw is not None
        result = detect_markers(clustered_adata)
        params = result.uns["rank_genes_groups"]["params"]
        assert params.get("use_raw") is True

    def test_genes_come_from_raw_var_names(self, clustered_adata: ad.AnnData) -> None:
        """Detected gene names exist in adata.raw.var_names."""
        result = detect_markers(clustered_adata)
        raw_genes = set(result.raw.var_names)
        top = get_top_markers(result, n_genes=5)
        detected_genes = set(top["names"].tolist())
        assert detected_genes.issubset(raw_genes)


class TestDetectMarkers:
    """Test Wilcoxon marker detection."""

    def test_rank_genes_groups_added(self, clustered_adata: ad.AnnData) -> None:
        """rank_genes_groups is added to .uns after detection."""
        result = detect_markers(clustered_adata)
        assert "rank_genes_groups" in result.uns

    def test_all_clusters_represented(self, clustered_adata: ad.AnnData) -> None:
        """Every Leiden cluster has markers detected."""
        result = detect_markers(clustered_adata)
        clusters_in_leiden = set(clustered_adata.obs["leiden"].cat.categories)
        clusters_in_markers = set(
            result.uns["rank_genes_groups"]["names"].dtype.names
        )
        assert clusters_in_leiden == clusters_in_markers

    def test_scores_are_finite(self, clustered_adata: ad.AnnData) -> None:
        """All DE scores are finite (no NaN or Inf)."""
        result = detect_markers(clustered_adata)
        scores = result.uns["rank_genes_groups"]["scores"]
        for group_scores in scores:
            assert np.all(np.isfinite(list(group_scores)))

    def test_pvalues_between_zero_and_one(self, clustered_adata: ad.AnnData) -> None:
        """Adjusted p-values are in [0, 1]."""
        result = detect_markers(clustered_adata)
        pvals = result.uns["rank_genes_groups"]["pvals_adj"]
        for group_pvals in pvals:
            vals = np.array(list(group_pvals), dtype=float)
            assert np.all(vals >= 0)
            assert np.all(vals <= 1)


class TestGetTopMarkers:
    """Test top marker gene extraction."""

    def test_returns_dataframe(self, clustered_adata: ad.AnnData) -> None:
        """get_top_markers returns a pandas DataFrame."""
        import pandas as pd
        adata = detect_markers(clustered_adata)
        result = get_top_markers(adata, n_genes=5)
        assert isinstance(result, pd.DataFrame)

    def test_cluster_column_present(self, clustered_adata: ad.AnnData) -> None:
        """Result has a 'cluster' column."""
        adata = detect_markers(clustered_adata)
        result = get_top_markers(adata, n_genes=5)
        assert "cluster" in result.columns

    def test_gene_column_present(self, clustered_adata: ad.AnnData) -> None:
        """Result has a 'names' column (gene names)."""
        adata = detect_markers(clustered_adata)
        result = get_top_markers(adata, n_genes=5)
        assert "names" in result.columns

    def test_max_n_genes_per_cluster(self, clustered_adata: ad.AnnData) -> None:
        """No cluster has more than n_genes rows in the result."""
        adata = detect_markers(clustered_adata)
        n_genes = 5
        result = get_top_markers(adata, n_genes=n_genes)
        per_cluster = result.groupby("cluster").size()
        assert (per_cluster <= n_genes).all()

    def test_all_clusters_present(self, clustered_adata: ad.AnnData) -> None:
        """All Leiden clusters appear in the top markers table."""
        adata = detect_markers(clustered_adata)
        result = get_top_markers(adata, n_genes=5)
        leiden_clusters = set(clustered_adata.obs["leiden"].cat.categories)
        marker_clusters = set(result["cluster"].unique())
        assert leiden_clusters == marker_clusters


class TestScoreMarkerOverlap:
    """Test Jaccard-based canonical marker matching."""

    def test_returns_dataframe(self, clustered_adata: ad.AnnData) -> None:
        """score_marker_overlap returns a DataFrame."""
        import pandas as pd
        detected = {"0": ["CD3D", "CD3E", "IL7R"], "1": ["CD79A", "MS4A1"]}
        result = score_marker_overlap(clustered_adata, detected)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, clustered_adata: ad.AnnData) -> None:
        """Result has all required columns."""
        detected = {"0": ["CD3D", "IL7R"], "1": ["CD79A"]}
        result = score_marker_overlap(clustered_adata, detected)
        for col in ["cluster", "best_match", "jaccard_score", "n_matched"]:
            assert col in result.columns

    def test_jaccard_score_range(self, clustered_adata: ad.AnnData) -> None:
        """Jaccard scores are in [0, 1]."""
        detected = {"0": ["CD3D", "IL7R", "CD4"], "1": ["CD79A", "MS4A1"]}
        result = score_marker_overlap(clustered_adata, detected)
        assert (result["jaccard_score"] >= 0).all()
        assert (result["jaccard_score"] <= 1).all()

    def test_t_cell_markers_match_t_cell_type(self, clustered_adata: ad.AnnData) -> None:
        """Cluster with canonical T cell markers is matched to T cells."""
        detected = {"0": ["CD3D", "CD3E", "IL7R", "CD4"]}
        result = score_marker_overlap(clustered_adata, detected)
        match = result.loc[result["cluster"] == "0", "best_match"].iloc[0]
        assert "T" in match

    def test_b_cell_markers_match_b_cell_type(self, clustered_adata: ad.AnnData) -> None:
        """Cluster with canonical B cell markers is matched to B cells."""
        detected = {"0": ["CD79A", "CD79B", "MS4A1", "CD19"]}
        result = score_marker_overlap(clustered_adata, detected)
        match = result.loc[result["cluster"] == "0", "best_match"].iloc[0]
        assert match == "B cells"

    def test_unknown_genes_give_zero_score(self, clustered_adata: ad.AnnData) -> None:
        """Completely novel genes not in any reference give Jaccard score near 0."""
        detected = {"0": ["FAKE_GENE_XYZ", "ANOTHER_FAKE_123"]}
        result = score_marker_overlap(clustered_adata, detected)
        assert result.loc[result["cluster"] == "0", "jaccard_score"].iloc[0] == 0.0
