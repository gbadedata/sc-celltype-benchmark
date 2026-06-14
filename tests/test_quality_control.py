"""Tests for quality_control module.

Tests QC metric computation, cell filtering, gene filtering,
and the full QC pipeline using synthetic data.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from src.quality_control import (
    QCReport,
    compute_qc_metrics,
    filter_cells,
    filter_genes,
    run_qc,
)


class TestComputeQcMetrics:
    """Test QC metric computation."""

    def test_adds_mt_flag(self, synthetic_adata: ad.AnnData) -> None:
        """Mitochondrial genes are flagged in var."""
        compute_qc_metrics(synthetic_adata)
        assert "mt" in synthetic_adata.var.columns
        mt_count = synthetic_adata.var["mt"].sum()
        assert mt_count >= 3  # Fixture has MT-CO1, MT-CO2, etc.

    def test_adds_obs_columns(self, synthetic_adata: ad.AnnData) -> None:
        """QC metric columns are added to obs."""
        compute_qc_metrics(synthetic_adata)
        expected = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
        for col in expected:
            assert col in synthetic_adata.obs.columns

    def test_pct_mito_non_negative(self, synthetic_adata: ad.AnnData) -> None:
        """Mitochondrial percentage is non-negative."""
        compute_qc_metrics(synthetic_adata)
        assert (synthetic_adata.obs["pct_counts_mt"] >= 0).all()

    def test_gene_counts_positive(self, synthetic_adata: ad.AnnData) -> None:
        """Gene counts per cell are positive (no empty cells in fixture)."""
        compute_qc_metrics(synthetic_adata)
        assert (synthetic_adata.obs["n_genes_by_counts"] > 0).all()


class TestFilterCells:
    """Test cell filtering."""

    def test_removes_high_mito_cells(self) -> None:
        """Cells with high mitochondrial % are removed."""
        rng = np.random.default_rng(42)
        n_cells, n_genes = 100, 50
        X = sp.csr_matrix(rng.poisson(5, (n_cells, n_genes)).astype(np.float32))

        var_names = [f"GENE_{i}" for i in range(n_genes - 5)] + [
            "MT-CO1", "MT-CO2", "MT-ND1", "MT-ND2", "MT-ATP6"
        ]
        adata = ad.AnnData(X=X, var={"_index": var_names})
        adata.var.index = var_names

        # Inject extreme mito in first 10 cells
        dense = adata.X.toarray()
        dense[:10, -5:] = 500
        adata.X = sp.csr_matrix(dense)

        from src.quality_control import compute_qc_metrics
        compute_qc_metrics(adata)
        filtered, counts = filter_cells(adata)

        assert filtered.n_obs < n_cells
        assert counts["high_mito"] >= 10

    def test_preserves_good_cells(self, synthetic_adata: ad.AnnData) -> None:
        """Cells within thresholds are preserved."""
        compute_qc_metrics(synthetic_adata)
        filtered, counts = filter_cells(synthetic_adata)
        # Synthetic fixture has moderate values, most cells should pass
        assert filtered.n_obs > 0
        assert filtered.n_obs <= synthetic_adata.n_obs

    def test_returns_removal_counts(self, synthetic_adata: ad.AnnData) -> None:
        """filter_cells returns a dict with removal category counts."""
        compute_qc_metrics(synthetic_adata)
        _, counts = filter_cells(synthetic_adata)
        assert "low_genes" in counts
        assert "high_genes" in counts
        assert "high_mito" in counts
        assert "total_removed" in counts


class TestFilterGenes:
    """Test gene filtering."""

    def test_removes_rare_genes(self) -> None:
        """Genes expressed in too few cells are removed."""
        rng = np.random.default_rng(42)
        n_cells, n_genes = 100, 50
        X = rng.poisson(2, (n_cells, n_genes)).astype(np.float32)
        # Make last 5 genes expressed in only 1 cell each
        X[:, -5:] = 0
        for i in range(5):
            X[i, -(i + 1)] = 1
        adata = ad.AnnData(X=sp.csr_matrix(X))

        filtered = filter_genes(adata)
        assert filtered.n_vars < n_genes

    def test_preserves_common_genes(self, synthetic_adata: ad.AnnData) -> None:
        """Genes expressed in many cells are preserved."""
        compute_qc_metrics(synthetic_adata)
        filtered = filter_genes(synthetic_adata)
        # Most genes in the fixture should be expressed in >= 3 cells
        assert filtered.n_vars > 0


class TestRunQc:
    """Test the full QC pipeline."""

    def test_returns_adata_and_report(self, synthetic_adata: ad.AnnData) -> None:
        """run_qc returns a tuple of (AnnData, QCReport)."""
        adata, report = run_qc(synthetic_adata)
        assert isinstance(adata, ad.AnnData)
        assert isinstance(report, QCReport)

    def test_report_counts_consistent(self, synthetic_adata: ad.AnnData) -> None:
        """Report cell counts are internally consistent."""
        adata, report = run_qc(synthetic_adata)
        assert report.cells_before == 200
        assert report.cells_after == adata.n_obs
        assert report.cells_removed == report.cells_before - report.cells_after

    def test_report_gene_counts_consistent(self, synthetic_adata: ad.AnnData) -> None:
        """Report gene counts are internally consistent."""
        adata, report = run_qc(synthetic_adata)
        assert report.genes_before == 500
        assert report.genes_after == adata.n_vars
        assert report.genes_removed == report.genes_before - report.genes_after

    def test_report_medians_positive(self, synthetic_adata: ad.AnnData) -> None:
        """Median QC values in the report are positive."""
        _, report = run_qc(synthetic_adata)
        assert report.median_genes_per_cell > 0
        assert report.median_counts_per_cell > 0
        assert report.median_pct_mito >= 0

    def test_output_has_qc_columns(self, synthetic_adata: ad.AnnData) -> None:
        """Output AnnData has QC metric columns."""
        adata, _ = run_qc(synthetic_adata)
        for col in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]:
            assert col in adata.obs.columns
