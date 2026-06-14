"""Tests for synthetic data fixtures.

Validates that test fixtures produce correct AnnData structures
with expected properties.
"""

import anndata as ad
import numpy as np
import scipy.sparse as sp


class TestSyntheticAdata:
    """Validate the synthetic_adata fixture."""

    def test_shape(self, synthetic_adata: ad.AnnData) -> None:
        assert synthetic_adata.n_obs == 200
        assert synthetic_adata.n_vars == 500

    def test_sparse_matrix(self, synthetic_adata: ad.AnnData) -> None:
        assert sp.issparse(synthetic_adata.X)

    def test_ground_truth_labels_present(self, synthetic_adata: ad.AnnData) -> None:
        assert "celltype_ground_truth" in synthetic_adata.obs.columns

    def test_four_cell_types(self, synthetic_adata: ad.AnnData) -> None:
        types = synthetic_adata.obs["celltype_ground_truth"].unique()
        assert len(types) == 4
        expected = {"CD4+ T cells", "B cells", "CD14+ Monocytes", "NK cells"}
        assert set(types) == expected

    def test_marker_genes_present(self, synthetic_adata: ad.AnnData) -> None:
        expected_markers = ["CD3D", "CD79A", "CD14", "NKG7"]
        for gene in expected_markers:
            assert gene in synthetic_adata.var_names

    def test_mitochondrial_genes_present(self, synthetic_adata: ad.AnnData) -> None:
        mito_genes = [g for g in synthetic_adata.var_names if g.startswith("MT-")]
        assert len(mito_genes) >= 3

    def test_marker_enrichment(self, synthetic_adata: ad.AnnData) -> None:
        """Verify markers are enriched in their expected cell types."""
        adata = synthetic_adata
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X

        # CD3D (index 0) should be higher in T cells than B cells
        t_mask = adata.obs["celltype_ground_truth"] == "CD4+ T cells"
        b_mask = adata.obs["celltype_ground_truth"] == "B cells"
        cd3d_idx = list(adata.var_names).index("CD3D")

        t_mean = X[t_mask, cd3d_idx].mean()
        b_mean = X[b_mask, cd3d_idx].mean()
        assert t_mean > b_mean, f"CD3D should be higher in T cells ({t_mean}) than B cells ({b_mean})"

    def test_no_negative_counts(self, synthetic_adata: ad.AnnData) -> None:
        X = synthetic_adata.X.toarray() if sp.issparse(synthetic_adata.X) else synthetic_adata.X
        assert np.all(X >= 0)


class TestAnnotatedAdata:
    """Validate the annotated_adata fixture."""

    def test_manual_annotation_column(self, annotated_adata: ad.AnnData) -> None:
        assert "celltype_manual" in annotated_adata.obs.columns

    def test_reference_annotation_column(self, annotated_adata: ad.AnnData) -> None:
        assert "celltype_reference" in annotated_adata.obs.columns
        assert "celltype_reference_coarse" in annotated_adata.obs.columns

    def test_manual_matches_ground_truth(self, annotated_adata: ad.AnnData) -> None:
        manual = annotated_adata.obs["celltype_manual"]
        truth = annotated_adata.obs["celltype_ground_truth"]
        assert (manual == truth).all()

    def test_reference_has_noise(self, annotated_adata: ad.AnnData) -> None:
        ref = annotated_adata.obs["celltype_reference"]
        truth = annotated_adata.obs["celltype_ground_truth"]
        agreement = (ref == truth).mean()
        assert 0.9 <= agreement < 1.0, f"Expected ~95% agreement, got {agreement:.2%}"
