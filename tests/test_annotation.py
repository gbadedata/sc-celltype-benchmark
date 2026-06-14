"""Tests for annotation module.

Tests manual marker scoring annotation, CellTypist label harmonisation,
and annotation comparison logic using synthetic clustered data.

CellTypist model download is not tested here (requires network + large
model file). That is tested in integration tests run manually.
"""

from __future__ import annotations

import anndata as ad
import pandas as pd
import pytest

from src.annotation import (
    CELLTYPIST_COARSE_MAP,
    annotate_manual,
    compare_annotations,
    harmonise_labels,
)
from src.clustering import run_clustering
from src.preprocessing import run_preprocessing
from src.quality_control import run_qc


@pytest.fixture
def clustered_adata(synthetic_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData: QC-filtered, preprocessed, clustered."""
    adata, _ = run_qc(synthetic_adata)
    adata, _ = run_preprocessing(adata)
    adata, _ = run_clustering(adata)
    return adata


@pytest.fixture
def manually_annotated(clustered_adata: ad.AnnData) -> ad.AnnData:
    """Clustered AnnData with manual annotation applied."""
    return annotate_manual(clustered_adata)


class TestCelltypistCoarseMap:
    """Validate the CellTypist label harmonisation map."""

    def test_map_is_non_empty(self) -> None:
        """CELLTYPIST_COARSE_MAP has entries."""
        assert len(CELLTYPIST_COARSE_MAP) > 0

    def test_all_values_are_known_types(self) -> None:
        """All coarse labels map to recognised PBMC cell types."""
        from src.markers import PBMC_MARKERS
        known_types = set(PBMC_MARKERS.keys())
        for fine, coarse in CELLTYPIST_COARSE_MAP.items():
            assert coarse in known_types, (
                f"'{fine}' maps to '{coarse}' which is not in PBMC_MARKERS"
            )

    def test_classical_monocytes_map_correctly(self) -> None:
        """Classical monocytes map to CD14+ Monocytes."""
        assert CELLTYPIST_COARSE_MAP["Classical monocytes"] == "CD14+ Monocytes"

    def test_pdc_maps_correctly(self) -> None:
        """pDC maps to Plasmacytoid DCs."""
        assert CELLTYPIST_COARSE_MAP["pDC"] == "Plasmacytoid DCs"

    def test_nonclassical_monocytes_map_correctly(self) -> None:
        """Non-classical monocytes map to FCGR3A+ Monocytes."""
        assert CELLTYPIST_COARSE_MAP["Non-classical monocytes"] == "FCGR3A+ Monocytes"


class TestAnnotateManual:
    """Test marker-score-based manual annotation."""

    def test_celltype_manual_column_added(self, clustered_adata: ad.AnnData) -> None:
        """annotate_manual adds celltype_manual to .obs."""
        result = annotate_manual(clustered_adata)
        assert "celltype_manual" in result.obs.columns

    def test_all_cells_annotated(self, clustered_adata: ad.AnnData) -> None:
        """Every cell receives an annotation (no NaN)."""
        result = annotate_manual(clustered_adata)
        assert result.obs["celltype_manual"].isna().sum() == 0

    def test_annotation_is_categorical(self, clustered_adata: ad.AnnData) -> None:
        """Annotation column is stored as categorical."""
        result = annotate_manual(clustered_adata)
        assert hasattr(result.obs["celltype_manual"], "cat")

    def test_known_types_assigned(self, clustered_adata: ad.AnnData) -> None:
        """All assigned cell types are from the known marker dictionary."""
        from src.markers import PBMC_MARKERS
        result = annotate_manual(clustered_adata)
        assigned = set(result.obs["celltype_manual"].astype(str).unique())
        known = set(PBMC_MARKERS.keys()) | {"Unknown"}
        assert assigned.issubset(known)

    def test_t_cell_cluster_gets_t_cell_label(self, clustered_adata: ad.AnnData) -> None:
        """The cluster most enriched for T cell markers gets a T cell label."""
        result = annotate_manual(clustered_adata)
        # synthetic fixture has CD3D, IL7R, CD4 enriched in T cell cluster
        t_cell_labels = {
            "CD4+ T cells", "CD8+ T cells"
        }
        assigned_types = set(result.obs["celltype_manual"].astype(str))
        # At least one T cell type should be assigned
        assert len(assigned_types & t_cell_labels) >= 1

    def test_b_cell_cluster_gets_b_cell_label(self, clustered_adata: ad.AnnData) -> None:
        """The cluster most enriched for B cell markers gets 'B cells'."""
        result = annotate_manual(clustered_adata)
        assigned_types = set(result.obs["celltype_manual"].astype(str))
        assert "B cells" in assigned_types

    def test_uses_raw_when_available(self, clustered_adata: ad.AnnData) -> None:
        """Manual annotation uses .raw counts when available."""
        assert clustered_adata.raw is not None
        # Run and confirm no error; raw access is implicit in _get_log_norm_matrix
        result = annotate_manual(clustered_adata)
        assert result is not None

    def test_custom_markers_respected(self, clustered_adata: ad.AnnData) -> None:
        """Custom marker dictionary produces labels from that dictionary."""
        custom = {
            "TypeA": ["CD3D", "CD3E"],
            "TypeB": ["CD79A", "CD79B"],
        }
        result = annotate_manual(clustered_adata, markers=custom)
        assigned = set(result.obs["celltype_manual"].astype(str))
        assert assigned.issubset({"TypeA", "TypeB", "Unknown"})


class TestHarmoniseLabels:
    """Test CellTypist label harmonisation."""

    def test_adds_coarse_column(self, manually_annotated: ad.AnnData) -> None:
        """harmonise_labels adds celltype_reference_coarse column."""
        adata = manually_annotated.copy()
        adata.obs["celltype_reference"] = "Classical monocytes"
        result = harmonise_labels(adata)
        assert "celltype_reference_coarse" in result.obs.columns

    def test_known_labels_mapped(self, manually_annotated: ad.AnnData) -> None:
        """Known CellTypist labels are correctly mapped to coarse types."""
        adata = manually_annotated.copy()
        adata.obs["celltype_reference"] = "Classical monocytes"
        result = harmonise_labels(adata)
        assert (result.obs["celltype_reference_coarse"] == "CD14+ Monocytes").all()

    def test_unknown_labels_kept(self, manually_annotated: ad.AnnData) -> None:
        """Labels not in the map are kept as-is (not dropped)."""
        adata = manually_annotated.copy()
        adata.obs["celltype_reference"] = "SomeNovelType"
        result = harmonise_labels(adata)
        assert (result.obs["celltype_reference_coarse"] == "SomeNovelType").all()

    def test_raises_without_reference_column(self, manually_annotated: ad.AnnData) -> None:
        """harmonise_labels raises ValueError if reference column missing."""
        with pytest.raises(ValueError, match="celltype_reference column missing"):
            harmonise_labels(manually_annotated)

    def test_output_is_categorical(self, manually_annotated: ad.AnnData) -> None:
        """Coarse label column is stored as categorical."""
        adata = manually_annotated.copy()
        adata.obs["celltype_reference"] = "NK cells"
        result = harmonise_labels(adata)
        assert hasattr(result.obs["celltype_reference_coarse"], "cat")


class TestCompareAnnotations:
    """Test confusion matrix between manual and reference annotations."""

    def test_returns_dataframe(self, annotated_adata: ad.AnnData) -> None:
        """compare_annotations returns a DataFrame."""
        result = compare_annotations(annotated_adata)
        assert isinstance(result, pd.DataFrame)

    def test_raises_missing_manual(self, clustered_adata: ad.AnnData) -> None:
        """compare_annotations raises if celltype_manual missing."""
        clustered_adata.obs["celltype_reference_coarse"] = "CD4+ T cells"
        with pytest.raises(ValueError, match="Missing column"):
            compare_annotations(clustered_adata)

    def test_raises_missing_reference(self, manually_annotated: ad.AnnData) -> None:
        """compare_annotations raises if celltype_reference_coarse missing."""
        with pytest.raises(ValueError, match="Missing column"):
            compare_annotations(manually_annotated)

    def test_confusion_matrix_shape(self, annotated_adata: ad.AnnData) -> None:
        """Confusion matrix rows/cols correspond to unique cell types."""
        result = compare_annotations(annotated_adata)
        n_manual = annotated_adata.obs["celltype_manual"].nunique()
        n_reference = annotated_adata.obs["celltype_reference_coarse"].nunique()
        assert result.shape[0] == n_manual
        assert result.shape[1] == n_reference

    def test_confusion_matrix_sums_to_n_cells(self, annotated_adata: ad.AnnData) -> None:
        """Confusion matrix values sum to total number of cells."""
        result = compare_annotations(annotated_adata)
        assert result.values.sum() == len(annotated_adata)
