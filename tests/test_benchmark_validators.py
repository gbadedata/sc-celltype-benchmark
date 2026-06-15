"""Tests for biological constraint validators."""

from __future__ import annotations

import anndata as ad
import pytest

from src.benchmark.validators import (
    ValidationResult,
    run_all_validators,
    validate_biological_consistency,
    validate_cluster_purity,
    validate_marker_recovery,
)
from src.clustering import run_clustering
from src.preprocessing import run_preprocessing
from src.quality_control import run_qc


@pytest.fixture
def clustered_annotated(synthetic_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData: QC, preprocessed, clustered, manually annotated."""
    adata, _ = run_qc(synthetic_adata)
    adata, _ = run_preprocessing(adata)
    adata, _ = run_clustering(adata)
    from src.annotation import annotate_manual
    adata = annotate_manual(adata)
    return adata


class TestValidationResult:

    def test_dataclass_fields(self) -> None:
        r = ValidationResult(
            name="test", passed=True, score=0.9, details="ok"
        )
        assert r.name == "test"
        assert r.passed is True
        assert r.score == 0.9
        assert r.details == "ok"
        assert r.evidence == {}


class TestValidateClusterPurity:

    def test_returns_validation_result(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_cluster_purity(clustered_annotated)
        assert isinstance(result, ValidationResult)
        assert result.name == "cluster_purity"

    def test_score_in_unit_interval(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_cluster_purity(clustered_annotated)
        assert 0.0 <= result.score <= 1.0

    def test_missing_column_returns_failed(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_cluster_purity(
            clustered_annotated, label_col="nonexistent"
        )
        assert result.passed is False
        assert result.score == 0.0

    def test_perfect_clusters_score_one(self, annotated_adata: ad.AnnData) -> None:
        """When each leiden cluster has one pure cell type, score should be 1."""
        adata = annotated_adata.copy()
        # Assign unique leiden cluster per cell type
        cell_types = adata.obs["celltype_manual"].astype(str)
        type_to_cluster = {t: str(i) for i, t in enumerate(cell_types.unique())}
        adata.obs["leiden"] = cell_types.map(type_to_cluster).astype("category")
        result = validate_cluster_purity(adata, label_col="celltype_manual")
        assert result.score == pytest.approx(1.0)


class TestValidateMarkerRecovery:

    def test_returns_validation_result(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_marker_recovery(clustered_annotated)
        assert isinstance(result, ValidationResult)
        assert result.name == "marker_recovery"

    def test_score_in_unit_interval(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_marker_recovery(clustered_annotated)
        assert 0.0 <= result.score <= 1.0

    def test_missing_column_returns_failed(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_marker_recovery(
            clustered_annotated, predicted_col="nonexistent"
        )
        assert result.passed is False

    def test_evidence_keys_present(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_marker_recovery(clustered_annotated)
        assert "passing_types" in result.evidence
        assert "failing_types" in result.evidence


class TestValidateBiologicalConsistency:

    def test_returns_validation_result(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_biological_consistency(clustered_annotated)
        assert isinstance(result, ValidationResult)
        assert result.name == "biological_consistency"

    def test_score_in_unit_interval(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_biological_consistency(clustered_annotated)
        assert 0.0 <= result.score <= 1.0

    def test_missing_column_returns_failed(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_biological_consistency(
            clustered_annotated, predicted_col="nonexistent"
        )
        assert result.passed is False

    def test_correct_annotation_passes_constraints(self, clustered_annotated: ad.AnnData) -> None:
        """Biologically correct annotation (from fixture) should satisfy constraints."""
        result = validate_biological_consistency(clustered_annotated)
        # Synthetic fixture has marker enrichment built in — should pass
        assert result.score > 0.0

    def test_evidence_has_failed_constraints(self, clustered_annotated: ad.AnnData) -> None:
        result = validate_biological_consistency(clustered_annotated)
        assert "failed_constraints" in result.evidence


class TestRunAllValidators:

    def test_returns_list_of_results(self, clustered_annotated: ad.AnnData) -> None:
        results = run_all_validators(clustered_annotated)
        assert isinstance(results, list)
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_three_validators_run(self, clustered_annotated: ad.AnnData) -> None:
        results = run_all_validators(clustered_annotated)
        assert len(results) == 3

    def test_all_have_distinct_names(self, clustered_annotated: ad.AnnData) -> None:
        results = run_all_validators(clustered_annotated)
        names = [r.name for r in results]
        assert len(names) == len(set(names))
