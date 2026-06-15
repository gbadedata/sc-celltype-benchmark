"""Tests for benchmark evaluator and difficulty tier modules."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.benchmark.difficulty import (
    PBMC_DIFFICULTY_MAP,
    DifficultyTier,
    assign_difficulty_tiers,
    compute_tier_statistics,
)
from src.benchmark.evaluator import BenchmarkEvaluator, BenchmarkResult


@pytest.fixture
def perfect_predictions(annotated_adata: ad.AnnData) -> tuple[pd.Series, pd.Series]:
    """Perfect predictions: match ground truth exactly."""
    gt = annotated_adata.obs["celltype_reference_coarse"].astype(str)
    return gt.copy(), gt


@pytest.fixture
def noisy_predictions(annotated_adata: ad.AnnData) -> tuple[pd.Series, pd.Series]:
    """Noisy predictions: 20% of labels deliberately wrong."""
    gt = annotated_adata.obs["celltype_reference_coarse"].astype(str)
    pred = gt.copy()
    rng = np.random.default_rng(42)
    noise_idx = rng.choice(len(pred), size=len(pred) // 5, replace=False)
    pred.iloc[noise_idx] = "B cells"
    return pred, gt


class TestDifficultyTier:

    def test_tier_enum_values(self) -> None:
        assert DifficultyTier.EASY == "easy"
        assert DifficultyTier.MEDIUM == "medium"
        assert DifficultyTier.HARD == "hard"

    def test_pbmc_map_covers_all_common_types(self) -> None:
        expected = {"CD4+ T cells", "B cells", "CD14+ Monocytes", "NK cells"}
        assert expected.issubset(set(PBMC_DIFFICULTY_MAP.keys()))

    def test_easy_types_include_major_lineages(self) -> None:
        easy = {k for k, v in PBMC_DIFFICULTY_MAP.items() if v == DifficultyTier.EASY}
        assert "CD4+ T cells" in easy
        assert "B cells" in easy
        assert "CD14+ Monocytes" in easy

    def test_hard_types_include_rare_populations(self) -> None:
        hard = {k for k, v in PBMC_DIFFICULTY_MAP.items() if v == DifficultyTier.HARD}
        assert "Platelets" in hard
        assert "HSPCs" in hard

    def test_assign_difficulty_tiers_returns_series(self) -> None:
        cell_types = pd.Series(["CD4+ T cells", "B cells", "Platelets"])
        result = assign_difficulty_tiers(cell_types)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_known_types_assigned_correctly(self) -> None:
        cell_types = pd.Series(["CD4+ T cells", "Platelets", "FCGR3A+ Monocytes"])
        result = assign_difficulty_tiers(cell_types)
        assert result.iloc[0] == "easy"
        assert result.iloc[1] == "hard"
        assert result.iloc[2] == "medium"

    def test_unknown_type_defaults_to_hard(self) -> None:
        cell_types = pd.Series(["SomeUnknownType"])
        result = assign_difficulty_tiers(cell_types)
        assert result.iloc[0] == "hard"

    def test_compute_tier_statistics_returns_dataframe(self) -> None:
        cell_types = pd.Series(["CD4+ T cells"] * 5 + ["Platelets"] * 2)
        tiers = assign_difficulty_tiers(cell_types)
        stats = compute_tier_statistics(cell_types, tiers)
        assert isinstance(stats, pd.DataFrame)
        assert "tier" in stats.columns
        assert "n_cells" in stats.columns

    def test_tier_statistics_sums_to_total(self) -> None:
        cell_types = pd.Series(["CD4+ T cells"] * 5 + ["Platelets"] * 3)
        tiers = assign_difficulty_tiers(cell_types)
        stats = compute_tier_statistics(cell_types, tiers)
        assert stats["n_cells"].sum() == 8


class TestBenchmarkEvaluator:

    def test_perfect_predictions_accuracy_one(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        evaluator = BenchmarkEvaluator()
        result = evaluator.score(pred, gt)
        assert result.accuracy == pytest.approx(1.0)

    def test_perfect_predictions_f1_one(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        evaluator = BenchmarkEvaluator()
        result = evaluator.score(pred, gt)
        assert result.weighted_f1 == pytest.approx(1.0)

    def test_perfect_predictions_kappa_one(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        evaluator = BenchmarkEvaluator()
        result = evaluator.score(pred, gt)
        assert result.cohens_kappa == pytest.approx(1.0)

    def test_noisy_predictions_lower_accuracy(
        self, noisy_predictions: tuple, perfect_predictions: tuple
    ) -> None:
        pred_noisy, gt = noisy_predictions
        pred_perfect, _ = perfect_predictions
        evaluator = BenchmarkEvaluator()
        result_noisy = evaluator.score(pred_noisy, gt)
        result_perfect = evaluator.score(pred_perfect, gt)
        assert result_noisy.accuracy < result_perfect.accuracy

    def test_returns_benchmark_result(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        result = BenchmarkEvaluator().score(pred, gt)
        assert isinstance(result, BenchmarkResult)

    def test_per_type_metrics_has_required_columns(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        result = BenchmarkEvaluator().score(pred, gt)
        for col in ["cell_type", "precision", "recall", "f1", "support"]:
            assert col in result.per_type_metrics.columns

    def test_confusion_matrix_shape(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        result = BenchmarkEvaluator().score(pred, gt)
        n_types = len(gt.unique())
        assert result.confusion_df.shape == (n_types, n_types)

    def test_confusion_matrix_sum_equals_n_cells(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        result = BenchmarkEvaluator().score(pred, gt)
        assert result.confusion_df.values.sum() == len(gt)

    def test_n_cells_evaluated_correct(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        result = BenchmarkEvaluator().score(pred, gt)
        assert result.n_cells_evaluated == len(gt)

    def test_mismatched_index_raises(self, annotated_adata: ad.AnnData) -> None:
        gt = annotated_adata.obs["celltype_reference_coarse"].astype(str)
        pred = pd.Series(["B cells"] * 50)  # wrong index
        with pytest.raises(ValueError, match="same index"):
            BenchmarkEvaluator().score(pred, gt)

    def test_score_by_tier_returns_dict(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        tiers = assign_difficulty_tiers(gt)
        result = BenchmarkEvaluator().score_by_tier(pred, gt, tiers)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_score_by_tier_keys_are_tier_names(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        tiers = assign_difficulty_tiers(gt)
        result = BenchmarkEvaluator().score_by_tier(pred, gt, tiers)
        valid_tiers = {"easy", "medium", "hard"}
        assert set(result.keys()).issubset(valid_tiers)

    def test_generate_report_is_serialisable(self, perfect_predictions: tuple) -> None:
        import json
        pred, gt = perfect_predictions
        result = BenchmarkEvaluator().score(pred, gt)
        report = BenchmarkEvaluator().generate_report(result)
        # Should not raise
        json.dumps(report, default=str)

    def test_generate_report_has_required_keys(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        result = BenchmarkEvaluator().score(pred, gt)
        report = BenchmarkEvaluator().generate_report(result)
        for key in ["metadata", "summary", "per_type_metrics", "confusion_matrix"]:
            assert key in report

    def test_generate_report_summary_accuracy(self, perfect_predictions: tuple) -> None:
        pred, gt = perfect_predictions
        result = BenchmarkEvaluator().score(pred, gt)
        report = BenchmarkEvaluator().generate_report(result)
        assert report["summary"]["accuracy"] == pytest.approx(1.0)
