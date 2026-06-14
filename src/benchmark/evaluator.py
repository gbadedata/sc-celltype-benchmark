"""Benchmark evaluator.

Scores predicted cell-type annotations against oracle ground truth.
Produces per-cell-type and aggregate metrics, confusion matrices,
and structured benchmark reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results.

    Attributes:
        accuracy: Overall fraction of correctly annotated cells.
        weighted_f1: Weighted-average F1 across cell types.
        per_type_metrics: DataFrame with precision, recall, F1 per type.
        confusion_matrix: DataFrame confusion matrix.
        cohens_kappa: Inter-rater agreement statistic.
        n_cells_evaluated: Number of cells in the test set.
        tier_results: Dict mapping difficulty tier to BenchmarkResult.
    """

    accuracy: float
    weighted_f1: float
    per_type_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    cohens_kappa: float
    n_cells_evaluated: int
    tier_results: dict | None = None


class BenchmarkEvaluator:
    """Evaluate annotation predictions against oracle ground truth."""

    def score(
        self,
        predictions: pd.Series,
        ground_truth: pd.Series,
    ) -> BenchmarkResult:
        """Score predictions against ground truth.

        Args:
            predictions: Predicted cell-type labels.
            ground_truth: Oracle ground-truth labels (same index).

        Returns:
            BenchmarkResult with all computed metrics.
        """
        raise NotImplementedError("Phase 7")

    def score_by_tier(
        self,
        predictions: pd.Series,
        ground_truth: pd.Series,
        tier_assignments: pd.Series,
    ) -> dict[str, BenchmarkResult]:
        """Score predictions separately for each difficulty tier.

        Args:
            predictions: Predicted labels.
            ground_truth: Oracle labels.
            tier_assignments: Series mapping cell types to difficulty tiers.

        Returns:
            Dict mapping tier name to BenchmarkResult.
        """
        raise NotImplementedError("Phase 7")

    def generate_report(self, result: BenchmarkResult) -> dict:
        """Produce a structured JSON-serialisable benchmark report.

        Includes all metrics, confusion matrix, per-tier breakdown,
        and metadata (timestamp, settings, dataset info).
        """
        raise NotImplementedError("Phase 7")
