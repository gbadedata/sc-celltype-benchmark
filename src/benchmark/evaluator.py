"""Benchmark evaluator.

Scores predicted cell-type annotations against oracle ground truth.
Produces per-cell-type and aggregate metrics, confusion matrices,
and structured benchmark reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results.

    Attributes:
        accuracy: Overall fraction of correctly annotated cells.
        weighted_f1: Weighted-average F1 across cell types.
        macro_f1: Unweighted-average F1 (equal weight per type).
        cohens_kappa: Inter-rater agreement statistic.
        n_cells_evaluated: Number of cells in the test set.
        per_type_metrics: DataFrame with precision, recall, F1 per type.
        confusion_df: DataFrame confusion matrix (predicted x true).
        tier_results: Dict mapping difficulty tier name to BenchmarkResult.
    """

    accuracy: float
    weighted_f1: float
    macro_f1: float
    cohens_kappa: float
    n_cells_evaluated: int
    per_type_metrics: pd.DataFrame
    confusion_df: pd.DataFrame
    tier_results: dict[str, "BenchmarkResult"] = field(default_factory=dict)


class BenchmarkEvaluator:
    """Evaluate annotation predictions against oracle ground truth."""

    def score(
        self,
        predictions: pd.Series,
        ground_truth: pd.Series,
    ) -> BenchmarkResult:
        """Score predictions against ground truth.

        Args:
            predictions: Predicted cell-type labels (same index as ground_truth).
            ground_truth: Oracle ground-truth labels.

        Returns:
            BenchmarkResult with all computed metrics.

        Raises:
            ValueError: If predictions and ground_truth have different indices.
        """
        if not predictions.index.equals(ground_truth.index):
            raise ValueError(
                "predictions and ground_truth must have the same index. "
                f"Got {len(predictions)} predictions and {len(ground_truth)} ground truth labels."
            )

        y_pred = predictions.astype(str).values
        y_true = ground_truth.astype(str).values
        labels = sorted(set(y_true) | set(y_pred))

        accuracy = float((y_pred == y_true).mean())
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        kappa = float(cohen_kappa_score(y_true, y_pred))

        # Per-type metrics as DataFrame
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        per_type_rows = []
        for label in labels:
            if label in report:
                per_type_rows.append(
                    {
                        "cell_type": label,
                        "precision": round(report[label]["precision"], 4),
                        "recall": round(report[label]["recall"], 4),
                        "f1": round(report[label]["f1-score"], 4),
                        "support": int(report[label]["support"]),
                    }
                )
        per_type_df = pd.DataFrame(per_type_rows)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        confusion_df = pd.DataFrame(cm, index=labels, columns=labels)
        confusion_df.index.name = "true"
        confusion_df.columns.name = "predicted"

        logger.info(
            "benchmark_scored: n=%d, accuracy=%.4f, weighted_f1=%.4f, kappa=%.4f",
            len(y_true),
            accuracy,
            weighted_f1,
            kappa,
        )

        return BenchmarkResult(
            accuracy=accuracy,
            weighted_f1=weighted_f1,
            macro_f1=macro_f1,
            cohens_kappa=kappa,
            n_cells_evaluated=len(y_true),
            per_type_metrics=per_type_df,
            confusion_df=confusion_df,
        )

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
            tier_assignments: Series mapping each cell to a difficulty tier
                (same index as ground_truth).

        Returns:
            Dict mapping tier name to BenchmarkResult.
        """
        tier_results: dict[str, BenchmarkResult] = {}

        for tier in sorted(tier_assignments.unique()):
            tier_mask = tier_assignments == tier
            tier_gt = ground_truth[tier_mask]
            tier_pred = predictions.reindex(tier_gt.index)

            if len(tier_gt) == 0:
                logger.warning("No cells in tier '%s' — skipping.", tier)
                continue

            # Drop cells where prediction is NaN (not all cells are annotated)
            valid = tier_pred.notna()
            if valid.sum() == 0:
                logger.warning("No valid predictions for tier '%s' — skipping.", tier)
                continue

            result = self.score(tier_pred[valid], tier_gt[valid])
            tier_results[str(tier)] = result
            logger.info(
                "tier_%s: n=%d, accuracy=%.4f, weighted_f1=%.4f",
                tier,
                result.n_cells_evaluated,
                result.accuracy,
                result.weighted_f1,
            )

        return tier_results

    def generate_report(
        self,
        result: BenchmarkResult,
        dataset_name: str | None = None,
    ) -> dict:
        """Produce a structured JSON-serialisable benchmark report.

        Args:
            result: BenchmarkResult from score().
            dataset_name: Optional dataset label for the report header.

        Returns:
            Dict containing all metrics, confusion matrix, per-tier breakdown,
            and metadata (timestamp, settings snapshot).
        """
        report: dict = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset_name or settings.dataset_name,
                "holdout_fraction": settings.benchmark_holdout_fraction,
                "seed": settings.benchmark_seed,
            },
            "summary": {
                "n_cells_evaluated": result.n_cells_evaluated,
                "accuracy": round(result.accuracy, 4),
                "weighted_f1": round(result.weighted_f1, 4),
                "macro_f1": round(result.macro_f1, 4),
                "cohens_kappa": round(result.cohens_kappa, 4),
            },
            "per_type_metrics": result.per_type_metrics.to_dict(orient="records"),
            "confusion_matrix": {
                "labels": list(result.confusion_df.columns),
                "matrix": result.confusion_df.values.tolist(),
            },
        }

        if result.tier_results:
            report["tier_breakdown"] = {
                tier: {
                    "n_cells": tr.n_cells_evaluated,
                    "accuracy": round(tr.accuracy, 4),
                    "weighted_f1": round(tr.weighted_f1, 4),
                    "cohens_kappa": round(tr.cohens_kappa, 4),
                }
                for tier, tr in result.tier_results.items()
            }

        return report
