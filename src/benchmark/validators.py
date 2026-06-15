"""Biological constraint validators.

Each validator checks a specific biological property that must hold
regardless of annotation method. A pipeline that violates these
constraints has a bug — not a performance limitation.

Validators return structured ValidationResult objects with a pass/fail
flag, a numeric score, and a human-readable explanation. They are
distinct from the evaluator (which computes statistical metrics against
oracle labels): validators check necessary biological conditions that
can be verified from the data alone, without needing ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a single biological constraint validator.

    Attributes:
        name: Validator name.
        passed: Whether the constraint was satisfied.
        score: Numeric score in [0, 1]. 1.0 = fully satisfied.
        details: Human-readable explanation of the result.
        evidence: Supporting data (e.g. expression values, gene lists).
    """

    name: str
    passed: bool
    score: float
    details: str
    evidence: dict = field(default_factory=dict)


def _get_expression(adata: ad.AnnData, gene: str) -> np.ndarray | None:
    """Extract per-cell expression for a gene from layers['log_norm'] or .X."""
    # Prefer log_norm layer (log1p-normalised, pre-scaling)
    if "log_norm" in adata.layers:
        X = adata.layers["log_norm"]
    else:
        X = adata.X

    if gene not in adata.var_names:
        return None

    idx = list(adata.var_names).index(gene)
    col = X[:, idx]
    if issparse(col):
        col = col.toarray().flatten()
    return col.flatten()


def validate_cluster_purity(
    adata: ad.AnnData,
    cluster_col: str = "leiden",
    label_col: str = "celltype_manual",
    purity_threshold: float = 0.8,
) -> ValidationResult:
    """Check whether clusters are dominated by a single cell type.

    A pure cluster has >= purity_threshold of cells belonging to one type.
    Mixed clusters indicate over-clustering, biological heterogeneity,
    or annotation errors.

    Args:
        adata: Annotated AnnData.
        cluster_col: Column with cluster labels.
        label_col: Column with cell-type annotation.
        purity_threshold: Minimum fraction for a cluster to be pure.

    Returns:
        ValidationResult with fraction of pure clusters as score.
    """
    if cluster_col not in adata.obs.columns or label_col not in adata.obs.columns:
        return ValidationResult(
            name="cluster_purity",
            passed=False,
            score=0.0,
            details=f"Required columns missing: {cluster_col}, {label_col}",
        )

    clusters = adata.obs[cluster_col].astype(str)
    labels = adata.obs[label_col].astype(str)

    purity_scores = []
    pure_clusters = []
    mixed_clusters = []

    for cluster_id in sorted(clusters.unique()):
        mask = clusters == cluster_id
        type_counts = labels[mask].value_counts()
        dominant_fraction = float(type_counts.iloc[0] / mask.sum())
        purity_scores.append(dominant_fraction)

        if dominant_fraction >= purity_threshold:
            pure_clusters.append(cluster_id)
        else:
            mixed_clusters.append(
                f"cluster_{cluster_id}={dominant_fraction:.2f}"
            )

    mean_purity = float(np.mean(purity_scores))
    fraction_pure = len(pure_clusters) / len(purity_scores)
    passed = fraction_pure >= 0.75  # at least 75% of clusters should be pure

    details = (
        f"{len(pure_clusters)}/{len(purity_scores)} clusters are pure "
        f"(>= {purity_threshold:.0%} dominant type). "
        f"Mean purity: {mean_purity:.3f}."
    )
    if mixed_clusters:
        details += f" Mixed: {', '.join(mixed_clusters[:3])}"

    logger.info("validate_cluster_purity: score=%.4f, passed=%s", fraction_pure, passed)
    return ValidationResult(
        name="cluster_purity",
        passed=passed,
        score=round(fraction_pure, 4),
        details=details,
        evidence={"pure_clusters": pure_clusters, "mixed_clusters": mixed_clusters},
    )


def validate_marker_recovery(
    adata: ad.AnnData,
    predicted_col: str = "celltype_manual",
) -> ValidationResult:
    """Check whether predicted cell types express expected canonical markers.

    For each predicted cell type, verifies that at least one canonical
    marker gene is expressed at above-background levels (mean > overall mean
    across all cells). A type that does not express any of its canonical
    markers has been misannotated.

    Args:
        adata: Annotated AnnData with log_norm layer or .X.
        predicted_col: Column with predicted cell-type labels.

    Returns:
        ValidationResult with fraction of types passing marker recovery.
    """
    from src.markers import PBMC_MARKERS

    if predicted_col not in adata.obs.columns:
        return ValidationResult(
            name="marker_recovery",
            passed=False,
            score=0.0,
            details=f"Column '{predicted_col}' not found in adata.obs",
        )

    labels = adata.obs[predicted_col].astype(str)
    assigned_types = [t for t in labels.unique() if t != "Unknown"]
    n_types = len(assigned_types)

    if n_types == 0:
        return ValidationResult(
            name="marker_recovery",
            passed=False,
            score=0.0,
            details="No cell types assigned (all Unknown)",
        )

    passing = []
    failing = []

    for cell_type in assigned_types:
        if cell_type not in PBMC_MARKERS:
            continue

        type_mask = labels == cell_type
        markers = PBMC_MARKERS[cell_type]
        available = [g for g in markers if g in adata.var_names]

        if not available:
            failing.append(cell_type)
            continue

        # Check: is mean expression of any marker gene higher in this type
        # than in the global mean?
        type_passes = False
        for gene in available:
            expr = _get_expression(adata, gene)
            if expr is None:
                continue
            global_mean = float(expr.mean())
            type_mean = float(expr[type_mask.values].mean())
            if type_mean > global_mean:
                type_passes = True
                break

        if type_passes:
            passing.append(cell_type)
        else:
            failing.append(cell_type)

    n_checked = len(passing) + len(failing)
    score = len(passing) / n_checked if n_checked > 0 else 0.0
    passed = score >= 0.8

    details = (
        f"{len(passing)}/{n_checked} cell types express at least one canonical marker "
        f"above background."
    )
    if failing:
        details += f" Failing: {', '.join(failing)}"

    logger.info("validate_marker_recovery: score=%.4f, passed=%s", score, passed)
    return ValidationResult(
        name="marker_recovery",
        passed=passed,
        score=round(score, 4),
        details=details,
        evidence={"passing_types": passing, "failing_types": failing},
    )


def validate_biological_consistency(
    adata: ad.AnnData,
    predicted_col: str = "celltype_manual",
) -> ValidationResult:
    """Check hard biological constraints that must hold for any correct annotation.

    These constraints are necessary (not sufficient) conditions. A pipeline
    that violates them has made a definitive error. They are checked from
    the expression data directly, without needing oracle labels.

    Constraints:
    1. CD3D mean expression: T cells > B cells (T cell lineage marker)
    2. CD79A mean expression: B cells > T cells (B cell lineage marker)
    3. CD14 mean expression: CD14+ Monocytes > B cells (monocyte marker)
    4. NKG7 mean expression: NK cells > CD4+ T cells (cytotoxic marker)
    5. No cell type has zero cells assigned

    Args:
        adata: Annotated AnnData.
        predicted_col: Column with predicted cell-type labels.

    Returns:
        ValidationResult with fraction of constraints satisfied.
    """
    if predicted_col not in adata.obs.columns:
        return ValidationResult(
            name="biological_consistency",
            passed=False,
            score=0.0,
            details=f"Column '{predicted_col}' not found in adata.obs",
        )

    labels = adata.obs[predicted_col].astype(str)
    assigned_types = set(labels.unique())

    constraints_checked = 0
    constraints_passed = 0
    failed_details: list[str] = []

    def mean_expr(gene: str, cell_type: str) -> float | None:
        if cell_type not in assigned_types:
            return None
        expr = _get_expression(adata, gene)
        if expr is None:
            return None
        mask = (labels == cell_type).values
        if mask.sum() == 0:
            return None
        return float(expr[mask].mean())

    # Constraint 1: CD3D higher in T cells than B cells
    t_cd3d = mean_expr("CD3D", "CD4+ T cells") or mean_expr("CD3D", "CD8+ T cells")
    b_cd3d = mean_expr("CD3D", "B cells")
    if t_cd3d is not None and b_cd3d is not None:
        constraints_checked += 1
        if t_cd3d > b_cd3d:
            constraints_passed += 1
        else:
            failed_details.append(
                f"CD3D: T cells ({t_cd3d:.3f}) should > B cells ({b_cd3d:.3f})"
            )

    # Constraint 2: CD79A higher in B cells than T cells
    b_cd79a = mean_expr("CD79A", "B cells")
    t_cd79a = mean_expr("CD79A", "CD4+ T cells") or mean_expr("CD79A", "CD8+ T cells")
    if b_cd79a is not None and t_cd79a is not None:
        constraints_checked += 1
        if b_cd79a > t_cd79a:
            constraints_passed += 1
        else:
            failed_details.append(
                f"CD79A: B cells ({b_cd79a:.3f}) should > T cells ({t_cd79a:.3f})"
            )

    # Constraint 3: CD14 higher in CD14+ Mono than B cells
    m_cd14 = mean_expr("CD14", "CD14+ Monocytes")
    b_cd14 = mean_expr("CD14", "B cells")
    if m_cd14 is not None and b_cd14 is not None:
        constraints_checked += 1
        if m_cd14 > b_cd14:
            constraints_passed += 1
        else:
            failed_details.append(
                f"CD14: Monocytes ({m_cd14:.3f}) should > B cells ({b_cd14:.3f})"
            )

    # Constraint 4: NKG7 higher in NK cells than CD4+ T cells
    nk_nkg7 = mean_expr("NKG7", "NK cells")
    t_nkg7 = mean_expr("NKG7", "CD4+ T cells")
    if nk_nkg7 is not None and t_nkg7 is not None:
        constraints_checked += 1
        if nk_nkg7 > t_nkg7:
            constraints_passed += 1
        else:
            failed_details.append(
                f"NKG7: NK cells ({nk_nkg7:.3f}) should > CD4+ T ({t_nkg7:.3f})"
            )

    # Constraint 5: no cell type assigned to zero cells
    zero_types = [t for t in assigned_types if (labels == t).sum() == 0]
    constraints_checked += 1
    if not zero_types:
        constraints_passed += 1
    else:
        failed_details.append(f"Zero-cell types: {zero_types}")

    score = constraints_passed / constraints_checked if constraints_checked > 0 else 0.0
    passed = score >= 0.8

    details = f"{constraints_passed}/{constraints_checked} biological constraints satisfied."
    if failed_details:
        details += " Failed: " + "; ".join(failed_details)

    logger.info(
        "validate_biological_consistency: score=%.4f, passed=%s", score, passed
    )
    return ValidationResult(
        name="biological_consistency",
        passed=passed,
        score=round(score, 4),
        details=details,
        evidence={"failed_constraints": failed_details},
    )


def run_all_validators(
    adata: ad.AnnData,
    predicted_col: str = "celltype_manual",
) -> list[ValidationResult]:
    """Run all validators and return a list of results.

    Args:
        adata: Annotated AnnData.
        predicted_col: Column with predicted cell-type labels.

    Returns:
        List of ValidationResult, one per validator.
    """
    validators = [
        lambda a: validate_cluster_purity(a, label_col=predicted_col),
        lambda a: validate_marker_recovery(a, predicted_col=predicted_col),
        lambda a: validate_biological_consistency(a, predicted_col=predicted_col),
    ]

    results = [v(adata) for v in validators]
    n_passed = sum(r.passed for r in results)
    logger.info(
        "all_validators_complete: %d/%d passed",
        n_passed,
        len(results),
    )
    return results
