"""Validator functions.

Each validator checks a specific aspect of annotation quality.
Validators return structured results with pass/fail status,
a numeric score, and a human-readable explanation.

These are distinct from the evaluator (which computes metrics).
Validators enforce biological constraints that metrics alone
cannot capture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anndata as ad

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a single validator.

    Attributes:
        name: Validator name.
        passed: Whether the check passed.
        score: Numeric score (0.0 to 1.0).
        details: Human-readable explanation.
        evidence: Supporting data (e.g., which genes matched).
    """

    name: str
    passed: bool
    score: float
    details: str
    evidence: dict | None = None


def validate_cluster_purity(
    adata: ad.AnnData,
    cluster_col: str = "leiden",
    label_col: str = "celltype_reference",
) -> ValidationResult:
    """Check whether clusters are dominated by a single cell type.

    A pure cluster has >80% of cells belonging to one type.
    Mixed clusters indicate either over-clustering, biological
    heterogeneity, or annotation errors.

    Returns:
        ValidationResult with fraction of pure clusters.
    """
    raise NotImplementedError("Phase 7")


def validate_marker_recovery(
    adata: ad.AnnData,
    predicted_col: str = "celltype_manual",
    reference_markers: dict[str, list[str]] | None = None,
) -> ValidationResult:
    """Check whether predicted cell types express expected markers.

    For each predicted cell type, verifies that the canonical
    marker genes are among the top differentially expressed genes
    in that population.

    Returns:
        ValidationResult with per-type marker recovery rate.
    """
    raise NotImplementedError("Phase 7")


def validate_biological_consistency(
    adata: ad.AnnData,
    predicted_col: str = "celltype_manual",
) -> ValidationResult:
    """Check biological constraints that must hold regardless of method.

    Constraints checked:
    - CD3D expression should be high in T cells, low in B cells
    - CD79A expression should be high in B cells, low in T cells
    - CD14 expression should be high in CD14+ Monocytes
    - NKG7 expression should be high in NK cells
    - No cell type should have zero cells assigned

    These are necessary (not sufficient) conditions for correct
    annotation. A pipeline that violates them has a bug.

    Returns:
        ValidationResult with number of constraints satisfied.
    """
    raise NotImplementedError("Phase 7")


def run_all_validators(
    adata: ad.AnnData,
    predicted_col: str = "celltype_manual",
) -> list[ValidationResult]:
    """Run all validators and return a list of results."""
    raise NotImplementedError("Phase 7")
