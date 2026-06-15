"""Difficulty tier assignment for benchmark evaluation.

Cell types are assigned to difficulty tiers based on how hard they
are to annotate correctly. This separates benchmark performance on
easy cases (abundant, distinctive) from hard cases (rare, ambiguous).

Difficulty is determined by:
- Cell abundance: rare types are harder to cluster and annotate
- Marker distinctiveness: types with unique, high-expression markers are easier
- Transcriptional similarity: types similar to neighbours are harder
- Biological state: transitional or activated states are harder than resting
"""

from __future__ import annotations

import logging
from enum import StrEnum

import pandas as pd

logger = logging.getLogger(__name__)


class DifficultyTier(StrEnum):
    """Annotation difficulty tiers."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# Tier assignments for PBMC cell types.
# Rationale for each assignment documented inline.
PBMC_DIFFICULTY_MAP: dict[str, DifficultyTier] = {
    # EASY: abundant populations with distinctive, well-characterised markers.
    # CD4+ T cells: largest cluster, CD3D/IL7R uniquely high.
    "CD4+ T cells":        DifficultyTier.EASY,
    # CD8+ T cells: well-separated from CD4 by CD8A/CD8B expression.
    "CD8+ T cells":        DifficultyTier.EASY,
    # B cells: CD79A/CD79B expression is highly specific, no overlap with T.
    "B cells":             DifficultyTier.EASY,
    # CD14+ Monocytes: S100A9/LYZ very high; large cluster.
    "CD14+ Monocytes":     DifficultyTier.EASY,
    # NK cells: NKG7/GNLY distinctive; moderate abundance.
    "NK cells":            DifficultyTier.EASY,

    # MEDIUM: smaller populations with overlapping markers or subtypes.
    # FCGR3A+ Monocytes: similar to CD14+ Mono but smaller; FCGR3A distinguishes.
    "FCGR3A+ Monocytes":   DifficultyTier.MEDIUM,
    # Dendritic cells: HLA-DR high but shared with monocytes; small population.
    "Dendritic cells":     DifficultyTier.MEDIUM,

    # HARD: rare populations with weak or shared markers.
    # Plasmacytoid DCs: very small, IL3RA/CLEC4C markers rarely in top genes.
    "Plasmacytoid DCs":    DifficultyTier.HARD,
    # Platelets: extremely small cluster; PPBP marker very specific but tiny count.
    "Platelets":           DifficultyTier.HARD,
    # HSPCs: rarest population; CD34 lowly expressed; easily confused with cycling T.
    "HSPCs":               DifficultyTier.HARD,
}


def assign_difficulty_tiers(
    cell_types: pd.Series,
    difficulty_map: dict[str, DifficultyTier] | None = None,
) -> pd.Series:
    """Map cell-type labels to difficulty tiers.

    Args:
        cell_types: Series of cell-type label strings.
        difficulty_map: Mapping from type name to tier.
            Defaults to PBMC_DIFFICULTY_MAP. Unknown types default to HARD.

    Returns:
        Series of DifficultyTier values with same index as input.
    """
    difficulty_map = difficulty_map or PBMC_DIFFICULTY_MAP
    tiers = cell_types.map(
        lambda x: difficulty_map.get(str(x), DifficultyTier.HARD)
    )
    return tiers.astype(str)


def compute_tier_statistics(
    cell_types: pd.Series,
    tier_assignments: pd.Series,
) -> pd.DataFrame:
    """Summarise cell counts and proportions per difficulty tier.

    Args:
        cell_types: Series of cell-type labels.
        tier_assignments: Series of tier labels (same index).

    Returns:
        DataFrame with columns: tier, n_types, n_cells, pct_of_total.
    """
    df = pd.DataFrame({"cell_type": cell_types, "tier": tier_assignments})
    total = len(df)

    rows = []
    for tier in [DifficultyTier.EASY, DifficultyTier.MEDIUM, DifficultyTier.HARD]:
        subset = df[df["tier"] == str(tier)]
        rows.append(
            {
                "tier": str(tier),
                "n_types": subset["cell_type"].nunique(),
                "n_cells": len(subset),
                "pct_of_total": round(len(subset) / total * 100, 1) if total > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows)
