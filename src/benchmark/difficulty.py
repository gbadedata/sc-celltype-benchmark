"""Difficulty tier assignment for benchmark evaluation.

Cell types are assigned to difficulty tiers based on how hard they
are to annotate correctly. This allows the benchmark to report
performance separately for easy, medium, and hard cases.

Difficulty is determined by:
- Cell abundance (rare types are harder)
- Marker distinctiveness (unique markers = easier)
- Similarity to other types (similar types = harder)
- Biological state (transitional/activated states are harder)
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


# Default tier assignments for PBMC cell types.
# Based on literature consensus on annotation difficulty.
PBMC_DIFFICULTY_MAP: dict[str, DifficultyTier] = {
    # Easy: abundant, distinctive markers, well-separated clusters
    "CD4+ T cells": DifficultyTier.EASY,
    "CD8+ T cells": DifficultyTier.EASY,
    "B cells": DifficultyTier.EASY,
    "CD14+ Monocytes": DifficultyTier.EASY,
    "NK cells": DifficultyTier.EASY,
    # Medium: subtypes with overlapping markers, smaller populations
    "FCGR3A+ Monocytes": DifficultyTier.MEDIUM,
    "Dendritic cells": DifficultyTier.MEDIUM,
    # Hard: rare, weak markers, easily confused with other types
    "Plasmacytoid DCs": DifficultyTier.HARD,
    "Platelets": DifficultyTier.HARD,
    "HSPCs": DifficultyTier.HARD,
}


def assign_difficulty_tiers(
    cell_types: pd.Series,
    difficulty_map: dict[str, DifficultyTier] | None = None,
) -> pd.Series:
    """Map cell-type labels to difficulty tiers.

    Args:
        cell_types: Series of cell-type labels.
        difficulty_map: Mapping from type name to tier. Defaults
            to PBMC_DIFFICULTY_MAP.

    Returns:
        Series of DifficultyTier values, same index as input.
        Unknown types default to HARD.
    """
    raise NotImplementedError("Phase 7")


def compute_tier_statistics(
    cell_types: pd.Series,
    tier_assignments: pd.Series,
) -> pd.DataFrame:
    """Summarise cell counts and proportions per tier.

    Returns DataFrame with tier, n_cells, n_types, pct_of_total.
    """
    raise NotImplementedError("Phase 7")
