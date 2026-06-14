"""Oracle answer generation.

The oracle holds ground-truth cell-type labels and controls the
benchmark experiment. It:
1. Stores the full set of reference annotations (from CellTypist).
2. Generates holdout masks (which cells have labels withheld).
3. Provides the answer key for scoring after the pipeline runs blind.

The oracle is the single source of truth for evaluation. The pipeline
never sees oracle labels during the annotation step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OracleGenerator:
    """Manages ground-truth labels and holdout experiments.

    Attributes:
        ground_truth: Series mapping cell barcodes to true cell types.
        holdout_mask: Boolean array; True = label withheld (test set).
        seed: Random seed for reproducibility.
    """

    ground_truth: pd.Series = field(default_factory=pd.Series)
    holdout_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    seed: int = 42

    def fit(self, adata: ad.AnnData, label_column: str = "celltype_reference") -> None:
        """Store ground-truth labels from annotated AnnData.

        Args:
            adata: AnnData with reference annotations.
            label_column: Column in adata.obs containing ground truth.
        """
        raise NotImplementedError("Phase 7")

    def generate_holdout(self, fraction: float = 0.2) -> np.ndarray:
        """Create a stratified holdout mask.

        Ensures each cell type is represented proportionally in both
        the visible (train) and withheld (test) sets.

        Args:
            fraction: Fraction of cells to withhold.

        Returns:
            Boolean array; True = withheld.
        """
        raise NotImplementedError("Phase 7")

    def get_answers(self, mask: np.ndarray | None = None) -> pd.Series:
        """Return ground-truth labels for the withheld cells.

        Args:
            mask: Optional custom mask. Defaults to self.holdout_mask.

        Returns:
            Series of true labels for test cells only.
        """
        raise NotImplementedError("Phase 7")

    def get_visible_labels(self) -> pd.Series:
        """Return ground-truth labels for visible (non-withheld) cells.

        These are the labels the pipeline is allowed to see during
        any supervised or semi-supervised annotation step.
        """
        raise NotImplementedError("Phase 7")
