"""Oracle answer generation.

The oracle holds ground-truth cell-type labels and controls the
benchmark experiment. It generates stratified holdout masks,
partitions cells into visible (training) and withheld (test) sets,
and releases ground-truth answers only for scoring after the pipeline
has run blind on the withheld set.

The oracle is the single source of truth for evaluation. The pipeline
never sees oracle labels during the annotation step — only the evaluator
receives them, after predictions are submitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


@dataclass
class OracleGenerator:
    """Manages ground-truth labels and holdout experiments.

    Attributes:
        ground_truth: Series mapping cell barcodes to true cell types.
        holdout_mask: Boolean array; True = cell is in test set (withheld).
        seed: Random seed for reproducible splits.
    """

    ground_truth: pd.Series = field(default_factory=pd.Series)
    holdout_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    seed: int = 42

    def fit(self, adata: ad.AnnData, label_column: str = "celltype_reference_coarse") -> None:
        """Store ground-truth labels from an annotated AnnData.

        Labels must be pre-populated in adata.obs[label_column] before
        calling fit(). Typically this is the CellTypist majority-voting
        output after harmonisation to coarse categories.

        Args:
            adata: AnnData with reference annotations in .obs.
            label_column: Column in .obs containing ground-truth labels.

        Raises:
            ValueError: If label_column is not present in adata.obs.
        """
        if label_column not in adata.obs.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        self.ground_truth = adata.obs[label_column].astype(str).copy()
        n_types = self.ground_truth.nunique()
        logger.info(
            "oracle_fitted: %d cells, %d cell types",
            len(self.ground_truth),
            n_types,
        )

    def generate_holdout(self, fraction: float = 0.2) -> np.ndarray:
        """Create a stratified holdout mask.

        Uses stratified sampling to ensure each cell type is represented
        proportionally in both the visible and withheld sets. Cell types
        with fewer than 2 cells are excluded from the test set (they
        cannot be stratified) and all their cells remain in the visible set.

        Args:
            fraction: Fraction of cells to withhold as the test set.
                Default 0.2 (20%).

        Returns:
            Boolean array of length n_cells; True = cell is withheld.
        """
        if len(self.ground_truth) == 0:
            raise RuntimeError("Call fit() before generate_holdout().")

        labels = self.ground_truth.values
        indices = np.arange(len(labels))

        # Identify cell types with enough cells to stratify
        unique, counts = np.unique(labels, return_counts=True)
        stratifiable_mask = np.isin(labels, unique[counts >= 2])

        holdout_mask = np.zeros(len(labels), dtype=bool)

        if stratifiable_mask.sum() > 0:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=fraction,
                random_state=self.seed,
            )
            strat_indices = indices[stratifiable_mask]
            strat_labels = labels[stratifiable_mask]

            _, test_local = next(splitter.split(strat_indices, strat_labels))
            test_global = strat_indices[test_local]
            holdout_mask[test_global] = True

        self.holdout_mask = holdout_mask
        n_withheld = holdout_mask.sum()
        n_visible = (~holdout_mask).sum()

        logger.info(
            "holdout_generated: fraction=%.2f, test=%d cells, train=%d cells",
            fraction,
            n_withheld,
            n_visible,
        )
        return self.holdout_mask

    def get_answers(self, mask: np.ndarray | None = None) -> pd.Series:
        """Return ground-truth labels for the withheld (test) cells.

        This method is called by the evaluator after the pipeline has
        produced predictions. It should not be called during annotation.

        Args:
            mask: Optional custom boolean mask. Defaults to self.holdout_mask.

        Returns:
            Series of true labels for test cells only.
        """
        if len(self.ground_truth) == 0:
            raise RuntimeError("Call fit() before get_answers().")
        if mask is None:
            if len(self.holdout_mask) == 0:
                raise RuntimeError("Call generate_holdout() before get_answers().")
            mask = self.holdout_mask

        return self.ground_truth[mask]

    def get_visible_labels(self) -> pd.Series:
        """Return ground-truth labels for the visible (non-withheld) cells.

        These are the labels the pipeline is allowed to see during any
        supervised or semi-supervised annotation step.

        Returns:
            Series of true labels for non-withheld cells.
        """
        if len(self.holdout_mask) == 0:
            raise RuntimeError("Call generate_holdout() before get_visible_labels().")

        return self.ground_truth[~self.holdout_mask]

    @property
    def n_cells(self) -> int:
        """Total number of cells in the oracle."""
        return len(self.ground_truth)

    @property
    def n_withheld(self) -> int:
        """Number of cells in the test (withheld) set."""
        return int(self.holdout_mask.sum()) if len(self.holdout_mask) > 0 else 0

    @property
    def n_visible(self) -> int:
        """Number of cells in the training (visible) set."""
        return int((~self.holdout_mask).sum()) if len(self.holdout_mask) > 0 else 0
