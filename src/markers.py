"""Marker gene detection.

Identifies differentially expressed genes per cluster using
Wilcoxon rank-sum test. Provides marker gene tables and
known canonical marker definitions for PBMC cell types.
"""

from __future__ import annotations

import logging

import anndata as ad
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical PBMC marker genes curated from literature.
# Each key is a cell type; values are genes expected to be
# highly expressed in that population.
PBMC_MARKERS: dict[str, list[str]] = {
    "CD4+ T cells": ["CD3D", "CD3E", "IL7R", "CD4"],
    "CD8+ T cells": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMK"],
    "NK cells": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "B cells": ["CD79A", "CD79B", "MS4A1", "CD19"],
    "CD14+ Monocytes": ["CD14", "LYZ", "S100A9", "S100A8"],
    "FCGR3A+ Monocytes": ["FCGR3A", "MS4A7", "IFITM3"],
    "Dendritic cells": ["FCER1A", "CST3", "CLEC10A"],
    "Plasmacytoid DCs": ["IL3RA", "CLEC4C", "IRF7", "TCF4"],
    "Platelets": ["PPBP", "PF4", "GP9"],
    "HSPCs": ["CD34", "SPINK2", "CRHBP"],
}


def detect_markers(adata: ad.AnnData, groupby: str = "leiden") -> ad.AnnData:
    """Run differential expression to find marker genes per group.

    Uses Wilcoxon rank-sum test. Stores results in adata.uns['rank_genes_groups'].
    """
    raise NotImplementedError("Phase 5")


def get_top_markers(adata: ad.AnnData, n_genes: int = 10) -> pd.DataFrame:
    """Extract top N marker genes per cluster as a tidy DataFrame.

    Columns: cluster, gene, score, pval, pval_adj, log2fc.
    """
    raise NotImplementedError("Phase 5")


def score_marker_overlap(
    detected: dict[str, list[str]],
    reference: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Score overlap between detected markers and canonical reference markers.

    Returns per-cluster Jaccard similarity and matched genes.
    """
    raise NotImplementedError("Phase 5")
