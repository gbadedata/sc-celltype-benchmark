"""Cell-type annotation.

Two annotation strategies:
1. Manual: assign cell types to clusters based on canonical marker gene
   expression using a scoring approach.
2. Reference-based: use CellTypist pre-trained models for automated
   per-cell annotation with confidence scores.

Both methods produce a cell_type column in adata.obs. The benchmark
framework compares their outputs.
"""

from __future__ import annotations

import logging

import anndata as ad
import pandas as pd

logger = logging.getLogger(__name__)


def annotate_manual(adata: ad.AnnData, groupby: str = "leiden") -> ad.AnnData:
    """Assign cell types to clusters using canonical marker scoring.

    For each cluster, scores expression of canonical marker gene sets
    (from markers.PBMC_MARKERS). Assigns the cell type whose markers
    have the highest mean expression in that cluster.

    Stores result in adata.obs['celltype_manual'].
    """
    raise NotImplementedError("Phase 6")


def annotate_celltypist(adata: ad.AnnData) -> ad.AnnData:
    """Annotate cells using CellTypist reference model.

    Downloads the configured model if not cached. Runs per-cell
    prediction with majority voting. Stores results in
    adata.obs['celltype_reference'] and confidence scores in
    adata.obs['celltype_reference_conf'].
    """
    raise NotImplementedError("Phase 6")


def harmonise_labels(adata: ad.AnnData) -> ad.AnnData:
    """Map fine-grained CellTypist labels to coarse PBMC categories.

    CellTypist returns detailed labels like 'Classical monocytes'.
    This maps them to the categories used by manual annotation
    (e.g., 'CD14+ Monocytes') for fair comparison.

    Stores result in adata.obs['celltype_reference_coarse'].
    """
    raise NotImplementedError("Phase 6")


def compare_annotations(adata: ad.AnnData) -> pd.DataFrame:
    """Compare manual and reference annotations.

    Returns a confusion matrix and agreement statistics
    (Cohen's kappa, overall accuracy, per-type F1).
    """
    raise NotImplementedError("Phase 6")


def run_annotation(adata: ad.AnnData) -> ad.AnnData:
    """Full annotation pipeline: manual + CellTypist + comparison.

    Returns AnnData with both annotation columns populated.
    """
    raise NotImplementedError("Phase 6")
