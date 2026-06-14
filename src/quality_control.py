"""Quality control for single-cell RNA-seq data.

Computes QC metrics (mitochondrial %, gene counts, UMI counts),
applies filtering thresholds, and reports cell/gene dropout statistics.
"""

from __future__ import annotations

import logging

import anndata as ad

logger = logging.getLogger(__name__)


def compute_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """Compute QC metrics: n_genes, total_counts, pct_counts_mt.

    Adds columns to adata.obs and adata.var in place.
    """
    raise NotImplementedError("Phase 2")


def filter_cells(adata: ad.AnnData) -> ad.AnnData:
    """Filter cells based on QC thresholds from settings.

    Returns a new AnnData with passing cells only.
    Logs the number of cells removed and reasons.
    """
    raise NotImplementedError("Phase 2")


def filter_genes(adata: ad.AnnData) -> ad.AnnData:
    """Filter genes expressed in fewer than min_cells_per_gene cells.

    Returns a new AnnData with passing genes only.
    """
    raise NotImplementedError("Phase 2")


def run_qc(adata: ad.AnnData) -> ad.AnnData:
    """Full QC pipeline: compute metrics, filter cells, filter genes.

    Returns cleaned AnnData. Logs QC summary statistics.
    """
    raise NotImplementedError("Phase 2")
