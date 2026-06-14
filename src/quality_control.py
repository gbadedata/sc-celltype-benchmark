"""Quality control for single-cell RNA-seq data.

Computes QC metrics (mitochondrial %, gene counts, UMI counts),
applies filtering thresholds, and reports cell/gene dropout statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anndata as ad
import scanpy as sc

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class QCReport:
    """Summary statistics from quality control filtering.

    Attributes:
        cells_before: Total cells before filtering.
        cells_after: Total cells after filtering.
        cells_removed: Number of cells removed.
        genes_before: Total genes before filtering.
        genes_after: Total genes after filtering.
        genes_removed: Number of genes removed.
        median_genes_per_cell: Median gene count per cell (after QC).
        median_counts_per_cell: Median UMI count per cell (after QC).
        median_pct_mito: Median mitochondrial percentage (after QC).
        pct_mito_failures: Cells removed for high mitochondrial %.
        low_gene_failures: Cells removed for too few genes.
        high_gene_failures: Cells removed for too many genes (potential doublets).
    """

    cells_before: int
    cells_after: int
    cells_removed: int
    genes_before: int
    genes_after: int
    genes_removed: int
    median_genes_per_cell: float
    median_counts_per_cell: float
    median_pct_mito: float
    pct_mito_failures: int
    low_gene_failures: int
    high_gene_failures: int


def compute_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """Compute QC metrics: n_genes_by_counts, total_counts, pct_counts_mt.

    Identifies mitochondrial genes (prefixed 'MT-') and computes
    per-cell and per-gene quality statistics. Adds columns to
    adata.obs and adata.var in place.

    Args:
        adata: AnnData with raw counts.

    Returns:
        Same AnnData with QC columns added to .obs and .var.
    """
    # Flag mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    n_mito = int(adata.var["mt"].sum())
    logger.info("mitochondrial_genes_found: %d", n_mito)

    # Compute metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    logger.info(
        "qc_metrics_computed: median_genes=%.0f, median_counts=%.0f, median_pct_mt=%.2f",
        adata.obs["n_genes_by_counts"].median(),
        adata.obs["total_counts"].median(),
        adata.obs["pct_counts_mt"].median(),
    )
    return adata


def filter_cells(adata: ad.AnnData) -> tuple[ad.AnnData, dict[str, int]]:
    """Filter cells based on QC thresholds from settings.

    Removes cells with:
    - Fewer than min_genes_per_cell genes detected
    - More than max_genes_per_cell genes detected (potential doublets)
    - More than max_pct_mito mitochondrial reads

    Args:
        adata: AnnData with QC metrics computed.

    Returns:
        Tuple of (filtered AnnData, dict of removal counts by reason).
    """
    n_before = adata.n_obs

    # Count failures per category before filtering
    low_genes = int((adata.obs["n_genes_by_counts"] < settings.min_genes_per_cell).sum())
    high_genes = int((adata.obs["n_genes_by_counts"] > settings.max_genes_per_cell).sum())
    high_mito = int((adata.obs["pct_counts_mt"] > settings.max_pct_mito).sum())

    # Apply filters
    keep = (
        (adata.obs["n_genes_by_counts"] >= settings.min_genes_per_cell)
        & (adata.obs["n_genes_by_counts"] <= settings.max_genes_per_cell)
        & (adata.obs["pct_counts_mt"] <= settings.max_pct_mito)
    )
    adata = adata[keep, :].copy()
    n_removed = n_before - adata.n_obs

    removal_counts = {
        "low_genes": low_genes,
        "high_genes": high_genes,
        "high_mito": high_mito,
        "total_removed": n_removed,
    }

    logger.info(
        "cells_filtered: %d -> %d (removed %d: low_genes=%d, high_genes=%d, high_mito=%d)",
        n_before,
        adata.n_obs,
        n_removed,
        low_genes,
        high_genes,
        high_mito,
    )
    return adata, removal_counts


def filter_genes(adata: ad.AnnData) -> ad.AnnData:
    """Filter genes expressed in fewer than min_cells_per_gene cells.

    Removes genes that are too rarely expressed to be informative.

    Args:
        adata: AnnData (cell-filtered).

    Returns:
        AnnData with low-expression genes removed.
    """
    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=settings.min_cells_per_gene)
    n_removed = n_before - adata.n_vars

    logger.info(
        "genes_filtered: %d -> %d (removed %d with < %d cells)",
        n_before,
        adata.n_vars,
        n_removed,
        settings.min_cells_per_gene,
    )
    return adata


def run_qc(adata: ad.AnnData) -> tuple[ad.AnnData, QCReport]:
    """Full QC pipeline: compute metrics, filter cells, filter genes.

    Args:
        adata: AnnData with raw counts.

    Returns:
        Tuple of (cleaned AnnData, QCReport with summary statistics).
    """
    cells_before = adata.n_obs
    genes_before = adata.n_vars

    # Step 1: compute metrics
    adata = compute_qc_metrics(adata)

    # Step 2: filter cells
    adata, removal_counts = filter_cells(adata)

    # Step 3: filter genes
    adata = filter_genes(adata)

    # Build report
    report = QCReport(
        cells_before=cells_before,
        cells_after=adata.n_obs,
        cells_removed=cells_before - adata.n_obs,
        genes_before=genes_before,
        genes_after=adata.n_vars,
        genes_removed=genes_before - adata.n_vars,
        median_genes_per_cell=float(adata.obs["n_genes_by_counts"].median()),
        median_counts_per_cell=float(adata.obs["total_counts"].median()),
        median_pct_mito=float(adata.obs["pct_counts_mt"].median()),
        pct_mito_failures=removal_counts["high_mito"],
        low_gene_failures=removal_counts["low_genes"],
        high_gene_failures=removal_counts["high_genes"],
    )

    logger.info(
        "qc_complete: %d cells x %d genes (removed %d cells, %d genes)",
        report.cells_after,
        report.genes_after,
        report.cells_removed,
        report.genes_removed,
    )
    return adata, report
