"""Preprocessing: normalisation, HVG selection, scaling, PCA.

Transforms raw counts into a reduced representation suitable for
clustering and visualisation.

Steps:
1. Store raw counts in adata.raw (preserves original for downstream DE).
2. Library-size normalise to 10,000 counts per cell.
3. Log1p transform.
4. Select highly variable genes (Seurat flavour on log-normalised data).
5. Scale to zero mean, unit variance (per gene, max 10 SD clip).
6. PCA on HVG-subset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anndata as ad
import scanpy as sc

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingReport:
    """Summary statistics from preprocessing.

    Attributes:
        n_cells: Number of cells after preprocessing.
        n_genes_total: Total genes (before HVG selection).
        n_hvg: Number of highly variable genes selected.
        n_pcs: Number of PCA components computed.
        pct_variance_explained: Cumulative variance explained by n_pcs components.
    """

    n_cells: int
    n_genes_total: int
    n_hvg: int
    n_pcs: int
    pct_variance_explained: float


def normalize(adata: ad.AnnData) -> ad.AnnData:
    """Library-size normalisation and log1p transformation.

    Normalises each cell to total_counts=10,000, then applies log1p.
    Saves raw counts to adata.raw before any transformation so
    downstream differential expression tests can access original values.

    Args:
        adata: AnnData with raw integer counts.

    Returns:
        AnnData with normalised log1p counts in .X. Raw counts in .raw.
    """
    # Take a copy before any transformation so .raw holds true integer counts.
    # Assigning adata.raw = adata after in-place ops can capture a partially
    # modified matrix; copying first guarantees the snapshot is clean.
    adata.raw = adata.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(
        "normalized: target_sum=10000, log1p applied, raw counts saved to .raw"
    )
    return adata


def select_hvg(adata: ad.AnnData) -> ad.AnnData:
    """Select highly variable genes.

    Uses the Seurat flavour of highly_variable_genes, which operates
    on log-normalised data (mean/dispersion approach). This is the
    correct flavour post log-normalisation and does not require the
    skmisc dependency that seurat_v3 needs.

    HVGs are marked in adata.var['highly_variable']. The full gene
    set is retained; the flag is applied during PCA.

    Args:
        adata: AnnData with normalised log1p counts.

    Returns:
        Same AnnData with HVG flags added to .var.
    """
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=settings.n_top_genes,
        flavor="seurat",
    )

    n_hvg = int(adata.var["highly_variable"].sum())
    logger.info(
        "hvg_selected: %d / %d genes (top %d by dispersion, seurat flavour)",
        n_hvg,
        adata.n_vars,
        settings.n_top_genes,
    )
    return adata


def scale_and_pca(adata: ad.AnnData) -> ad.AnnData:
    """Scale to unit variance, then compute PCA.

    Scales gene expression to zero mean and unit variance per gene.
    Clips values at max_value=10 to limit the influence of outliers.
    PCA is computed on the HVG subset only, which is standard practice
    to focus on informative variation.

    Args:
        adata: AnnData with normalised log1p counts and HVG flags.

    Returns:
        AnnData with PCA embedding in .obsm['X_pca'] and variance
        ratios in .uns['pca']['variance_ratio'].
    """
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(
        adata,
        n_comps=settings.n_pcs,
        use_highly_variable=True,
        svd_solver="arpack",
        random_state=settings.random_seed,
    )

    variance_ratio = adata.uns["pca"]["variance_ratio"]
    cumulative_pct = float(variance_ratio[: settings.n_pcs].sum() * 100)

    logger.info(
        "pca_computed: %d components, %.1f%% cumulative variance explained",
        settings.n_pcs,
        cumulative_pct,
    )
    return adata


def run_preprocessing(adata: ad.AnnData) -> tuple[ad.AnnData, PreprocessingReport]:
    """Full preprocessing pipeline: normalize, HVG, scale, PCA.

    Args:
        adata: QC-filtered AnnData with raw counts.

    Returns:
        Tuple of (preprocessed AnnData, PreprocessingReport).
    """
    n_genes_total = adata.n_vars

    adata = normalize(adata)
    adata = select_hvg(adata)
    n_hvg = int(adata.var["highly_variable"].sum())
    adata = scale_and_pca(adata)

    variance_ratio = adata.uns["pca"]["variance_ratio"]
    pct_var = float(variance_ratio[: settings.n_pcs].sum() * 100)

    report = PreprocessingReport(
        n_cells=adata.n_obs,
        n_genes_total=n_genes_total,
        n_hvg=n_hvg,
        n_pcs=settings.n_pcs,
        pct_variance_explained=round(pct_var, 2),
    )

    logger.info(
        "preprocessing_complete: %d cells, %d HVGs, %d PCs (%.1f%% variance)",
        report.n_cells,
        report.n_hvg,
        report.n_pcs,
        report.pct_variance_explained,
    )
    return adata, report
