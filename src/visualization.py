"""Visualization for single-cell analysis and benchmark results.

Generates publication-quality figures for QC, clustering, annotation,
and benchmark evaluation. All figures are saved to evidence/figures/.
"""

from __future__ import annotations

import logging

import anndata as ad
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

FIGURE_DIR = settings.evidence_dir / "figures"


def plot_qc_metrics(adata: ad.AnnData, save: bool = True) -> None:
    """Violin plots of QC metrics: n_genes, total_counts, pct_mito."""
    raise NotImplementedError("Phase 8")


def plot_umap_clusters(adata: ad.AnnData, save: bool = True) -> None:
    """UMAP coloured by Leiden cluster."""
    raise NotImplementedError("Phase 8")


def plot_umap_celltypes(
    adata: ad.AnnData,
    label_col: str = "celltype_manual",
    save: bool = True,
) -> None:
    """UMAP coloured by cell-type annotation."""
    raise NotImplementedError("Phase 8")


def plot_marker_dotplot(adata: ad.AnnData, save: bool = True) -> None:
    """Dot plot of canonical markers across annotated cell types."""
    raise NotImplementedError("Phase 8")


def plot_marker_heatmap(adata: ad.AnnData, save: bool = True) -> None:
    """Heatmap of top marker genes per cluster."""
    raise NotImplementedError("Phase 8")


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    title: str = "Annotation Confusion Matrix",
    save: bool = True,
) -> None:
    """Heatmap confusion matrix comparing two annotation methods."""
    raise NotImplementedError("Phase 8")


def plot_benchmark_summary(
    tier_results: dict,
    save: bool = True,
) -> None:
    """Bar chart of benchmark F1 scores by difficulty tier."""
    raise NotImplementedError("Phase 8")


def plot_annotation_comparison(
    adata: ad.AnnData,
    save: bool = True,
) -> None:
    """Side-by-side UMAPs: manual vs reference annotation."""
    raise NotImplementedError("Phase 8")
