"""Cell-type annotation.

Two independent annotation strategies:

1. Manual (marker scoring): For each cluster, scores the mean
   expression of each canonical cell type's marker genes across
   all cells in that cluster. The cell type with the highest
   normalised mean expression score wins. Operates on log-normalised
   counts from adata.raw.

2. Reference-based (CellTypist): Downloads a pre-trained immune cell
   model and runs per-cell annotation with majority voting per cluster.
   Produces fine-grained labels that are then harmonised to the same
   coarse categories used by manual annotation.

Both methods write a column to adata.obs. The benchmark framework
compares them using the reference-based labels as oracle ground truth.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from src.markers import PBMC_MARKERS

logger = logging.getLogger(__name__)

# Exhaustive mapping from Immune_All_Low.pkl labels to coarse PBMC categories.
#
# Source: Dominguez Conde et al. 2022 (Science), CellTypist Immune_All_Low model.
# Every label produced by this model is explicitly mapped. No label passes
# through unmapped — unmappable labels are set to "Unknown" explicitly.
#
# This map was validated against the actual labels returned on PBMC 3k:
# Tcm/Naive helper T cells, CD16+ NK cells, Megakaryocytes/platelets,
# DC, Tem/Trm cytotoxic T cells.
CELLTYPIST_COARSE_MAP: dict[str, str] = {
    # ── CD4+ T cells ──────────────────────────────────────────────────
    "CD4+ T cells":                "CD4+ T cells",
    "CD4+ TCM":                    "CD4+ T cells",    # Central memory CD4
    "CD4+ TEM":                    "CD4+ T cells",    # Effector memory CD4
    "CD4+ Treg":                   "CD4+ T cells",    # Regulatory T (CD4 lineage)
    "Tcm/Naive helper T cells":    "CD4+ T cells",    # Naive/central memory helper
    "T(agonist)":                  "CD4+ T cells",    # Agonist-stimulated (CD4-enriched)
    "MAIT cells":                  "CD4+ T cells",    # Mucosal-associated invariant T

    # ── CD8+ T cells ──────────────────────────────────────────────────
    "CD8+ T cells":                "CD8+ T cells",
    "CD8+ TCM":                    "CD8+ T cells",    # Central memory CD8
    "CD8+ TEM":                    "CD8+ T cells",    # Effector memory CD8
    "CD8+ TEx":                    "CD8+ T cells",    # Exhausted CD8
    "Tem/Trm cytotoxic T cells":   "CD8+ T cells",    # Tissue-resident cytotoxic
    "gdT":                         "CD8+ T cells",    # Gamma-delta T (cytotoxic)

    # ── NK cells ──────────────────────────────────────────────────────
    "NK cells":                    "NK cells",
    "NK_CD56bright":               "NK cells",        # CD56-bright (immature NK)
    "CD16+ NK cells":              "NK cells",        # CD56dim CD16+ (mature NK)
    "NKT cells":                   "NK cells",        # NKT (NK transcriptional programme)
    "ILC":                         "NK cells",        # Innate lymphoid cells (NK lineage)

    # ── B cells ───────────────────────────────────────────────────────
    "B cells":                     "B cells",
    "B naive":                     "B cells",
    "B memory":                    "B cells",
    "B intermediate":              "B cells",
    "Age-associated B cells":      "B cells",         # Atypical memory B cells
    "Plasma cells":                "B cells",         # Terminally differentiated B
    "Plasmablasts":                "B cells",         # Transitional plasma cells

    # ── CD14+ Monocytes ───────────────────────────────────────────────
    "CD14+ Monocytes":             "CD14+ Monocytes",
    "Classical monocytes":         "CD14+ Monocytes", # Synonymous (HCA nomenclature)
    "Intermediate monocytes":      "CD14+ Monocytes", # CD14+CD16+ intermediate

    # ── FCGR3A+ Monocytes ─────────────────────────────────────────────
    "FCGR3A+ Monocytes":           "FCGR3A+ Monocytes",
    "CD16+ Monocytes":             "FCGR3A+ Monocytes", # CD16+ = FCGR3A+
    "Non-classical monocytes":     "FCGR3A+ Monocytes", # Synonymous

    # ── Dendritic cells ───────────────────────────────────────────────
    "Dendritic cells":             "Dendritic cells",
    "DC":                          "Dendritic cells",   # Generic DC label
    "DC1":                         "Dendritic cells",
    "DC2":                         "Dendritic cells",
    "cDC1":                        "Dendritic cells",   # Classical DC type 1
    "cDC2":                        "Dendritic cells",   # Classical DC type 2
    "AS DCs":                      "Dendritic cells",   # Axl-Siglec DCs
    "Migratory DCs":               "Dendritic cells",

    # ── Plasmacytoid DCs ──────────────────────────────────────────────
    "pDC":                         "Plasmacytoid DCs",
    "Plasmacytoid dendritic cells": "Plasmacytoid DCs",

    # ── Platelets / Megakaryocytes ────────────────────────────────────
    "Platelets":                   "Platelets",
    "Megakaryocytes":              "Platelets",          # MK precursors
    "Megakaryocytes/platelets":    "Platelets",          # Combined label

    # ── HSPCs ─────────────────────────────────────────────────────────
    "HSC":                         "HSPCs",
    "HSPCs":                       "HSPCs",
    "Progenitors":                 "HSPCs",
    "Cycling cells":               "HSPCs",              # Proliferating progenitors

    # ── Unmappable / rare / artefact ──────────────────────────────────
    # Retained as Unknown so they appear explicitly in the confusion matrix
    # rather than being silently misclassified.
    "Doublets":                    "Unknown",
    "Mast cells":                  "Unknown",            # Not expected in PBMC 3k
    "Basophils":                   "Unknown",
    "Neutrophils":                 "Unknown",
    "Eosinophils":                 "Unknown",
}


def _get_log_norm_matrix(adata: ad.AnnData) -> tuple[np.ndarray, list[str]]:
    """Extract log-normalised count matrix from adata.raw.

    Returns:
        Tuple of (dense numpy array, list of gene names).
    """
    if adata.raw is not None:
        X = adata.raw.X
        gene_names = list(adata.raw.var_names)
    else:
        X = adata.X
        gene_names = list(adata.var_names)
        logger.warning("adata.raw not set; using scaled .X for annotation scoring")

    if issparse(X):
        X = X.toarray()
    return X.astype(np.float32), gene_names


def annotate_manual(
    adata: ad.AnnData,
    groupby: str = "leiden",
    markers: dict[str, list[str]] | None = None,
) -> ad.AnnData:
    """Assign cell types to clusters using canonical marker gene scoring.

    For each cluster, computes the mean log-normalised expression of
    each cell type's canonical marker gene set. The cell type whose
    markers have the highest mean expression (normalised by the number
    of markers) is assigned to that cluster.

    This is a simple but interpretable approach that mirrors what a
    computational biologist does manually when reading a dot plot.

    Args:
        adata: Clustered AnnData with leiden labels and .raw set.
        groupby: Column in .obs with cluster labels.
        markers: Canonical marker dictionary. Defaults to PBMC_MARKERS.

    Returns:
        AnnData with 'celltype_manual' column added to .obs.
    """
    markers = markers or PBMC_MARKERS
    X, gene_names = _get_log_norm_matrix(adata)
    gene_index = {g: i for i, g in enumerate(gene_names)}
    clusters = adata.obs[groupby].astype(str)

    cluster_labels: dict[str, str] = {}

    for cluster_id in sorted(clusters.unique()):
        mask = clusters == cluster_id
        cluster_X = X[mask, :]

        best_type = "Unknown"
        best_score = -np.inf

        for cell_type, type_markers in markers.items():
            # Only use markers that exist in this dataset
            available = [g for g in type_markers if g in gene_index]
            if not available:
                continue

            idx = [gene_index[g] for g in available]
            # Mean expression across all cells in cluster, then mean across markers
            score = float(cluster_X[:, idx].mean())

            if score > best_score:
                best_score = score
                best_type = cell_type

        cluster_labels[cluster_id] = best_type
        logger.debug(
            "cluster_%s -> %s (score=%.4f)", cluster_id, best_type, best_score
        )

    adata.obs["celltype_manual"] = clusters.map(cluster_labels).astype("category")

    n_types = adata.obs["celltype_manual"].nunique()
    logger.info(
        "manual_annotation_complete: %d clusters -> %d cell types",
        len(cluster_labels),
        n_types,
    )
    return adata


def annotate_celltypist(adata: ad.AnnData) -> ad.AnnData:
    """Annotate cells using CellTypist pre-trained reference model.

    Downloads only Immune_All_Low.pkl if not already cached (not all
    61 models). Runs per-cell prediction with majority voting so each
    Leiden cluster receives a single consensus label.

    CellTypist requires log1p-normalised counts (target_sum=10000) in
    the input .X matrix. We build a clean query AnnData from adata.raw
    to avoid passing the scaled matrix that lives in adata.X after
    preprocessing.

    Stores fine-grained labels in 'celltype_reference' and confidence
    scores in 'celltype_reference_conf'.

    Args:
        adata: AnnData with .raw set to log-normalised counts and
               leiden clustering in .obs.

    Returns:
        AnnData with reference annotation columns added to .obs.
    """
    try:
        import celltypist
        from celltypist import models
    except ImportError:
        logger.error("celltypist not installed. Run: pip install celltypist")
        raise

    # Download only the required model, not all 61
    logger.info("downloading_celltypist_model: Immune_All_Low.pkl (if not cached)")
    models.download_models(model=["Immune_All_Low.pkl"], force_update=False)
    model = models.Model.load(model="Immune_All_Low.pkl")

    # Build a clean query AnnData with log1p-normalised counts.
    # adata.raw holds raw integer counts (saved before normalize_total).
    # adata.layers['log_norm'] holds the log1p-normalised matrix saved
    # before sc.pp.scale() overwrote adata.X. CellTypist validates that
    # expm1(X).sum(axis=1) ≈ 10000 per cell — only log_norm satisfies this.
    if "log_norm" not in adata.layers:
        raise ValueError(
            "adata.layers['log_norm'] not found. "
            "Ensure preprocessing.normalize() has been called — it saves "
            "log1p-normalised counts to this layer before scaling."
        )

    query = ad.AnnData(
        X=adata.layers["log_norm"].copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    logger.info("running_celltypist_prediction: majority_voting=True")
    predictions = celltypist.annotate(
        query,
        model=model,
        majority_voting=True,
        over_clustering="leiden",
    )

    result = predictions.to_adata()
    adata.obs["celltype_reference"] = result.obs["majority_voting"].values
    adata.obs["celltype_reference_conf"] = result.obs["conf_score"].values

    n_types = adata.obs["celltype_reference"].nunique()
    logger.info(
        "celltypist_complete: %d fine-grained cell types predicted", n_types
    )
    return adata


def harmonise_labels(adata: ad.AnnData) -> ad.AnnData:
    """Map fine-grained CellTypist labels to coarse PBMC categories.

    CellTypist returns labels like 'Classical monocytes'. This maps
    them to 'CD14+ Monocytes' for fair comparison with manual annotation.
    Unmapped labels are kept as-is.

    Args:
        adata: AnnData with 'celltype_reference' column set.

    Returns:
        AnnData with 'celltype_reference_coarse' column added.
    """
    if "celltype_reference" not in adata.obs.columns:
        raise ValueError(
            "celltype_reference column missing. Run annotate_celltypist() first."
        )

    coarse = adata.obs["celltype_reference"].map(
        lambda x: CELLTYPIST_COARSE_MAP.get(str(x), str(x))
    )
    adata.obs["celltype_reference_coarse"] = coarse.astype("category")

    n_coarse = adata.obs["celltype_reference_coarse"].nunique()
    logger.info("label_harmonisation_complete: %d coarse cell types", n_coarse)
    return adata


def compare_annotations(adata: ad.AnnData) -> pd.DataFrame:
    """Compare manual and reference annotations at the coarse level.

    Computes a confusion matrix and summary statistics comparing
    celltype_manual vs celltype_reference_coarse. Uses only cells
    where both labels are available.

    Returns:
        DataFrame confusion matrix (manual as rows, reference as columns).
    """
    required = ["celltype_manual", "celltype_reference_coarse"]
    for col in required:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing column: {col}. Run full annotation pipeline.")

    manual = adata.obs["celltype_manual"].astype(str)
    reference = adata.obs["celltype_reference_coarse"].astype(str)

    confusion = pd.crosstab(
        manual,
        reference,
        rownames=["manual"],
        colnames=["reference"],
    )

    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(manual, reference)

    logger.info(
        "annotation_comparison: kappa=%.4f, n_cells=%d", kappa, len(manual)
    )
    return confusion


def run_annotation(adata: ad.AnnData) -> ad.AnnData:
    """Full annotation pipeline: manual scoring + CellTypist + harmonise.

    Args:
        adata: Clustered AnnData with leiden labels and .raw set.

    Returns:
        AnnData with celltype_manual, celltype_reference, and
        celltype_reference_coarse columns in .obs.
    """
    adata = annotate_manual(adata)
    adata = annotate_celltypist(adata)
    adata = harmonise_labels(adata)
    return adata
