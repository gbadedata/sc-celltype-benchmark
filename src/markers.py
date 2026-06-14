"""Marker gene detection.

Identifies differentially expressed genes per Leiden cluster using
the Wilcoxon rank-sum test. Provides marker gene tables and
canonical PBMC marker definitions for downstream annotation.

The Wilcoxon test is preferred over the t-test in single-cell analysis
because it is non-parametric (no normality assumption), robust to
outliers, and produces results that correlate well with fold-change
while accounting for zero inflation.
"""

from __future__ import annotations

import logging

import anndata as ad
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)

# Canonical PBMC marker genes curated from literature.
# Each key is a cell type; values are genes expected to be
# highly expressed in that population.
# Sources: Zheng et al. 2017 (Nature Communications),
#          Seurat PBMC tutorial, Human Cell Atlas PBMC references.
PBMC_MARKERS: dict[str, list[str]] = {
    "CD4+ T cells":        ["CD3D", "CD3E", "IL7R", "CD4"],
    "CD8+ T cells":        ["CD3D", "CD3E", "CD8A", "CD8B", "GZMK"],
    "NK cells":            ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "B cells":             ["CD79A", "CD79B", "MS4A1", "CD19"],
    "CD14+ Monocytes":     ["CD14", "LYZ", "S100A9", "S100A8"],
    "FCGR3A+ Monocytes":   ["FCGR3A", "MS4A7", "IFITM3"],
    "Dendritic cells":     ["FCER1A", "CST3", "CLEC10A"],
    "Plasmacytoid DCs":    ["IL3RA", "CLEC4C", "IRF7", "TCF4"],
    "Platelets":           ["PPBP", "PF4", "GP9"],
    "HSPCs":               ["CD34", "SPINK2", "CRHBP"],
}


def detect_markers(
    adata: ad.AnnData,
    groupby: str = "leiden",
    method: str = "wilcoxon",
    n_genes: int = 50,
) -> ad.AnnData:
    """Run differential expression to find marker genes per cluster.

    Uses Wilcoxon rank-sum test (one-vs-rest) with multiple testing
    correction (Benjamini-Hochberg). Results are stored in
    adata.uns['rank_genes_groups'].

    Args:
        adata: Clustered AnnData with .obs[groupby] labels.
        groupby: Column in .obs defining groups (default: 'leiden').
        method: DE test method (default: 'wilcoxon').
        n_genes: Number of top genes to retain per group.

    Returns:
        AnnData with DE results stored in .uns['rank_genes_groups'].
    """
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        pts=True,            # compute % of cells expressing the gene
        key_added="rank_genes_groups",
    )

    n_groups = len(adata.uns["rank_genes_groups"]["names"].dtype.names)
    logger.info(
        "markers_detected: %d groups, top %d genes each, method=%s",
        n_groups,
        n_genes,
        method,
    )
    return adata


def get_top_markers(
    adata: ad.AnnData,
    n_genes: int = 10,
    key: str = "rank_genes_groups",
) -> pd.DataFrame:
    """Extract top N marker genes per cluster as a tidy DataFrame.

    Args:
        adata: AnnData with rank_genes_groups results.
        n_genes: Number of top genes to return per cluster.
        key: Key in adata.uns where results are stored.

    Returns:
        Tidy DataFrame with columns:
        cluster, rank, gene, score, pval, pval_adj, log2fc,
        pct_expressed_group, pct_expressed_rest.
    """
    result = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        key=key,
        pval_cutoff=None,
        log2fc_min=None,
    )

    # sc.get.rank_genes_groups_df returns all groups; take top n per group
    result = (
        result
        .groupby("group", observed=True)
        .head(n_genes)
        .reset_index(drop=True)
    )

    result = result.rename(columns={"group": "cluster"})

    logger.info(
        "top_markers_extracted: %d clusters x %d genes = %d rows",
        result["cluster"].nunique(),
        n_genes,
        len(result),
    )
    return result


def score_marker_overlap(
    adata: ad.AnnData,
    detected_markers: dict[str, list[str]],
    reference: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Score overlap between detected cluster markers and canonical reference.

    For each cluster, computes the Jaccard similarity between its
    detected marker genes and the canonical markers of every reference
    cell type. The best-matching reference type and its score are returned.

    This is used during manual annotation to find the closest canonical
    cell type for each cluster.

    Args:
        adata: AnnData (used to verify gene availability).
        detected_markers: Dict mapping cluster ID -> list of top marker genes.
        reference: Dict mapping cell type -> canonical markers.
            Defaults to PBMC_MARKERS.

    Returns:
        DataFrame with columns:
        cluster, best_match, jaccard_score, matched_genes, n_matched.
    """
    reference = reference or PBMC_MARKERS

    rows = []
    for cluster, detected in detected_markers.items():
        detected_set = set(detected)
        best_type = "Unknown"
        best_score = 0.0
        best_matched: list[str] = []

        for cell_type, canonical in reference.items():
            canonical_set = set(canonical)
            intersection = detected_set & canonical_set
            union = detected_set | canonical_set
            score = len(intersection) / len(union) if union else 0.0

            if score > best_score:
                best_score = score
                best_type = cell_type
                best_matched = sorted(intersection)

        rows.append(
            {
                "cluster": cluster,
                "best_match": best_type,
                "jaccard_score": round(best_score, 4),
                "matched_genes": best_matched,
                "n_matched": len(best_matched),
            }
        )

    df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    logger.info(
        "marker_overlap_scored: %d clusters matched against %d reference types",
        len(df),
        len(reference),
    )
    return df
