"""Visualisation for single-cell analysis and benchmark results.

Generates publication-quality figures saved to evidence/figures/.
All functions accept an AnnData and write PNG files; they do not
display interactively (no plt.show()).

Note: requires matplotlib >= 3.7. Uses mpl.colormaps API (3.9+
compatible) instead of the deprecated plt.cm.get_cmap().
"""

from __future__ import annotations

import logging

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config.settings import settings

mpl.use("Agg")  # non-interactive backend; no display required

logger = logging.getLogger(__name__)
FIGURE_DIR = settings.evidence_dir / "figures"
DPI = 150


def _get_colours(n: int, palette: str = "tab20") -> list:
    """Return n distinct colours from a matplotlib colormap.

    Compatible with matplotlib >= 3.7 and >= 3.9 (new colormaps API).
    """
    cmap = mpl.colormaps[palette].resampled(max(n, 2))
    return [cmap(i) for i in range(n)]


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to evidence/figures/ and close it."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("figure_saved: %s", path)


def plot_qc_metrics(adata: ad.AnnData) -> None:
    """Violin plots of QC metrics: n_genes, total_counts, pct_mito."""
    metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    titles = ["Genes per cell", "UMI counts per cell", "Mitochondrial %"]
    colours = ["#4C8BE2", "#54B06D", "#E25C4C"]

    # Only plot metrics that exist (QC may not have run for some inputs)
    available = [(m, t, c) for m, t, c in zip(metrics, titles, colours)
                 if m in adata.obs.columns]
    if not available:
        logger.warning("No QC metric columns found; skipping QC plot")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, (metric, title, colour) in zip(axes, available):
        vals = adata.obs[metric].values
        parts = ax.violinplot(vals, positions=[0], showmedians=True,
                              showextrema=True)
        for pc in parts.get("bodies", []):
            pc.set_facecolor(colour)
            pc.set_alpha(0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_ylabel(metric, fontsize=9)
        median = float(np.median(vals))
        ax.text(0.05, 0.97, f"median = {median:.0f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    fig.suptitle(f"QC metrics — {adata.n_obs:,} cells after filtering",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "01_qc_metrics")


def plot_umap_clusters(adata: ad.AnnData) -> None:
    """UMAP coloured by Leiden cluster."""
    if "X_umap" not in adata.obsm or "leiden" not in adata.obs.columns:
        logger.warning("X_umap or leiden not found; skipping cluster UMAP")
        return

    umap = adata.obsm["X_umap"]
    clusters = adata.obs["leiden"].astype(str)
    unique = sorted(clusters.unique(), key=lambda x: int(x))
    colours = _get_colours(len(unique))

    fig, ax = plt.subplots(figsize=(8, 7))
    for cluster, colour in zip(unique, colours):
        mask = clusters == cluster
        ax.scatter(umap[mask, 0], umap[mask, 1],
                   c=[colour], s=5, alpha=0.7, label=f"Cluster {cluster} (n={mask.sum()})")

    ax.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left",
              framealpha=0.9)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(f"Leiden clusters (resolution={settings.leiden_resolution})",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    _save(fig, "02_umap_leiden_clusters")


def plot_umap_celltypes(
    adata: ad.AnnData,
    label_col: str = "celltype_manual",
    title_suffix: str = "Manual annotation",
    fig_name: str = "03_umap_celltypes_manual",
) -> None:
    """UMAP coloured by cell-type annotation."""
    if "X_umap" not in adata.obsm or label_col not in adata.obs.columns:
        logger.warning("Cannot plot cell-type UMAP: missing embedding or column '%s'", label_col)
        return

    umap = adata.obsm["X_umap"]
    labels = adata.obs[label_col].astype(str)
    unique = sorted(labels.unique())
    colours = _get_colours(len(unique))
    colour_map = dict(zip(unique, colours))

    fig, ax = plt.subplots(figsize=(9, 7))
    for cell_type in unique:
        mask = labels == cell_type
        ax.scatter(umap[mask, 0], umap[mask, 1],
                   c=[colour_map[cell_type]], s=5, alpha=0.7,
                   label=f"{cell_type} (n={mask.sum()})")

    ax.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left",
              framealpha=0.9)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(f"Cell-type annotation — {title_suffix}",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    _save(fig, fig_name)


def plot_annotation_comparison(adata: ad.AnnData) -> None:
    """Side-by-side UMAP: manual annotation vs CellTypist reference."""
    if "X_umap" not in adata.obsm:
        return

    cols = ["celltype_manual", "celltype_reference_coarse"]
    titles = ["Manual (marker scoring)", "CellTypist (Immune_All_Low)"]
    if not all(c in adata.obs.columns for c in cols):
        logger.warning("Cannot plot comparison: annotation columns missing")
        return

    all_types = sorted(
        set(adata.obs[cols[0]].astype(str).unique()) |
        set(adata.obs[cols[1]].astype(str).unique())
    )
    colours = _get_colours(len(all_types), "tab20")
    colour_map = dict(zip(all_types, colours))
    umap = adata.obsm["X_umap"]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, col, title in zip(axes, cols, titles):
        labels = adata.obs[col].astype(str)
        for cell_type in sorted(labels.unique()):
            mask = labels == cell_type
            ax.scatter(umap[mask, 0], umap[mask, 1],
                       c=[colour_map[cell_type]], s=5, alpha=0.7,
                       label=f"{cell_type} (n={mask.sum()})")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=3, fontsize=7, bbox_to_anchor=(1.01, 1),
                  loc="upper left", framealpha=0.9)

    fig.suptitle("Annotation comparison: manual marker scoring vs CellTypist reference",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_annotation_comparison")


def plot_marker_dotplot(adata: ad.AnnData) -> None:
    """Dot plot: canonical marker expression across annotated cell types."""
    if "celltype_manual" not in adata.obs.columns:
        return

    import scipy.sparse as sp

    from src.markers import PBMC_MARKERS

    # Use log_norm layer (log1p-normalised) for biologically interpretable values
    if "log_norm" in adata.layers:
        X = adata.layers["log_norm"]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    # Select canonical markers present in the dataset, deduplicated, capped at 40
    markers_flat: list[str] = []
    for genes in PBMC_MARKERS.values():
        markers_flat.extend([g for g in genes if g in adata.var_names])
    seen: set[str] = set()
    markers_dedup = [g for g in markers_flat if not (g in seen or seen.add(g))]
    markers_use = markers_dedup[:40]

    if not markers_use:
        logger.warning("No canonical markers found in dataset; skipping dotplot")
        return

    cell_types = adata.obs["celltype_manual"].astype(str)
    type_order = sorted(cell_types.unique())
    gene_idx = [list(adata.var_names).index(g) for g in markers_use]

    mean_expr = np.zeros((len(type_order), len(markers_use)))
    pct_expr = np.zeros((len(type_order), len(markers_use)))

    for i, ct in enumerate(type_order):
        mask = (cell_types == ct).values
        sub = X[mask, :][:, gene_idx]
        mean_expr[i] = sub.mean(axis=0)
        pct_expr[i] = (sub > 0).mean(axis=0)

    # Normalise per-gene to [0, 1] for colour scaling
    gene_max = mean_expr.max(axis=0, keepdims=True)
    gene_max[gene_max == 0] = 1
    norm_expr = mean_expr / gene_max

    fig, ax = plt.subplots(figsize=(max(12, len(markers_use) * 0.42),
                                    max(4, len(type_order) * 0.65)))
    for i, ct in enumerate(type_order):
        for j in range(len(markers_use)):
            size = pct_expr[i, j] * 220
            colour = plt.cm.Reds(norm_expr[i, j])
            ax.scatter(j, i, s=size, c=[colour],
                       linewidths=0.3, edgecolors="grey", alpha=0.9)

    ax.set_xticks(range(len(markers_use)))
    ax.set_xticklabels(markers_use, rotation=90, fontsize=8)
    ax.set_yticks(range(len(type_order)))
    ax.set_yticklabels(type_order, fontsize=9)
    ax.set_xlabel("Canonical marker genes", fontsize=10)
    ax.set_title(
        "Marker gene expression by annotated cell type\n"
        "(dot size = % cells expressing  ·  colour = normalised mean expression)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.15, linestyle="--")
    plt.tight_layout()
    _save(fig, "05_marker_dotplot")


def plot_confusion_matrix(result: object) -> None:
    """Heatmap confusion matrix: true labels (rows) vs predicted (cols)."""
    cm = result.confusion_df  # type: ignore[attr-defined]
    if cm.empty:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(cm.columns) * 0.95),
                                    max(6, len(cm.index) * 0.85)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_xlabel("Predicted (manual annotation)", fontsize=10)
    ax.set_ylabel("True (CellTypist reference)", fontsize=10)
    ax.set_title(
        f"Confusion matrix — {result.n_cells_evaluated} test cells\n"  # type: ignore[attr-defined]
        f"Weighted F1 = {result.weighted_f1:.3f}   Cohen's κ = {result.cohens_kappa:.3f}",  # type: ignore[attr-defined]
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "06_confusion_matrix")


def plot_benchmark_tier_summary(tier_results: dict) -> None:
    """Bar chart of weighted F1 scores by difficulty tier."""
    if not tier_results:
        return

    tiers = sorted(tier_results.keys())
    f1_scores = [tier_results[t].weighted_f1 for t in tiers]
    n_cells = [tier_results[t].n_cells_evaluated for t in tiers]
    tier_colours = {"easy": "#54B06D", "medium": "#E2A43C", "hard": "#E25C4C"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        tiers,
        f1_scores,
        color=[tier_colours.get(t, "#888888") for t in tiers],
        edgecolor="white",
        linewidth=1.5,
        width=0.5,
    )
    for bar, score, n in zip(bars, f1_scores, n_cells):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.03, 1.07),
            f"F1 = {score:.3f}\n(n = {n})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Difficulty tier", fontsize=11)
    ax.set_ylabel("Weighted F1", fontsize=11)
    ax.set_title("Benchmark performance by annotation difficulty tier",
                 fontsize=12, fontweight="bold")
    ax.axhline(0.9, color="grey", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(tiers) - 0.48, 0.915, "F1 = 0.90", fontsize=8, color="grey")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout()
    _save(fig, "07_benchmark_tier_summary")


def plot_per_type_f1(result: object) -> None:
    """Horizontal bar chart of per-cell-type F1 scores."""
    df = result.per_type_metrics.copy()  # type: ignore[attr-defined]
    df = df[df["support"] > 0].sort_values("f1", ascending=True)
    if df.empty:
        return

    colours = [
        "#54B06D" if f >= 0.9 else "#E2A43C" if f >= 0.6 else "#E25C4C"
        for f in df["f1"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.52)))
    bars = ax.barh(df["cell_type"], df["f1"], color=colours,
                   edgecolor="white", linewidth=1.2)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            min(bar.get_width() + 0.01, 1.01),
            bar.get_y() + bar.get_height() / 2,
            f"F1 = {row['f1']:.3f}  (n = {int(row['support'])})",
            va="center", fontsize=8.5,
        )

    ax.set_xlim(0, 1.32)
    ax.set_xlabel("F1 score", fontsize=10)
    ax.set_title(
        "Per-cell-type F1 — manual annotation vs CellTypist oracle",
        fontsize=11, fontweight="bold",
    )
    ax.axvline(0.9, color="grey", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout()
    _save(fig, "08_per_type_f1")


def generate_all_figures(
    adata: ad.AnnData,
    result: object,
    validator_results: list,
) -> None:
    """Generate all eight evidence figures and save to evidence/figures/."""
    logger.info("generating_figures: output dir %s", FIGURE_DIR)

    plot_qc_metrics(adata)
    plot_umap_clusters(adata)
    plot_umap_celltypes(
        adata,
        label_col="celltype_manual",
        title_suffix="Manual annotation",
        fig_name="03_umap_celltypes_manual",
    )
    plot_umap_celltypes(
        adata,
        label_col="celltype_reference_coarse",
        title_suffix="CellTypist reference",
        fig_name="03b_umap_celltypes_reference",
    )
    plot_annotation_comparison(adata)
    plot_marker_dotplot(adata)
    plot_confusion_matrix(result)

    if hasattr(result, "tier_results") and result.tier_results:
        plot_benchmark_tier_summary(result.tier_results)

    plot_per_type_f1(result)
    logger.info("all_figures_generated: saved to %s", FIGURE_DIR)
