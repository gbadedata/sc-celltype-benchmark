"""Clustering and dimensionality reduction.

Builds the neighbourhood graph on the PCA embedding, runs Leiden
community detection, and computes a UMAP embedding for visualisation.

Steps:
1. k-nearest-neighbour graph on X_pca.
2. Leiden clustering (resolution tunable via settings).
3. UMAP embedding for 2D visualisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anndata as ad
import scanpy as sc

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ClusteringReport:
    """Summary statistics from clustering.

    Attributes:
        n_cells: Number of cells clustered.
        n_clusters: Number of Leiden clusters found.
        leiden_resolution: Resolution parameter used.
        n_neighbors: k for the kNN graph.
        min_cluster_size: Size of the smallest cluster.
        max_cluster_size: Size of the largest cluster.
        median_cluster_size: Median cluster size.
    """

    n_cells: int
    n_clusters: int
    leiden_resolution: float
    n_neighbors: int
    min_cluster_size: int
    max_cluster_size: int
    median_cluster_size: float


def build_neighbor_graph(adata: ad.AnnData) -> ad.AnnData:
    """Compute k-nearest-neighbour graph on PCA space.

    Constructs the kNN graph used by both Leiden clustering and UMAP.
    Uses the X_pca embedding with n_pcs components from settings.

    Args:
        adata: AnnData with X_pca embedding in .obsm.

    Returns:
        AnnData with connectivity graph stored in .obsp and .uns.
    """
    sc.pp.neighbors(
        adata,
        n_neighbors=settings.n_neighbors,
        n_pcs=settings.n_pcs,
        random_state=settings.random_seed,
    )

    logger.info(
        "neighbor_graph_built: k=%d, n_pcs=%d",
        settings.n_neighbors,
        settings.n_pcs,
    )
    return adata


def run_leiden(adata: ad.AnnData) -> ad.AnnData:
    """Run Leiden community detection on the neighbour graph.

    Leiden is the standard clustering algorithm for single-cell data,
    superseding Louvain. It guarantees well-connected communities.
    Cluster assignments are stored in adata.obs['leiden'].

    Args:
        adata: AnnData with neighbour graph computed.

    Returns:
        AnnData with Leiden cluster labels in .obs['leiden'].
    """
    sc.tl.leiden(
        adata,
        resolution=settings.leiden_resolution,
        random_state=settings.random_seed,
        flavor="igraph",
        n_iterations=2,
    )

    n_clusters = adata.obs["leiden"].nunique()
    cluster_sizes = adata.obs["leiden"].value_counts()

    logger.info(
        "leiden_complete: %d clusters (resolution=%.2f, min=%d, max=%d, median=%.0f)",
        n_clusters,
        settings.leiden_resolution,
        cluster_sizes.min(),
        cluster_sizes.max(),
        cluster_sizes.median(),
    )
    return adata


def compute_umap(adata: ad.AnnData) -> ad.AnnData:
    """Compute UMAP embedding for visualisation.

    UMAP projects the high-dimensional kNN graph into 2D for
    visualisation. Run after build_neighbor_graph.

    Args:
        adata: AnnData with neighbour graph computed.

    Returns:
        AnnData with 2D UMAP coordinates in .obsm['X_umap'].
    """
    sc.tl.umap(adata, random_state=settings.random_seed)

    logger.info("umap_computed: embedding shape %s", adata.obsm["X_umap"].shape)
    return adata


def run_clustering(adata: ad.AnnData) -> tuple[ad.AnnData, ClusteringReport]:
    """Full clustering pipeline: neighbours, Leiden, UMAP.

    Args:
        adata: Preprocessed AnnData with X_pca embedding.

    Returns:
        Tuple of (clustered AnnData, ClusteringReport).
    """
    adata = build_neighbor_graph(adata)
    adata = run_leiden(adata)
    adata = compute_umap(adata)

    cluster_sizes = adata.obs["leiden"].value_counts()
    report = ClusteringReport(
        n_cells=adata.n_obs,
        n_clusters=int(cluster_sizes.shape[0]),
        leiden_resolution=settings.leiden_resolution,
        n_neighbors=settings.n_neighbors,
        min_cluster_size=int(cluster_sizes.min()),
        max_cluster_size=int(cluster_sizes.max()),
        median_cluster_size=float(cluster_sizes.median()),
    )

    logger.info(
        "clustering_complete: %d cells -> %d clusters",
        report.n_cells,
        report.n_clusters,
    )
    return adata, report
