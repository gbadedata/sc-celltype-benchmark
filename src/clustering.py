"""Clustering and dimensionality reduction.

Builds the neighbourhood graph, runs Leiden clustering, and computes
UMAP embeddings for visualisation.
"""

from __future__ import annotations

import logging

import anndata as ad

logger = logging.getLogger(__name__)


def build_neighbor_graph(adata: ad.AnnData) -> ad.AnnData:
    """Compute k-nearest-neighbour graph on PCA space.

    Uses n_neighbors and n_pcs from settings.
    """
    raise NotImplementedError("Phase 4")


def run_leiden(adata: ad.AnnData) -> ad.AnnData:
    """Run Leiden community detection on the neighbour graph.

    Stores cluster assignments in adata.obs['leiden'].
    Uses resolution from settings.
    """
    raise NotImplementedError("Phase 4")


def compute_umap(adata: ad.AnnData) -> ad.AnnData:
    """Compute UMAP embedding for visualisation.

    Stores coordinates in adata.obsm['X_umap'].
    """
    raise NotImplementedError("Phase 4")


def run_clustering(adata: ad.AnnData) -> ad.AnnData:
    """Full clustering pipeline: neighbours, Leiden, UMAP.

    Returns AnnData with clustering and embedding computed.
    """
    raise NotImplementedError("Phase 4")
