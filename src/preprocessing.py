"""Preprocessing: normalisation, HVG selection, scaling, PCA.

Transforms raw counts into a reduced representation suitable for
clustering and visualisation.
"""

from __future__ import annotations

import logging

import anndata as ad

logger = logging.getLogger(__name__)


def normalize(adata: ad.AnnData) -> ad.AnnData:
    """Library-size normalisation and log1p transformation.

    Normalises each cell to 10,000 total counts, then applies log1p.
    Stores raw counts in adata.raw before transformation.
    """
    raise NotImplementedError("Phase 3")


def select_hvg(adata: ad.AnnData) -> ad.AnnData:
    """Select highly variable genes.

    Uses scanpy's Seurat v3 flavour. Marks HVGs in adata.var['highly_variable'].
    """
    raise NotImplementedError("Phase 3")


def scale_and_pca(adata: ad.AnnData) -> ad.AnnData:
    """Scale to unit variance, then compute PCA.

    Scales gene expression to zero mean and unit variance.
    Computes PCA with n_pcs components from settings.
    """
    raise NotImplementedError("Phase 3")


def run_preprocessing(adata: ad.AnnData) -> ad.AnnData:
    """Full preprocessing pipeline: normalize, HVG, scale, PCA.

    Returns preprocessed AnnData ready for clustering.
    """
    raise NotImplementedError("Phase 3")
