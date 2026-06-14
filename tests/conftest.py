"""Shared test fixtures.

Provides small synthetic AnnData objects for fast, reproducible testing
without downloading real datasets.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def synthetic_adata() -> ad.AnnData:
    """Create a small synthetic AnnData for unit testing.

    200 cells, 500 genes, 4 cell types with known markers.
    Sparse count matrix with realistic zero inflation.
    """
    rng = np.random.default_rng(42)
    n_cells = 200
    n_genes = 500

    # Cell type assignments (ground truth)
    cell_types = np.array(
        ["CD4+ T cells"] * 60
        + ["B cells"] * 50
        + ["CD14+ Monocytes"] * 50
        + ["NK cells"] * 40
    )

    # Gene names: include canonical markers in first positions
    marker_genes = [
        "CD3D", "CD3E", "IL7R", "CD4",       # T cell markers
        "CD79A", "CD79B", "MS4A1", "CD19",    # B cell markers
        "CD14", "LYZ", "S100A9", "S100A8",    # Monocyte markers
        "NKG7", "GNLY", "KLRD1", "NCAM1",     # NK markers
    ]
    other_genes = [f"GENE_{i}" for i in range(n_genes - len(marker_genes))]
    gene_names = marker_genes + other_genes

    # Build count matrix with cell-type-specific marker expression
    counts = rng.poisson(lam=0.5, size=(n_cells, n_genes)).astype(np.float32)

    # Boost markers for their respective cell types
    type_marker_ranges = {
        "CD4+ T cells": (0, 4),
        "B cells": (4, 8),
        "CD14+ Monocytes": (8, 12),
        "NK cells": (12, 16),
    }
    for i, ct in enumerate(cell_types):
        start, end = type_marker_ranges[ct]
        counts[i, start:end] += rng.poisson(lam=8, size=end - start)

    # Add mitochondrial genes
    mito_genes = [f"MT-{g}" for g in ["CO1", "CO2", "ND1", "ND2", "ATP6"]]
    mito_counts = rng.poisson(lam=2, size=(n_cells, len(mito_genes))).astype(np.float32)
    gene_names = gene_names[: n_genes - len(mito_genes)] + mito_genes
    counts[:, n_genes - len(mito_genes):] = mito_counts

    X = sp.csr_matrix(counts)

    obs = pd.DataFrame(
        {"celltype_ground_truth": cell_types},
        index=[f"CELL_{i:04d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names[:n_genes])
    var.index.name = None

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


@pytest.fixture
def annotated_adata(synthetic_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData with annotation columns already populated.

    For testing benchmark and evaluation modules.
    """
    adata = synthetic_adata.copy()
    # Perfect manual annotation (matches ground truth)
    adata.obs["celltype_manual"] = adata.obs["celltype_ground_truth"].copy()
    # Reference annotation with some noise
    rng = np.random.default_rng(42)
    ref = adata.obs["celltype_ground_truth"].values.copy()
    noise_idx = rng.choice(len(ref), size=10, replace=False)
    ref[noise_idx] = "B cells"
    adata.obs["celltype_reference"] = ref
    adata.obs["celltype_reference_coarse"] = ref
    return adata
