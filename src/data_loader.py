"""Data acquisition and loading.

Downloads scRNA-seq datasets and loads them into AnnData objects.
Default dataset: 10x Genomics PBMC 3k (via scanpy, hosted on figshare).
Also supports loading any local h5ad or h5 file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import scanpy as sc

from config.settings import settings

logger = logging.getLogger(__name__)


def download_pbmc3k(dest_dir: Path | str | None = None) -> ad.AnnData:
    """Download the PBMC 3k dataset via scanpy.

    Uses scanpy.datasets.pbmc3k() which fetches the 10x Genomics
    PBMC 3k filtered gene-barcode matrix from figshare. Returns
    raw integer counts (no preprocessing applied).

    The download is cached by scanpy in its data directory. We also
    save a local h5ad copy in our data/ folder for reproducibility.

    Args:
        dest_dir: Directory to cache the h5ad copy. Defaults to settings.data_dir.

    Returns:
        AnnData with raw counts (~2,700 cells x ~32,000 genes).
    """
    dest_dir = Path(dest_dir) if dest_dir else settings.data_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dest_dir / "pbmc3k_raw.h5ad"

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info("dataset_cached: %s (%.1f MB)", cache_path, size_mb)
        return ad.read_h5ad(cache_path)

    logger.info("downloading_pbmc3k via scanpy (figshare)")
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()

    # Cache locally
    adata.write_h5ad(cache_path)
    logger.info(
        "downloaded_and_cached: %d cells x %d genes -> %s",
        adata.n_obs,
        adata.n_vars,
        cache_path,
    )
    return adata


def load_10x_h5(filepath: str | Path) -> ad.AnnData:
    """Load a 10x Genomics h5 file into AnnData.

    Uses scanpy.read_10x_h5 which handles the 10x-specific HDF5
    structure (feature-barcode matrix with gene symbols and IDs).

    Args:
        filepath: Path to the .h5 file.

    Returns:
        AnnData with raw counts, gene symbols as var_names.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    logger.info("loading_10x_h5: %s", filepath)
    adata = sc.read_10x_h5(str(filepath))
    adata.var_names_make_unique()

    logger.info("loaded: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def load_h5ad(filepath: str | Path) -> ad.AnnData:
    """Load an h5ad file into AnnData.

    Args:
        filepath: Path to the .h5ad file.

    Returns:
        AnnData object.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    logger.info("loading_h5ad: %s", filepath)
    adata = ad.read_h5ad(filepath)
    logger.info("loaded: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def get_dataset(filepath: str | Path | None = None) -> ad.AnnData:
    """Download (if needed) and load the dataset.

    This is the main entry point.

    - If filepath is provided, loads from that file directly
      (.h5 or .h5ad format).
    - If no filepath, downloads the PBMC 3k dataset via scanpy
      and caches it locally.

    Args:
        filepath: Optional explicit path to a local dataset file.

    Returns:
        AnnData with raw counts.
    """
    if filepath is not None:
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        if suffix == ".h5":
            return load_10x_h5(filepath)
        elif suffix == ".h5ad":
            return load_h5ad(filepath)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Expected .h5 or .h5ad"
            )

    return download_pbmc3k()
