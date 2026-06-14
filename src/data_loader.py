"""Data acquisition and loading.

Downloads 10x Genomics scRNA-seq datasets and loads them into AnnData objects.
Supports h5, h5ad, and 10x MTX formats. Caches downloads locally so
repeated runs do not re-download.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve

import anndata as ad
import scanpy as sc

from config.settings import settings

logger = logging.getLogger(__name__)


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Report download progress to the log."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        if block_num % 200 == 0 or pct >= 100:
            logger.info("download_progress: %.1f MB (%d%%)", mb, pct)


def download_dataset(
    url: str | None = None,
    dest_dir: Path | str | None = None,
    filename: str | None = None,
) -> Path:
    """Download dataset from URL to local directory.

    Skips download if the file already exists locally.

    Args:
        url: Download URL. Defaults to settings.dataset_url.
        dest_dir: Target directory. Defaults to settings.data_dir.
        filename: Local filename. Defaults to settings.dataset_filename.

    Returns:
        Path to the downloaded file.
    """
    url = url or settings.dataset_url
    dest_dir = Path(dest_dir) if dest_dir else settings.data_dir
    filename = filename or settings.dataset_filename

    dest_dir.mkdir(parents=True, exist_ok=True)
    filepath = dest_dir / filename

    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info("dataset_cached: %s (%.1f MB)", filepath, size_mb)
        return filepath

    logger.info("downloading_dataset: %s -> %s", url, filepath)
    try:
        urlretrieve(url, filepath, reporthook=_progress_hook)
    except Exception:
        # Clean up partial download
        if filepath.exists():
            filepath.unlink()
        raise

    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info("download_complete: %s (%.1f MB)", filepath, size_mb)
    return filepath


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

    logger.info(
        "loaded: %d cells x %d genes",
        adata.n_obs,
        adata.n_vars,
    )
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
    """Download (if needed) and load the configured dataset.

    This is the main entry point. It checks whether the file
    exists locally, downloads if missing, loads, and returns.

    Args:
        filepath: Optional explicit path. If provided, skips download
            and loads directly. Supports .h5 and .h5ad formats.

    Returns:
        AnnData with raw counts.
    """
    if filepath is not None:
        filepath = Path(filepath)
    else:
        filepath = download_dataset()

    suffix = filepath.suffix.lower()
    if suffix == ".h5":
        return load_10x_h5(filepath)
    elif suffix == ".h5ad":
        return load_h5ad(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Expected .h5 or .h5ad"
        )
