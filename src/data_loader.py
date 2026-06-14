"""Data acquisition and loading.

Downloads 10x Genomics scRNA-seq datasets and loads them into AnnData objects.
Supports h5, h5ad, and 10x MTX formats.
"""

from __future__ import annotations

import logging

import anndata as ad

logger = logging.getLogger(__name__)


def download_dataset(url: str | None = None, dest_dir: str | None = None) -> str:
    """Download dataset from URL to local directory.

    Returns the path to the downloaded file.
    """
    raise NotImplementedError("Phase 1")


def load_10x_h5(filepath: str) -> ad.AnnData:
    """Load a 10x Genomics h5 file into AnnData."""
    raise NotImplementedError("Phase 1")


def load_h5ad(filepath: str) -> ad.AnnData:
    """Load an h5ad file into AnnData."""
    raise NotImplementedError("Phase 1")


def get_dataset() -> ad.AnnData:
    """Download (if needed) and load the configured dataset.

    This is the main entry point. It checks whether the file
    exists locally, downloads if missing, loads, and returns.
    """
    raise NotImplementedError("Phase 1")
