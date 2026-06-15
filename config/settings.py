"""Pipeline configuration.

All parameters are centralised here. Override via environment variables
or a .env file (see .env.example).
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class PipelineSettings(BaseSettings):
    """Single-cell annotation pipeline settings."""

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    evidence_dir: Path = Path(__file__).resolve().parent.parent / "evidence"

    # ── Data source ────────────────────────────────────────────────────
    # Default: PBMC 3k from 10x Genomics (downloaded via scanpy from figshare)
    dataset_name: str = "PBMC 3k (10x Genomics)"
    dataset_filename: str = "pbmc3k_raw.h5ad"

    # ── QC thresholds ──────────────────────────────────────────────────
    min_genes_per_cell: int = 200
    max_genes_per_cell: int = 5000
    max_pct_mito: float = 20.0
    min_cells_per_gene: int = 3

    # ── Preprocessing ──────────────────────────────────────────────────
    n_top_genes: int = 2000
    n_pcs: int = 50
    n_neighbors: int = 15
    leiden_resolution: float = 0.8  # balances cluster granularity vs annotation accuracy

    # ── Annotation ─────────────────────────────────────────────────────
    celltypist_model: str = "Immune_All_Low.pkl"

    # ── Benchmark ──────────────────────────────────────────────────────
    benchmark_holdout_fraction: float = 0.2
    benchmark_seed: int = 42
    benchmark_difficulty_tiers: list[str] = [
        "easy",     # Major lineages: T, B, Monocyte, NK
        "medium",   # Subtypes: CD4/CD8, classical/non-classical mono
        "hard",     # Rare types: pDC, HSPC, platelets
    ]

    # ── Reproducibility ────────────────────────────────────────────────
    random_seed: int = 42

    model_config = {"env_prefix": "SC_", "env_file": ".env", "extra": "ignore"}


settings = PipelineSettings()
