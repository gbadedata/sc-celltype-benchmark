"""Main pipeline entry point.

Orchestrates the full single-cell annotation and benchmark pipeline:
1. Load data
2. Quality control
3. Preprocessing
4. Clustering
5. Marker detection
6. Cell-type annotation (manual + reference)
7. Benchmark evaluation
8. Visualisation and reporting
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog

from config.settings import settings

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


def run_pipeline() -> None:
    """Execute the full pipeline end to end."""
    logger.info(
        "pipeline_started",
        dataset=settings.dataset_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Phase 1: Load data
    from src.data_loader import get_dataset

    adata = get_dataset()
    logger.info("data_loaded", n_cells=adata.n_obs, n_genes=adata.n_vars)

    # Phase 2: QC
    from src.quality_control import run_qc

    adata = run_qc(adata)
    logger.info("qc_complete", n_cells=adata.n_obs, n_genes=adata.n_vars)

    # Phase 3: Preprocessing
    from src.preprocessing import run_preprocessing

    adata = run_preprocessing(adata)
    logger.info("preprocessing_complete", n_hvg=int(adata.var["highly_variable"].sum()))

    # Phase 4: Clustering
    from src.clustering import run_clustering

    adata = run_clustering(adata)
    n_clusters = adata.obs["leiden"].nunique()
    logger.info("clustering_complete", n_clusters=n_clusters)

    # Phase 5: Marker detection
    from src.markers import detect_markers, get_top_markers

    adata = detect_markers(adata)
    markers_df = get_top_markers(adata)
    logger.info("markers_detected", n_markers=len(markers_df))

    # Phase 6: Annotation
    from src.annotation import run_annotation

    adata = run_annotation(adata)
    logger.info(
        "annotation_complete",
        manual_types=adata.obs["celltype_manual"].nunique(),
        reference_types=adata.obs["celltype_reference"].nunique(),
    )

    # Phase 7: Benchmark
    from src.benchmark.evaluator import BenchmarkEvaluator
    from src.benchmark.oracle import OracleGenerator

    oracle = OracleGenerator(seed=settings.random_seed)
    oracle.fit(adata, label_column="celltype_reference_coarse")
    oracle.generate_holdout(fraction=settings.benchmark_holdout_fraction)

    evaluator = BenchmarkEvaluator()
    answers = oracle.get_answers()
    predictions = adata.obs.loc[answers.index, "celltype_manual"]
    result = evaluator.score(predictions, answers)

    report = evaluator.generate_report(result)
    report_path = settings.evidence_dir / "reports" / "benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(
        "benchmark_complete",
        accuracy=round(result.accuracy, 4),
        weighted_f1=round(result.weighted_f1, 4),
        cohens_kappa=round(result.cohens_kappa, 4),
    )

    # Phase 8: Visualisation
    from src.visualization import (
        plot_annotation_comparison,
        plot_benchmark_summary,
        plot_marker_dotplot,
        plot_qc_metrics,
        plot_umap_celltypes,
        plot_umap_clusters,
    )

    plot_qc_metrics(adata)
    plot_umap_clusters(adata)
    plot_umap_celltypes(adata)
    plot_marker_dotplot(adata)
    plot_annotation_comparison(adata)
    if result.tier_results:
        plot_benchmark_summary(result.tier_results)

    logger.info("visualisation_complete", figures_dir=str(FIGURE_DIR))

    # Save processed data
    output_path = settings.data_dir / "processed_annotated.h5ad"
    adata.write_h5ad(output_path)
    logger.info("pipeline_complete", output=str(output_path))


FIGURE_DIR = settings.evidence_dir / "figures"

if __name__ == "__main__":
    run_pipeline()
