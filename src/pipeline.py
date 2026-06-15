"""End-to-end pipeline runner.

Orchestrates all eight phases in sequence and writes a structured
benchmark report to evidence/reports/benchmark_report.json.

Usage:
    python3 -m src.pipeline
    python3 -m src.pipeline --data path/to/custom.h5ad

The pipeline is idempotent: re-running overwrites the previous report
and figures. All intermediate AnnData objects are saved to data/ so
individual phases can be inspected independently.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import structlog

from config.settings import settings

# ── Structured logging ────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


def run_pipeline(data_path: str | None = None) -> dict:
    """Execute the full pipeline and return the benchmark report.

    Args:
        data_path: Optional path to a local h5ad or h5 file.
            If None, downloads PBMC 3k via scanpy.

    Returns:
        Benchmark report dict (also written to evidence/reports/).
    """
    start = datetime.now(timezone.utc)
    log.info("pipeline_started", dataset=settings.dataset_name, timestamp=start.isoformat())

    # ── Phase 1: Load ─────────────────────────────────────────────────
    from src.data_loader import get_dataset
    adata = get_dataset(filepath=data_path)
    log.info("phase1_complete", n_cells=adata.n_obs, n_genes=adata.n_vars)

    # ── Phase 2: QC ───────────────────────────────────────────────────
    from src.quality_control import run_qc
    adata, qc_report = run_qc(adata)
    log.info(
        "phase2_complete",
        cells_after=qc_report.cells_after,
        genes_after=qc_report.genes_after,
        cells_removed=qc_report.cells_removed,
    )

    # ── Phase 3: Preprocessing ────────────────────────────────────────
    from src.preprocessing import run_preprocessing
    adata, pp_report = run_preprocessing(adata)
    log.info(
        "phase3_complete",
        n_hvg=pp_report.n_hvg,
        n_pcs=pp_report.n_pcs,
        pct_variance=pp_report.pct_variance_explained,
    )

    # ── Phase 4: Clustering ───────────────────────────────────────────
    from src.clustering import run_clustering
    adata, cl_report = run_clustering(adata)
    log.info(
        "phase4_complete",
        n_clusters=cl_report.n_clusters,
        resolution=cl_report.leiden_resolution,
    )

    # ── Phase 5: Marker detection ─────────────────────────────────────
    from src.markers import detect_markers, get_top_markers
    adata = detect_markers(adata)
    top_markers = get_top_markers(adata, n_genes=5)
    log.info("phase5_complete", n_marker_rows=len(top_markers))

    # ── Phase 6: Annotation ───────────────────────────────────────────
    from src.annotation import annotate_celltypist, annotate_manual, harmonise_labels
    adata = annotate_manual(adata)
    adata = annotate_celltypist(adata)
    adata = harmonise_labels(adata)
    n_manual_types = adata.obs["celltype_manual"].nunique()
    n_reference_types = adata.obs["celltype_reference_coarse"].nunique()
    log.info(
        "phase6_complete",
        manual_types=n_manual_types,
        reference_types=n_reference_types,
    )

    # ── Phase 7: Benchmark ────────────────────────────────────────────
    from src.benchmark.difficulty import assign_difficulty_tiers
    from src.benchmark.evaluator import BenchmarkEvaluator
    from src.benchmark.oracle import OracleGenerator
    from src.benchmark.validators import run_all_validators

    oracle = OracleGenerator(seed=settings.benchmark_seed)
    oracle.fit(adata, label_column="celltype_reference_coarse")
    oracle.generate_holdout(fraction=settings.benchmark_holdout_fraction)

    answers = oracle.get_answers()
    predictions = adata.obs.loc[answers.index, "celltype_manual"]

    evaluator = BenchmarkEvaluator()
    result = evaluator.score(predictions, answers)

    tiers = assign_difficulty_tiers(answers)
    tier_results = evaluator.score_by_tier(predictions, answers, tiers)
    result.tier_results = tier_results

    validator_results = run_all_validators(adata)

    log.info(
        "phase7_complete",
        accuracy=round(result.accuracy, 4),
        weighted_f1=round(result.weighted_f1, 4),
        cohens_kappa=round(result.cohens_kappa, 4),
        n_cells_evaluated=result.n_cells_evaluated,
        validators_passed=sum(v.passed for v in validator_results),
    )

    # ── Phase 8: Visualisation ────────────────────────────────────────
    from src.visualization import generate_all_figures
    generate_all_figures(adata, result, validator_results)
    log.info("phase8_complete", figures_dir=str(settings.evidence_dir / "figures"))

    # ── Build and save report ─────────────────────────────────────────
    report = evaluator.generate_report(result, dataset_name=settings.dataset_name)

    # Add pipeline metadata
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    report["pipeline"] = {
        "runtime_seconds": round(elapsed, 1),
        "settings": {
            "leiden_resolution": settings.leiden_resolution,
            "n_top_genes": settings.n_top_genes,
            "n_pcs": settings.n_pcs,
            "n_neighbors": settings.n_neighbors,
            "benchmark_holdout_fraction": settings.benchmark_holdout_fraction,
            "benchmark_seed": settings.benchmark_seed,
        },
        "qc": {
            "cells_before": qc_report.cells_before,
            "cells_after": qc_report.cells_after,
            "genes_before": qc_report.genes_before,
            "genes_after": qc_report.genes_after,
            "median_genes_per_cell": qc_report.median_genes_per_cell,
            "median_pct_mito": qc_report.median_pct_mito,
        },
        "clustering": {
            "n_clusters": cl_report.n_clusters,
            "resolution": cl_report.leiden_resolution,
        },
        "validators": [
            {
                "name": v.name,
                "passed": v.passed,
                "score": v.score,
                "details": v.details,
            }
            for v in validator_results
        ],
    }

    report_dir = settings.evidence_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(
        "pipeline_complete",
        runtime_seconds=round(elapsed, 1),
        report=str(report_path),
    )

    # Save processed AnnData
    output_path = settings.data_dir / "pbmc3k_annotated.h5ad"
    adata.write_h5ad(output_path)
    log.info("adata_saved", path=str(output_path))

    return report


def _print_summary(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    s = report["summary"]
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Dataset:       {report['metadata']['dataset']}")
    print(f"  Cells eval:    {s['n_cells_evaluated']}")
    print(f"  Accuracy:      {s['accuracy']:.4f}")
    print(f"  Weighted F1:   {s['weighted_f1']:.4f}")
    print(f"  Macro F1:      {s['macro_f1']:.4f}")
    print(f"  Cohen kappa:   {s['cohens_kappa']:.4f}")
    if "tier_breakdown" in report:
        print("\nBy difficulty tier:")
        for tier, tr in sorted(report["tier_breakdown"].items()):
            print(f"  {tier:8s}: F1={tr['weighted_f1']:.4f}  n={tr['n_cells']}")
    if "pipeline" in report and "validators" in report["pipeline"]:
        print("\nBiological validators:")
        for v in report["pipeline"]["validators"]:
            status = "PASS" if v["passed"] else "FAIL"
            print(f"  [{status}] {v['name']}: {v['score']:.4f}")
    print(f"\nRuntime: {report['pipeline']['runtime_seconds']}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="sc-celltype-benchmark: single-cell annotation pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to local h5ad or h5 file. Defaults to PBMC 3k download.",
    )
    args = parser.parse_args()

    try:
        report = run_pipeline(data_path=args.data)
        _print_summary(report)
        sys.exit(0)
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        sys.exit(1)
