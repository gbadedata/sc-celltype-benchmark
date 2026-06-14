# Single-Cell RNA-seq Cell-Type Annotation Benchmark

A production scanpy pipeline for single-cell RNA-seq analysis with automated cell-type annotation and a benchmark evaluation framework that quantifies annotation accuracy against ground-truth labels across calibrated difficulty tiers.

## What this project does

Given a 10x Genomics scRNA-seq dataset, this pipeline:

1. Applies quality control filtering (mitochondrial %, gene counts, UMI counts)
2. Preprocesses data (normalisation, HVG selection, PCA)
3. Clusters cells (neighbourhood graph, Leiden community detection, UMAP)
4. Detects marker genes per cluster (Wilcoxon rank-sum test)
5. Annotates cell types using two independent methods:
   - **Manual**: canonical marker gene scoring against curated PBMC marker sets
   - **Reference-based**: CellTypist pre-trained immune cell model
6. Evaluates annotation quality through a benchmark framework with:
   - Oracle ground-truth generation from reference annotations
   - Stratified holdout experiments (20% withheld labels)
   - Per-cell-type precision, recall, and F1 scoring
   - Three difficulty tiers (easy/medium/hard) based on cell-type annotation challenge
   - Biological constraint validators (marker expression consistency checks)

## Why a benchmark framework matters

Most single-cell tutorials stop at "here are the clusters, here are the markers." They do not ask: **how do you know the annotation is correct?**

This project answers that question with a structured evaluation framework. The oracle holds ground-truth labels. The pipeline runs blind on withheld cells. Validator functions enforce biological constraints that metrics alone cannot capture (e.g., T cells must express CD3D, B cells must express CD79A). Difficulty tiers separate easy cases (well-separated major lineages) from hard cases (rare cell types, transitional states) so you can see where your pipeline succeeds and where it fails.

This is evaluation design, not just analysis.

## Dataset

**10x Genomics PBMC 10k v3** -- approximately 11,000 peripheral blood mononuclear cells profiled with 10x Chromium v3 chemistry. PBMCs contain well-characterised immune cell populations (T cells, B cells, monocytes, NK cells, dendritic cells, platelets) with established canonical markers, making them ideal for benchmarking annotation accuracy.

## Architecture

```
Raw 10x h5
    |
    v
[Quality Control] -- filter cells/genes by QC metrics
    |
    v
[Preprocessing] -- normalise, HVG, scale, PCA
    |
    v
[Clustering] -- neighbours, Leiden, UMAP
    |
    v
[Marker Detection] -- Wilcoxon rank-sum per cluster
    |
    v
[Annotation] -- manual (marker scoring) + CellTypist (reference)
    |
    v
[Benchmark Framework]
    |-- Oracle: ground-truth labels from reference annotation
    |-- Holdout: stratified 80/20 split
    |-- Evaluator: accuracy, F1, Cohen's kappa per tier
    |-- Validators: biological consistency checks
    |
    v
[Visualisation + Report]
    |-- UMAP plots (clusters, cell types, comparison)
    |-- Marker dot plots and heatmaps
    |-- Confusion matrix
    |-- Benchmark summary by difficulty tier
```

## Project structure

```
sc-celltype-benchmark/
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic Settings configuration
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Main pipeline orchestrator
│   ├── data_loader.py           # Download and load 10x data
│   ├── quality_control.py       # QC metrics and filtering
│   ├── preprocessing.py         # Normalisation, HVG, PCA
│   ├── clustering.py            # Neighbours, Leiden, UMAP
│   ├── markers.py               # Marker detection + canonical marker sets
│   ├── annotation.py            # Manual + CellTypist annotation
│   ├── visualization.py         # Publication-quality figures
│   └── benchmark/
│       ├── __init__.py
│       ├── oracle.py            # Ground-truth label management
│       ├── evaluator.py         # Scoring and reporting
│       ├── validators.py        # Biological constraint checks
│       └── difficulty.py        # Difficulty tier assignments
├── tests/
│   ├── conftest.py              # Synthetic AnnData fixtures
│   └── test_fixtures.py         # Fixture validation tests
├── evidence/
│   ├── figures/                 # Generated plots
│   ├── reports/                 # Benchmark JSON reports
│   └── screenshots/             # Deployment evidence
├── data/                        # Downloaded datasets (gitignored)
├── docs/                        # Additional documentation
├── .github/workflows/ci.yml    # GitHub Actions CI
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Setup

```bash
cd ~/projects/bioinformatics
git clone git@github.com:gbadedata/sc-celltype-benchmark.git
cd sc-celltype-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
python3 -m pytest tests/ -v
```

## Run the full pipeline

```bash
python3 -m src.pipeline
```

## Key results

*Results will be populated as each phase is implemented.*

| Metric | Value |
|---|---|
| Cells after QC | -- |
| Clusters (Leiden) | -- |
| Cell types annotated | -- |
| Benchmark accuracy | -- |
| Benchmark weighted F1 | -- |
| Easy tier F1 | -- |
| Medium tier F1 | -- |
| Hard tier F1 | -- |
| Tests passing | -- |

## Tools and libraries

- **scanpy** -- single-cell RNA-seq analysis
- **anndata** -- annotated data matrices
- **CellTypist** -- reference-based cell-type annotation
- **scikit-learn** -- evaluation metrics
- **scipy** -- statistical tests
- **matplotlib / seaborn** -- visualisation
- **pytest** -- testing
- **ruff** -- linting
- **GitHub Actions** -- CI

## Author

**O.J. Odimayo** -- Bioinformatics Data Engineer
- GitHub: [gbadedata](https://github.com/gbadedata)
- Email: oluwagbadeodimayo@gmail.com
