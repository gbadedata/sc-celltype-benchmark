# sc-celltype-benchmark

**Single-cell RNA-seq cell-type annotation pipeline with scientific benchmark evaluation framework**

[![CI](https://github.com/gbadedata/sc-celltype-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/gbadedata/sc-celltype-benchmark/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scanpy 1.12](https://img.shields.io/badge/scanpy-1.12-green.svg)](https://scanpy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Most single-cell tutorials stop at producing a UMAP. They cluster cells, assign labels by eye, and call it done. They do not ask the harder question: **how accurately did the annotation pipeline recover the true cell identities, and where does it fail?**

This project answers that question. It is a production-grade scanpy pipeline that annotates immune cell types from 10x Genomics scRNA-seq data using two independent strategies, then evaluates both strategies through a structured benchmark framework: oracle ground-truth labels from a pre-trained reference model, stratified holdout experiments, per-cell-type F1 scoring across calibrated difficulty tiers, and biological constraint validators that check necessary conditions independently of the oracle.

The benchmark framework is what distinguishes this project from a tutorial. It was designed to answer the kind of question a clinical bioinformatics team or a scientific software evaluation role actually cares about: given a new annotation method, how do you measure whether it works, where it works, and where it does not?

---

## Key results

| Metric | Value |
|---|---|
| Dataset | PBMC 3k (10x Genomics, 2,700 cells) |
| Cells passing QC | 2,699 |
| Genes passing QC | 13,714 |
| Highly variable genes | 2,000 |
| Leiden clusters | 8 (resolution 0.8) |
| Cell types annotated | 8 |
| Test cells evaluated | 540 (20% stratified holdout) |
| Benchmark accuracy | 0.9241 |
| Benchmark weighted F1 | 0.9151 |
| Cohen kappa | 0.8948 |
| Easy tier F1 | 0.991 (499 cells) |
| Medium tier F1 | 0.000 (38 cells, documented limitation) |
| Hard tier F1 | 1.000 (3 cells) |
| Cluster purity validator | 1.000 PASS |
| Marker recovery validator | 1.000 PASS |
| Biological consistency validator | 1.000 PASS |
| Pipeline runtime | 39.2 seconds |
| Tests passing | 168 |
| Test coverage | 58% |

---

## Per-cell-type benchmark results

| Cell type | Difficulty | F1 | Support |
|---|---|---|---|
| B cells | Easy | 1.000 | 69 |
| CD4+ T cells | Easy | 1.000 | 240 |
| Platelets | Hard | 1.000 | 3 |
| NK cells | Easy | 0.983 | 88 |
| CD14+ Monocytes | Easy | 0.966 | 99 |
| CD8+ T cells | Easy | 0.000 | 3 |
| Dendritic cells | Medium | 0.000 | 7 |
| FCGR3A+ Monocytes | Medium | 0.000 | 31 |

CD8+ T cells appear as easy-tier but score F1=0.000 with 3 test cells. Three cells is not a statistically meaningful test set; this reflects the small size of the CD8+ population in PBMC 3k, not a pipeline failure. The benchmark correctly surfaces this by reporting support alongside each score.

---

## Evidence

| Figure | Description |
|---|---|
| `evidence/screenshots/04_annotation_comparison.png` | Side-by-side UMAP: manual scoring vs CellTypist reference |
| `evidence/screenshots/05_marker_dotplot.png` | Canonical marker expression across annotated cell types |
| `evidence/screenshots/06_confusion_matrix.png` | Confusion matrix: 540 withheld test cells |
| `evidence/screenshots/07_benchmark_tier_summary.png` | F1 by difficulty tier |
| `evidence/screenshots/08_per_type_f1.png` | Per-cell-type F1 with support counts |

---

## Quick start

```bash
git clone git@github.com:gbadedata/sc-celltype-benchmark.git
cd sc-celltype-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.pipeline
```

The pipeline downloads the PBMC 3k dataset automatically on first run (~20 MB via scanpy/figshare), runs all eight phases, generates nine figures in `evidence/figures/`, writes a JSON benchmark report to `evidence/reports/benchmark_report.json`, and prints a structured summary to stdout. No external accounts or API keys required.

---

## Dataset

**10x Genomics PBMC 3k** -- 2,700 peripheral blood mononuclear cells from a healthy human donor, profiled with 10x Chromium v2 chemistry and sequenced on Illumina NextSeq 500. Sourced from the 10x Genomics public dataset library and distributed via figshare.

PBMCs are the standard benchmark dataset for single-cell methods because the cell types are well-characterised, canonical marker genes are established in literature, and published reference annotations exist from multiple independent sources, enabling oracle-based evaluation. The dataset is small enough to iterate quickly (39-second end-to-end runtime) while large enough to contain all major immune lineages.

---

## Architecture

Data flows through eight sequential phases. Each phase produces a clean output handed to the next. No phase is responsible for another's work.

```
Raw 10x counts
2,700 cells x 32,738 genes
          |
          v
[Phase 1]  Data Loader
           scanpy.datasets.pbmc3k() --> cached h5ad
           Supports .h5 and .h5ad input formats
          |
          v
[Phase 2]  Quality Control
           Mito% flagging, gene count thresholds, gene frequency filter
           2,700 cells x 32,738 genes  -->  2,699 cells x 13,714 genes
           QCReport: per-category removal counts, median statistics
          |
          v
[Phase 3]  Preprocessing
           1. Normalise: library-size normalise to 10,000 counts per cell
           2. Log1p transform
           3. Save layers['log_norm'] before scaling        <-- critical
           4. Select 2,000 highly variable genes (seurat flavour)
           5. Scale: zero mean, unit variance, clip at 10 SD
           6. PCA: 50 components, arpack solver, mask_var='highly_variable'
           PreprocessingReport: HVG count, variance explained
          |
          v
[Phase 4]  Clustering
           kNN graph: k=15 on X_pca
           Leiden: resolution=0.8, igraph flavour, n_iterations=2
           UMAP: 2D embedding for visualisation
           ClusteringReport: cluster sizes, resolution
          |
          v
[Phase 5]  Marker Detection
           Wilcoxon rank-sum (one-vs-rest), use_raw=True
           50 genes per cluster, pts=True (% cells expressing)
           Jaccard-based canonical marker matching for annotation guidance
          |
          v
[Phase 6]  Annotation
           Strategy A (Manual):
             Per-cluster mean log-norm expression of canonical marker sets
             Highest-scoring canonical type assigned to each cluster
             Operates on layers['log_norm'] (not scaled .X)
           Strategy B (CellTypist):
             Immune_All_Low.pkl (Dominguez Conde et al. 2022, Science)
             Per-cell prediction, majority voting per cluster
             53-label exhaustive harmonisation map to coarse PBMC categories
          |
          v
[Phase 7]  Benchmark Framework
           Oracle: CellTypist coarse labels as ground truth
           Holdout: stratified 20% split (seed=42), reproducible
           Evaluator: accuracy, weighted F1, macro F1, Cohen kappa
           Difficulty tiers: easy / medium / hard (documented rationale)
           Validators: cluster purity, marker recovery, biological consistency
           JSON report: all metrics, confusion matrix, tier breakdown
          |
          v
[Phase 8]  Visualisation + Report
           9 PNG figures --> evidence/figures/
           benchmark_report.json --> evidence/reports/
           pbmc3k_annotated.h5ad --> data/
```

---

## Annotation strategies

### Manual marker scoring

For each Leiden cluster, the pipeline computes the mean log-normalised expression of every canonical cell type's marker gene set across all cells in that cluster. The cell type whose markers have the highest mean expression score is assigned to the cluster. This mirrors what a computational biologist does manually when reading a dot plot, but executes it algorithmically and reproducibly.

Canonical markers are curated from Zheng et al. 2017 (Nature Communications) and the Human Cell Atlas PBMC reference. Ten cell types are covered: CD4+ T, CD8+ T, NK, B, CD14+ Monocytes, FCGR3A+ Monocytes, Dendritic cells, Plasmacytoid DCs, Platelets, and HSPCs.

### CellTypist reference annotation

The pipeline downloads the `Immune_All_Low.pkl` pre-trained model (Dominguez Conde et al. 2022, Science, 53 labels) and runs per-cell prediction with Leiden-based majority voting. The fine-grained CellTypist labels are mapped to the same coarse categories used by manual annotation through an exhaustive 53-label harmonisation table, enabling direct comparison.

CellTypist output serves as the oracle ground truth for benchmark evaluation. It is the more accurate method -- particularly for transcriptionally similar subtypes -- which is exactly what makes it the right reference for scoring the manual baseline.

---

## Benchmark framework design

### Oracle and holdout

The oracle holds the CellTypist majority-voting labels for all 2,699 cells. Before any benchmark evaluation, it generates a stratified 20% holdout (540 cells) using scikit-learn's `StratifiedShuffleSplit`, ensuring every cell type is represented proportionally in both the visible training set and the withheld test set. The random seed is fixed at 42.

The pipeline runs annotation blind on the full dataset without access to the oracle labels. Only after predictions are generated does the evaluator receive the oracle answers for scoring.

### Difficulty tiers

Cell types are assigned to three difficulty tiers based on annotation challenge, with rationale documented per assignment:

**Easy** -- abundant populations with distinctive, high-expression markers and large clusters: CD4+ T cells, CD8+ T cells, B cells, CD14+ Monocytes, NK cells. These are the types that any reasonable annotation method should get right.

**Medium** -- smaller populations with markers that overlap the transcriptional background of neighbouring types: FCGR3A+ Monocytes (shares monocyte programme with CD14+), Dendritic cells (shares HLA-DR with monocytes). These are where annotation methods are separated from each other.

**Hard** -- rare populations with weak or ambiguous markers: Plasmacytoid DCs, Platelets, HSPCs. These are the populations that even reference models sometimes miss.

Reporting performance separately per tier allows honest communication about where a method works and where it does not, rather than averaging across easy and hard cases into a misleadingly high overall score.

### Biological constraint validators

Three validators run independently of the oracle. They check necessary biological conditions that must hold for any correctly annotated PBMC dataset, and can be verified from the expression data alone:

**Cluster purity** -- each Leiden cluster should be dominated by a single cell type (threshold: 80% of cells). Mixed clusters indicate over-clustering, biological heterogeneity, or annotation errors.

**Marker recovery** -- each annotated cell type should express at least one of its canonical marker genes above the dataset-wide mean expression for that gene. A cell type that does not express any of its markers has been misannotated.

**Biological consistency** -- four cross-lineage expression constraints that must hold regardless of annotation method: CD3D mean expression in T cells must exceed B cells; CD79A mean expression in B cells must exceed T cells; CD14 mean expression in CD14+ Monocytes must exceed B cells; NKG7 mean expression in NK cells must exceed CD4+ T cells.

All three validators pass at 1.000 on this dataset.

---

## Known limitations

### Medium-tier annotation failure

FCGR3A+ Monocytes and Dendritic cells score F1=0.000 under manual marker scoring. This is not a bug. It is a documented limitation of the mean-expression scoring approach for transcriptionally similar subtypes.

FCGR3A+ Monocytes share the core monocyte transcriptional programme (LYZ, S100A8, S100A9) with CD14+ Monocytes. Their distinguishing markers (FCGR3A, MS4A7, IFITM3) are expressed at lower absolute levels. When the manual scorer computes mean expression across a cluster, the shared high-abundance monocyte markers dominate over the subtype-specific ones, and the cluster is assigned to CD14+ Monocytes instead.

The same logic applies to Dendritic cells, whose HLA-DR expression is shared with monocytes and dominates in the mean-expression comparison.

This is a known limitation of unsupervised marker-scoring annotation. Reference-based methods like CellTypist handle this correctly because they were trained on datasets with labelled examples of each subtype. The benchmark framework surfaces this failure precisely and correctly -- which is the point of having a benchmark.

### Small hard-tier test set

The hard tier contains only 3 cells in the test set (Platelets). A perfect F1=1.000 on 3 cells is not a statistically meaningful result. The benchmark reports support alongside every score exactly for this reason.

---

## Engineering decisions and challenges

### Challenge 1: 10x CDN blocks programmatic downloads

The initial implementation used `urllib.request.urlretrieve` to download the PBMC 10k dataset directly from `cf.10xgenomics.com`. The CDN returned `HTTP Error 403`. The CDN uses browser fingerprinting and rejects non-browser user agents.

**Fix:** Switched to `scanpy.datasets.pbmc3k()`, which fetches from figshare under a permissive access policy. Also switched from PBMC 10k to PBMC 3k, which has better-documented reference annotations for oracle generation.

### Challenge 2: `adata.raw = adata` does not snapshot correctly

The initial normalisation step assigned `adata.raw = adata` before calling `sc.pp.normalize_total`. When checked afterwards, `adata.raw.X` did not match the pre-normalisation counts -- it differed by thousands due to floating-point mutation during the in-place normalisation.

**Root cause:** `adata.raw = adata` creates a `Raw` object from the AnnData at assignment time, but the underlying matrix is passed by reference. In-place operations on `adata.X` can propagate into the snapshot.

**Fix:** Changed to `adata.raw = adata.copy()`. Explicit copy guarantees a clean pre-normalisation snapshot.

### Challenge 3: CellTypist requires log1p-normalised input, not raw or scaled data

CellTypist raises `ValueError: Invalid expression matrix in .X, expect log1p normalized expression to 10000 counts per cell`. This occurred even after building a query AnnData from `adata.raw.X`.

**Root cause:** `adata.raw` holds a copy taken before `normalize_total` -- raw integer counts summing to approximately 2,500 per cell. CellTypist checks that `expm1(X).sum(axis=1)` is approximately 10,000. Raw counts fail this check. After `sc.pp.scale()`, `adata.X` holds zero-mean unit-variance values, which also fail. There was no saved copy of the log1p-normalised intermediate.

**Fix:** Added `adata.layers['log_norm'] = adata.X.copy()` immediately after `sc.pp.log1p()` and before `sc.pp.scale()`. CellTypist queries are now built from this layer, which holds exactly the log1p-normalised matrix CellTypist expects.

### Challenge 4: Incomplete CellTypist label harmonisation caused kappa=0.27

The initial benchmark reported Cohen kappa=0.27 instead of the expected ~0.89. Investigation showed that 5 of the 8 CellTypist labels returned on PBMC 3k were absent from the harmonisation map: `Tcm/Naive helper T cells` (240 test cells), `CD16+ NK cells` (88 cells), `Tem/Trm cytotoxic T cells` (3 cells), `DC` (7 cells), and `Megakaryocytes/platelets` (3 cells).

These 341 cells passed through harmonisation unmapped, so the oracle held CellTypist's fine-grained strings while manual annotation returned coarse PBMC types. No cell in those 341 could ever match the oracle, producing F1=0 for most cell types regardless of annotation quality.

**Fix:** Built an exhaustive 53-label mapping covering every label in the `Immune_All_Low.pkl` model, with rationale documented per entry. Added a regression test that explicitly lists all 53 labels and asserts every one is present in the map.

### Challenge 5: HVG flavour selection

`seurat_v3` flavour for highly variable gene selection requires the `skmisc` package (`loess` regression) and expects raw counts as input. It raises `ModuleNotFoundError` without `skmisc`, and produces incorrect results when applied to log-normalised data.

**Fix:** Switched to `seurat` flavour, which operates correctly on log-normalised data using a mean/dispersion approach and has no external dependencies.

### Challenge 6: Deprecated PCA API

`sc.tl.pca(..., use_highly_variable=True)` raises a `FutureWarning` in scanpy >= 1.10. The parameter was deprecated in favour of `mask_var`.

**Fix:** Changed to `mask_var="highly_variable"`. The behaviour is identical; the API is current.

### Challenge 7: Sparse matrix densification warning

`sc.pp.scale()` emits `UserWarning: zero-centering a sparse array/matrix densifies it` when called on sparse input. This is expected behaviour -- scaling requires dense representation -- but the warning is noisy.

**Fix:** Explicitly converted to dense before calling scale: `if sp.issparse(adata.X): adata.X = adata.X.toarray()`. The densification is now intentional and documented, not a silent side-effect.

### Challenge 8: Wilcoxon DE warning fires incorrectly

`sc.tl.rank_genes_groups` emits a warning about raw count data even when `use_raw=True` is correctly set. This is a known scanpy bug (github.com/scverse/scanpy/issues/2239) where the internal check inspects the wrong object.

**Fix:** Suppressed with `warnings.filterwarnings` before the call and `warnings.resetwarnings()` immediately after, with the GitHub issue reference in the code comment.

---

## Project structure

```
sc-celltype-benchmark/
|
+-- config/
|   +-- settings.py              Pydantic Settings: all parameters, SC_ env prefix
|   +-- __init__.py
|
+-- src/
|   +-- pipeline.py              End-to-end runner: python3 -m src.pipeline
|   +-- data_loader.py           Download PBMC 3k, load h5/h5ad, local caching
|   +-- quality_control.py       QC metrics, cell filtering, gene filtering, QCReport
|   +-- preprocessing.py         Normalise, log1p, layers['log_norm'], HVG, scale, PCA
|   +-- clustering.py            kNN graph, Leiden, UMAP, ClusteringReport
|   +-- markers.py               Wilcoxon DE, PBMC_MARKERS, Jaccard matching
|   +-- annotation.py            Manual scoring, CellTypist, 53-label harmonisation
|   +-- visualization.py         9 publication-quality PNG figures
|   +-- benchmark/
|       +-- oracle.py            OracleGenerator: ground truth, stratified holdout
|       +-- evaluator.py         BenchmarkEvaluator: F1, kappa, confusion matrix, JSON
|       +-- validators.py        Cluster purity, marker recovery, biological consistency
|       +-- difficulty.py        DifficultyTier enum, PBMC tier assignments
|
+-- tests/
|   +-- conftest.py              Synthetic 200-cell fixtures, no network required
|   +-- test_fixtures.py         12 fixture validation tests
|   +-- test_data_loader.py      8 tests: loading, caching, format detection
|   +-- test_quality_control.py  14 tests: metrics, filtering, QCReport
|   +-- test_preprocessing.py    18 tests: normalise, HVG, scale, PCA, layers
|   +-- test_clustering.py       16 tests: graph, Leiden dtype, UMAP shape
|   +-- test_markers.py          21 tests: DE, use_raw, Jaccard matching
|   +-- test_annotation.py       35 tests: manual, harmonisation, comparison
|   +-- test_benchmark_oracle.py 13 tests: holdout, stratification, reproducibility
|   +-- test_benchmark_evaluator.py  15 tests: scoring, tier breakdown, report
|   +-- test_benchmark_validators.py 16 tests: purity, recovery, consistency
|
+-- evidence/
|   +-- figures/                 9 PNG figures (generated by pipeline)
|   +-- reports/                 benchmark_report.json (generated by pipeline)
|   +-- screenshots/             Key figures for portfolio evidence
|
+-- data/                        Downloaded datasets and processed h5ad (gitignored)
+-- docs/                        Additional documentation
+-- .github/workflows/ci.yml     GitHub Actions: ruff lint + pytest on every push
+-- requirements.txt
+-- pyproject.toml               ruff, pytest, coverage configuration
+-- .env.example                 All configurable parameters documented
```

---

## Configuration

All pipeline parameters are configurable via environment variables prefixed `SC_`. No source code changes are needed to adjust thresholds or experiment with different settings.

```bash
# Run with higher Leiden resolution (more clusters, finer cell types)
SC_LEIDEN_RESOLUTION=1.5 python3 -m src.pipeline

# Use a larger holdout for benchmark evaluation
SC_BENCHMARK_HOLDOUT_FRACTION=0.3 python3 -m src.pipeline

# Load a custom dataset instead of PBMC 3k
python3 -m src.pipeline --data /path/to/your_data.h5ad

# Adjust QC thresholds
SC_MIN_GENES_PER_CELL=300 SC_MAX_PCT_MITO=15.0 python3 -m src.pipeline
```

See `.env.example` for the complete parameter list with defaults.

---

## Running tests

```bash
python3 -m pytest tests/ -v
```

Tests run entirely on synthetic data (200-cell fixtures, seeded RNG) and require no network access or real dataset downloads. The synthetic fixtures are biologically grounded: canonical marker genes are embedded at known indices with Poisson-boosted expression, so tests that check marker enrichment and annotation logic are testing real biological expectations, not arbitrary pass/fail conditions.

Coverage gaps in `annotation.py`, `pipeline.py`, and `visualization.py` cover network-dependent and integration-level paths (CellTypist model download, end-to-end pipeline run, figure rendering) that are validated by manual integration testing against the real dataset rather than unit tests.

---

## Reproducing the results

The pipeline is fully reproducible. All random operations use `random_seed=42`. The dataset download is deterministic (same file from figshare). The CellTypist model is versioned and cached locally after first download.

```bash
# From a fresh clone
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.pipeline
# Expected output: accuracy=0.9241, weighted_f1=0.9151, kappa=0.8948
```

---

## Stack

| Category | Tools |
|---|---|
| Single-cell analysis | scanpy 1.12, anndata 0.12 |
| Reference annotation | CellTypist 1.6 (Immune_All_Low.pkl) |
| Statistical evaluation | scikit-learn, scipy |
| Scientific computing | numpy, pandas |
| Visualisation | matplotlib 3.10, seaborn |
| Configuration | Pydantic Settings 2.0 |
| Logging | structlog |
| Testing | pytest, pytest-cov |
| Linting | ruff |
| CI | GitHub Actions |
| Python | 3.12 |

---

## References

Zheng et al. (2017). Massively parallel digital transcriptional profiling of single cells. *Nature Communications*, 8, 14049.

Wolf et al. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*, 19, 15.

Dominguez Conde et al. (2022). Cross-tissue immune cell analysis reveals tissue-specific features in humans. *Science*, 376(6594).

Traag et al. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9, 5233.
