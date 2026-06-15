# sc-celltype-benchmark

A production scanpy pipeline for single-cell RNA-seq cell-type annotation with a scientific benchmark evaluation framework. Annotates immune cell types from 10x Genomics PBMC data, then evaluates annotation accuracy against a reference oracle across calibrated difficulty tiers.

[![CI](https://github.com/gbadedata/sc-celltype-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/gbadedata/sc-celltype-benchmark/actions/workflows/ci.yml)

---

## What this project does

Most single-cell tutorials stop at producing a UMAP. This project asks the harder question: **how do you know the annotation is correct?**

It answers that question with a structured benchmark framework: oracle ground-truth labels from CellTypist, stratified holdout experiments, per-cell-type F1 scoring across difficulty tiers, and biological constraint validators that check necessary conditions independently of the oracle.

---

## Quick start

```bash
git clone git@github.com:gbadedata/sc-celltype-benchmark.git
cd sc-celltype-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.pipeline   # downloads PBMC 3k and runs all 8 phases
```

---

## Dataset

**10x Genomics PBMC 3k** — 2,700 peripheral blood mononuclear cells from a healthy donor profiled with 10x Chromium v2 chemistry. Downloaded automatically via `scanpy.datasets.pbmc3k()` (figshare).

---

## Architecture

```
Raw 10x counts (2,700 cells × 32,738 genes)
        │
        ▼
[Phase 1] Data loader        — download and cache PBMC 3k h5ad
        │
        ▼
[Phase 2] Quality control    — mito%, gene count filtering → 2,699 × 13,714
        │
        ▼
[Phase 3] Preprocessing      — normalise (10k), log1p, save layers['log_norm'],
        │                       2,000 HVGs, scale, PCA (50 components)
        ▼
[Phase 4] Clustering         — kNN graph (k=15), Leiden (res=1.2), UMAP
        │
        ▼
[Phase 5] Marker detection   — Wilcoxon rank-sum (use_raw=True), 50 genes/cluster
        │
        ▼
[Phase 6] Annotation         — Manual: canonical marker mean-expression scoring
        │                       CellTypist: Immune_All_Low.pkl, majority voting
        │                       Harmonise: 53-label exhaustive coarse mapping
        ▼
[Phase 7] Benchmark          — Oracle holdout (20% stratified), evaluator,
        │                       difficulty tiers, biological validators
        ▼
[Phase 8] Visualisation      — 8 publication-quality figures → evidence/figures/
```

---

## Key results

| Metric | Value |
|---|---|
| Cells after QC | 2,699 |
| Genes after QC | 13,714 |
| HVGs selected | 2,000 |
| Leiden clusters | 10 (resolution 1.2) |
| Cell types annotated | 8 |
| Benchmark accuracy | 0.9241 |
| Benchmark weighted F1 | 0.9151 |
| Cohen's kappa | 0.8948 |
| Easy tier F1 | 0.991 (n=499) |
| Medium tier F1 | 0.000 (n=38, see known limitations) |
| Hard tier F1 | 1.000 (n=3) |
| Cluster purity validator | 1.000 — PASS |
| Marker recovery validator | 1.000 — PASS |
| Biological consistency validator | 1.000 — PASS |
| Tests passing | 168 |
| Test coverage | 79% |

---

## Per-cell-type F1

| Cell type | F1 | Support |
|---|---|---|
| B cells | 1.000 | 69 |
| CD4+ T cells | 1.000 | 240 |
| Platelets | 1.000 | 3 |
| NK cells | 0.983 | 88 |
| CD14+ Monocytes | 0.966 | 99 |
| CD8+ T cells | 0.000 | 3 |
| Dendritic cells | 0.000 | 7 |
| FCGR3A+ Monocytes | 0.000 | 31 |

---

## Known limitations

**Medium-tier subtypes (FCGR3A+ Monocytes, Dendritic cells) score F1=0 under manual annotation.** This is not a bug — it is a genuine limitation of mean-expression marker scoring for transcriptionally similar subtypes.

FCGR3A+ Monocytes share the monocyte transcriptional programme (LYZ, S100A8, S100A9) with CD14+ Monocytes. Their distinguishing markers (FCGR3A, MS4A7, IFITM3) are expressed at lower absolute levels. When manual scoring computes mean expression across a cluster, the shared high-abundance monocyte markers dominate over the subtype-specific ones, causing misassignment.

The correct approach for subtypes is reference-based annotation (CellTypist), which correctly distinguishes these populations — and which is exactly what this pipeline uses as the oracle. The limitation is in the manual baseline, not in the framework.

---

## Challenges and solutions

Eight real engineering failures were encountered and resolved. Selected highlights:

| Challenge | Root cause | Fix |
|---|---|---|
| 10x CDN returns 403 | CDN blocks programmatic downloads | Switched to `scanpy.datasets.pbmc3k()` via figshare |
| `adata.raw = adata` corrupts the snapshot | Assignment passes by reference, not by value | `adata.raw = adata.copy()` before any in-place ops |
| CellTypist rejects the expression matrix | `.raw.X` holds pre-normalisation integers, not log1p-normalised values | Saved `layers['log_norm'] = X.copy()` after `log1p`, before `scale` |
| CellTypist benchmark kappa=0.27 (should be ~0.89) | 5 of 53 Immune_All_Low labels missing from harmonisation map | Built exhaustive 53-label map; regression test prevents recurrence |
| Wilcoxon warning despite `use_raw=True` | Known scanpy bug — warning fires incorrectly | Suppressed at call site with `warnings.filterwarnings` |
| `seurat_v3` HVG selection fails | Requires `skmisc` package; wrong input type post-normalisation | Switched to `seurat` flavour on log-normalised data |

Full challenge documentation in [ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md).

---

## Project structure

```
sc-celltype-benchmark/
├── config/settings.py          # Pydantic Settings — all parameters configurable
├── src/
│   ├── pipeline.py             # End-to-end runner (python3 -m src.pipeline)
│   ├── data_loader.py          # Download and load 10x datasets
│   ├── quality_control.py      # QC metrics, cell/gene filtering
│   ├── preprocessing.py        # Normalise, HVG, scale, PCA
│   ├── clustering.py           # kNN, Leiden, UMAP
│   ├── markers.py              # Wilcoxon DE, canonical PBMC markers
│   ├── annotation.py           # Manual scoring + CellTypist + harmonisation
│   ├── visualization.py        # 8 publication-quality figures
│   └── benchmark/
│       ├── oracle.py           # Ground-truth management, stratified holdout
│       ├── evaluator.py        # F1, kappa, confusion matrix, JSON report
│       ├── validators.py       # Biological constraint checks
│       └── difficulty.py       # Easy/medium/hard tier assignment
├── tests/                      # 168 tests, 79% coverage
├── evidence/
│   ├── figures/                # 8 PNG plots (generated)
│   ├── reports/                # benchmark_report.json (generated)
│   └── screenshots/            # Deployment evidence
├── .github/workflows/ci.yml    # GitHub Actions: lint + test on every push
├── requirements.txt
└── pyproject.toml
```

---

## Run tests

```bash
python3 -m pytest tests/ -v
```

## Configuration

All parameters are configurable via environment variables prefixed `SC_`:

```bash
SC_LEIDEN_RESOLUTION=1.5 python3 -m src.pipeline
SC_BENCHMARK_HOLDOUT_FRACTION=0.3 python3 -m src.pipeline
```

See `.env.example` for the full list.

---

## Tools and libraries

scanpy · anndata · CellTypist · scikit-learn · scipy · pandas · numpy · matplotlib · seaborn · pytest · ruff · GitHub Actions

---

**Author:** O.J. Odimayo — Bioinformatics Data Engineer
[github.com/gbadedata](https://github.com/gbadedata) · oluwagbadeodimayo@gmail.com
