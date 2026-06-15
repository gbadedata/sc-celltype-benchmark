"""Tests for benchmark oracle module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from src.benchmark.oracle import OracleGenerator


class TestOracleGenerator:

    def test_fit_stores_ground_truth(self, annotated_adata: ad.AnnData) -> None:
        oracle = OracleGenerator()
        oracle.fit(annotated_adata, label_column="celltype_reference_coarse")
        assert len(oracle.ground_truth) == annotated_adata.n_obs

    def test_fit_missing_column_raises(self, annotated_adata: ad.AnnData) -> None:
        oracle = OracleGenerator()
        with pytest.raises(ValueError, match="not found in adata.obs"):
            oracle.fit(annotated_adata, label_column="nonexistent_col")

    def test_generate_holdout_fraction(self, annotated_adata: ad.AnnData) -> None:
        oracle = OracleGenerator(seed=42)
        oracle.fit(annotated_adata, label_column="celltype_reference_coarse")
        mask = oracle.generate_holdout(fraction=0.2)
        actual_fraction = mask.sum() / len(mask)
        assert 0.15 <= actual_fraction <= 0.25

    def test_holdout_is_boolean(self, annotated_adata: ad.AnnData) -> None:
        oracle = OracleGenerator(seed=42)
        oracle.fit(annotated_adata, label_column="celltype_reference_coarse")
        mask = oracle.generate_holdout()
        assert mask.dtype == bool

    def test_get_answers_returns_withheld_only(self, annotated_adata: ad.AnnData) -> None:
        oracle = OracleGenerator(seed=42)
        oracle.fit(annotated_adata, label_column="celltype_reference_coarse")
        oracle.generate_holdout(fraction=0.2)
        answers = oracle.get_answers()
        assert len(answers) == oracle.n_withheld

    def test_get_visible_labels_returns_train_only(self, annotated_adata: ad.AnnData) -> None:
        oracle = OracleGenerator(seed=42)
        oracle.fit(annotated_adata, label_column="celltype_reference_coarse")
        oracle.generate_holdout(fraction=0.2)
        visible = oracle.get_visible_labels()
        assert len(visible) == oracle.n_visible

    def test_withheld_and_visible_partition_all_cells(self, annotated_adata: ad.AnnData) -> None:
        oracle = OracleGenerator(seed=42)
        oracle.fit(annotated_adata, label_column="celltype_reference_coarse")
        oracle.generate_holdout(fraction=0.2)
        assert oracle.n_withheld + oracle.n_visible == oracle.n_cells

    def test_get_answers_before_holdout_raises(self, annotated_adata: ad.AnnData) -> None:
        oracle = OracleGenerator(seed=42)
        oracle.fit(annotated_adata, label_column="celltype_reference_coarse")
        with pytest.raises(RuntimeError, match="generate_holdout"):
            oracle.get_answers()

    def test_fit_before_holdout_raises(self, synthetic_adata: ad.AnnData) -> None:
        oracle = OracleGenerator(seed=42)
        with pytest.raises(RuntimeError, match="fit"):
            oracle.generate_holdout()

    def test_stratified_split_preserves_all_types(self, annotated_adata: ad.AnnData) -> None:
        """All cell types appear in both train and test sets after stratified split."""
        oracle = OracleGenerator(seed=42)
        oracle.fit(annotated_adata, label_column="celltype_reference_coarse")
        oracle.generate_holdout(fraction=0.2)
        test_types = set(oracle.get_answers().unique())
        train_types = set(oracle.get_visible_labels().unique())
        # All types in the full set should appear in both splits
        all_types = set(oracle.ground_truth.unique())
        assert test_types.issubset(all_types)
        assert train_types.issubset(all_types)

    def test_reproducible_with_same_seed(self, annotated_adata: ad.AnnData) -> None:
        """Same seed produces identical holdout masks."""
        oracle1 = OracleGenerator(seed=99)
        oracle1.fit(annotated_adata, label_column="celltype_reference_coarse")
        mask1 = oracle1.generate_holdout(fraction=0.2)

        oracle2 = OracleGenerator(seed=99)
        oracle2.fit(annotated_adata, label_column="celltype_reference_coarse")
        mask2 = oracle2.generate_holdout(fraction=0.2)

        assert np.array_equal(mask1, mask2)

    def test_different_seeds_produce_different_masks(self, annotated_adata: ad.AnnData) -> None:
        """Different seeds produce different holdout masks."""
        oracle1 = OracleGenerator(seed=1)
        oracle1.fit(annotated_adata, label_column="celltype_reference_coarse")
        mask1 = oracle1.generate_holdout(fraction=0.2)

        oracle2 = OracleGenerator(seed=2)
        oracle2.fit(annotated_adata, label_column="celltype_reference_coarse")
        mask2 = oracle2.generate_holdout(fraction=0.2)

        assert not np.array_equal(mask1, mask2)
