"""Evaluation benchmark framework for cell-type annotation.

This package implements a scientific evaluation framework that tests
how accurately an annotation pipeline recovers known cell identities.
It includes oracle answer generation, difficulty tier design,
validator functions, and scoring metrics.
"""

from src.benchmark.difficulty import DifficultyTier, assign_difficulty_tiers
from src.benchmark.evaluator import BenchmarkEvaluator
from src.benchmark.oracle import OracleGenerator
from src.benchmark.validators import (
    validate_biological_consistency,
    validate_cluster_purity,
    validate_marker_recovery,
)

__all__ = [
    "BenchmarkEvaluator",
    "DifficultyTier",
    "OracleGenerator",
    "assign_difficulty_tiers",
    "validate_biological_consistency",
    "validate_cluster_purity",
    "validate_marker_recovery",
]
