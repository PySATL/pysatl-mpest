"""Strategies for pruning components from mixture models."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from .prior_threshold_pruner import PriorThresholdPruner

__all__ = ["PriorThresholdPruner"]
