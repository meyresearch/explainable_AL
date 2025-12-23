"""Compatibility shim: provide `explainable_al.ecfp.run_experiment`.

The canonical experiment runner has been moved to :mod:`explainable_al.experiments`.
This module is a thin shim that keeps the old import path working.
"""

from .experiments import run_experiment

__all__ = ["run_experiment"]

