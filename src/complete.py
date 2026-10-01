# -*- coding: utf-8 -*-
"""Complete linkage (nejvzdálenější soused)."""

import numpy as np

from src.base import HierarchicalClustering


class CompleteLinkage(HierarchicalClustering):
    """
    Hierarchické shlukování metodou nejvzdálenějšího souseda (complete linkage).

    Vzdálenost dvou shluků = MAXIMUM vzdáleností přes všechny páry objektů.
    """

    def _cluster_distance(
        self,
        a: list[int],
        b: list[int],
        d: np.ndarray,
    ) -> float:
        """Úkol: Vraťte maximum vzdáleností d[i][j] přes všechna i ∈ a, j ∈ b."""
        raise NotImplementedError(
            "Metoda '_cluster_distance' v CompleteLinkage nebyla implementována!"
        )
