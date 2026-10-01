# -*- coding: utf-8 -*-
"""Single linkage (nejbližší soused)."""

import numpy as np

from src.base import HierarchicalClustering


class SingleLinkage(HierarchicalClustering):
    """
    Hierarchické shlukování metodou nejbližšího souseda (single linkage).

    Vzdálenost dvou shluků = MINIMUM vzdáleností přes všechny páry objektů.
    """

    def _cluster_distance(
        self,
        a: list[int],
        b: list[int],
        d: np.ndarray,
    ) -> float:
        """Úkol: Vraťte minimum vzdáleností d[i][j] přes všechna i ∈ a, j ∈ b."""
        raise NotImplementedError(
            "Metoda '_cluster_distance' v SingleLinkage nebyla implementována!"
        )
