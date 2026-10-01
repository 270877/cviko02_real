# -*- coding: utf-8 -*-
"""Average linkage / UPGMA (průměr vzdáleností)."""

import numpy as np

from src.base import HierarchicalClustering


class AverageLinkage(HierarchicalClustering):
    """
    Hierarchické shlukování metodou průměrných vzdáleností (average linkage = UPGMA).

    Vzdálenost dvou shluků = PRŮMĚR všech párových vzdáleností D[i][j], i ∈ A, j ∈ B.

    Poznámka: Průměr přes všechny původní párové vzdálenosti (UPGMA) odpovídá
    tomu, co scipy počítá metodou 'average' — výsledky by se měly shodovat.
    """

    def _cluster_distance(
        self,
        a: list[int],
        b: list[int],
        d: np.ndarray,
    ) -> float:
        """
        Úkol: Vraťte průměr vzdáleností d[i][j] přes všechna i ∈ a, j ∈ b.

        Hint: np.mean([d[i][j] for i in a for j in b])
        """
        raise NotImplementedError(
            "Metoda '_cluster_distance' v AverageLinkage nebyla implementována!"
        )
