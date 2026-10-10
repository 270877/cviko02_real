# -*- coding: utf-8 -*-
"""
Bázová třída aglomerativního hierarchického shlukování.
"""

from abc import ABC, abstractmethod
import numpy as np

class HierarchicalClustering(ABC):
    """
    Bázová třída aglomerativního hierarchického shlukování.
    """

    def __init__(self) -> None:
        self.z: np.ndarray | None = None

    @abstractmethod
    def _cluster_distance(
        self,
        a: list[int],
        b: list[int],
        d: np.ndarray,
    ) -> float:
        """
        Vzdálenost mezi dvěma shluky A a B z původní matice vzdáleností D.
        """
        raise NotImplementedError(
            "Metoda '_cluster_distance' ještě nebyla implementována!"
        )

    def fit(self, d: np.ndarray) -> np.ndarray:
        """
        Naprogramujte aglomerativní smyčku a sestavte linkage matici Z.
        """
        # tři příkazy assert před smyčkou: vstup musí být np.ndarray, 2D a čtvercový.
        assert isinstance(d, np.ndarray)
        assert d.ndim == 2
        assert d.shape[0] == d.shape[1]

        n = d.shape[0]
        active = {i: [i] for i in range(n)}
        z = np.zeros((n - 1, 4))

        for step in range(n - 1):
            min_dist = float("inf")
            best_pair = (-1, -1)

            active_ids = list(active.keys())
            for i in range(len(active_ids)):
                for j in range(i + 1, len(active_ids)):
                    id_a = active_ids[i]
                    id_b = active_ids[j]
                    dist = self._cluster_distance(active[id_a], active[id_b], d)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (id_a, id_b)

            id_i = min(best_pair)
            id_j = max(best_pair)

            z[step, 0] = id_i
            z[step, 1] = id_j
            z[step, 2] = min_dist
            z[step, 3] = len(active[id_i]) + len(active[id_j])

            active[n + step] = active[id_i] + active[id_j]
            del active[id_i]
            del active[id_j]

        self.z = z.astype(float)
        return self.z

    def fit_predict(self, d: np.ndarray, k: int) -> np.ndarray:
        """
        Spusťte shlukování, zastavte se při k shlucích, vraťte labely.
        """
        assert isinstance(d, np.ndarray)
        assert d.ndim == 2
        assert d.shape[0] == d.shape[1]
        assert 1 <= k <= d.shape[0]

        n = d.shape[0]
        active = {i: [i] for i in range(n)}

        for step in range(n - k):
            min_dist = float("inf")
            best_pair = (-1, -1)

            active_ids = list(active.keys())
            for i in range(len(active_ids)):
                for j in range(i + 1, len(active_ids)):
                    id_a = active_ids[i]
                    id_b = active_ids[j]
                    dist = self._cluster_distance(active[id_a], active[id_b], d)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (id_a, id_b)

            id_i = min(best_pair)
            id_j = max(best_pair)

            active[n + step] = active[id_i] + active[id_j]
            del active[id_i]
            del active[id_j]

        labels = np.zeros(n, dtype=int)
        for cluster_label, member_indices in enumerate(active.values()):
            for idx in member_indices:
                labels[idx] = cluster_label

        return labels