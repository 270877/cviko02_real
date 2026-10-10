# -*- coding: utf-8 -*-
"""
Třídy pro výpočet vzdáleností mezi vektory.

ÚKOL: Zkopírujte sem svou implementaci z Cvičení 01.

Potřebné třídy a jejich veřejné rozhraní:

    class Distance(ABC)
        .is_metric                             → bool  (abstraktní vlastnost)
        .calculate(x: np.ndarray, y: np.ndarray) → float
        .create_distance_matrix(data: np.ndarray) → np.ndarray

    class EuclideanDistance(Distance)
    class ManhattanDistance(Distance)
    class CosineCoeficient(Distance)

Bez tohoto kódu se shlukování nespustí — ImportError vám to připomene.
"""

from abc import ABC, abstractmethod
import numpy as np

class Distance(ABC):
    """
    Bázová třída pro výpočet vzdáleností mezi vektory.
    """
    def create_distance_matrix(self, data):
        """
        Vytvoří matici vzdáleností pro daná data.
        """
        # raise NotImplementedError("Metoda 'create_distance_matrix' ještě nebyla implementována!")
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        # Create an empty distance matrix
        n_samples = data.shape[0]
        matice = np.zeros((n_samples, n_samples))
        # for i in range(number of samples):
        #     for j in range(i+1, number of samples):
        #         call self.calculate(x,y) to fill the distance matrix symmetrically
        # return the distance matrix
        for idx in range(n_samples):
            for jdx in range(idx + 1, n_samples):
                distance = self.calculate(data[idx], data[jdx])
                matice[idx, jdx] = distance
                matice[jdx, idx] = distance

        return matice

class EuclideanDistance(Distance):
    """
    Vzdálenost mezi vektory x a y je definována jako
        d(x, y) = sqrt(sum((x_i - y_i)^2))
    """
    def __init__(self) -> None:
        """
        Inicializace třídy. Zde není potřeba žádný atribut, ale můžete si přidat, pokud chcete."""

    @property
    def is_metric(self) -> bool:
        """Vrací "True", protože Euklidovská vzdálenost "splnuje" definici metriky."""
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Úkol: Spočtěte Euklidovskou vzdálenost mezi 1D vektory x a y."""
        solution = float(np.sqrt(np.sum((x-y)**2)))
        return solution


class ManhattanDistance(Distance):
    """
    Vzdálenost mezi vektory x a y je definována jako
        d(x, y) = sum(|x_i - y_i|)
    """
    @property
    def is_metric(self) -> bool:
        """Vrací "True", protože Manhattanská vzdálenost "splnuje" definici metriky."""
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Úkol: Spočtěte Manhattanskou vzdálenost mezi 1D vektory x a y."""
        solution = float(np.sum(np.abs(x - y)))
        return solution

class CosineCoeficient(Distance):
    """
    Kosinová vzdálenost mezi vektory x a y je definována jako
        d(x, y) = 1 - (x . y) / (||x|| * ||y||)
    """

    @property
    def is_metric(self) -> bool:
        """Vrací "False", protože Cosinový koeficient "nesplnuje" definici metriky."""
        return False

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Úkol: Spočtěte Cosinový koeficient mezi 1D vektory x a y."""
        dot_solution = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)

        if norm_x == 0 or norm_y == 0:
            return 0.0

        cosine_sim = dot_solution / (norm_x * norm_y)

        return float(cosine_sim)