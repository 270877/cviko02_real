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

from abc import ABC


class Distance(ABC):
    """
    Bázová třída pro výpočet vzdáleností mezi vektory.
    """
    def create_distance_matrix(self, data):
        """
        Vytvoří matici vzdáleností pro daná data.
        """
        raise NotImplementedError("Metoda 'create_distance_matrix' ještě nebyla implementována!")

class EuclideanDistance(Distance):
    """
    Vzdálenost mezi vektory x a y je definována jako
        d(x, y) = sqrt(sum((x_i - y_i)^2))
    """
class ManhattanDistance(Distance):
    """
    Vzdálenost mezi vektory x a y je definována jako
        d(x, y) = sum(|x_i - y_i|)
    """
class CosineCoeficient(Distance):
    """
    Kosinová vzdálenost mezi vektory x a y je definována jako
        d(x, y) = 1 - (x . y) / (||x|| * ||y||)
    """
