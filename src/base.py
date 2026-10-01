# -*- coding: utf-8 -*-
"""
Bázová třída aglomerativního hierarchického shlukování.

Celý algoritmus (smyčka slévání, sestavení linkage matice Z) žije zde.
Dceřiné třídy přepisují pouze jednu metodu: _cluster_distance.

=======================================================================
PRACOVNÍ PŘÍKLAD  (4 objekty, single linkage)
=======================================================================
Matice vzdáleností D:

        0     1     2     3
   0  [ 0.0   1.0   4.0   5.0 ]
   1  [ 1.0   0.0   2.0   3.0 ]
   2  [ 4.0   2.0   0.0   1.0 ]
   3  [ 5.0   3.0   1.0   0.0 ]

n = 4  →  Z má tvar (3, 4)

Krok 0  (step=0):
  active = {0:[0], 1:[1], 2:[2], 3:[3]}
  Nejbližší pár: (0,1), vzdálenost = 1.0
  Nový shluk ID = n + step = 4 + 0 = 4,  členové = [0, 1]
  Z[0] = [0, 1, 1.0, 2]
  active = {2:[2], 3:[3], 4:[0,1]}

Krok 1  (step=1):
  Vzdálenosti (single = min přes všechny páry původních objektů):
    d(2,3) = D[2][3] = 1.0
    d(2,4) = min(D[2][0], D[2][1]) = min(4.0, 2.0) = 2.0
    d(3,4) = min(D[3][0], D[3][1]) = min(5.0, 3.0) = 3.0
  Nejbližší pár: (2,3), vzdálenost = 1.0
  Nový shluk ID = 4 + 1 = 5,  členové = [2, 3]
  Z[1] = [2, 3, 1.0, 2]
  active = {4:[0,1], 5:[2,3]}

Krok 2  (step=2):
  d(4,5) = min(D[0][2], D[0][3], D[1][2], D[1][3]) = min(4,5,2,3) = 2.0
  Nový shluk ID = 4 + 2 = 6,  členové = [0,1,2,3]
  Z[2] = [4, 5, 2.0, 4]

Výsledek:
  Z = [[ 0.,  1.,  1.,  2.],
       [ 2.,  3.,  1.,  2.],
       [ 4.,  5.,  2.,  4.]]
=======================================================================
"""

from abc import ABC, abstractmethod

import numpy as np


class HierarchicalClustering(ABC):
    """
    Bázová třída aglomerativního hierarchického shlukování.

    Algoritmus (metoda fit) je implementován jednou zde. Dceřiné třídy
    (SingleLinkage, CompleteLinkage, AverageLinkage) přepisují pouze
    _cluster_distance — tu jednu věc, v níž se od sebe liší.
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

        Args:
            A: Seznam indexů PŮVODNÍCH objektů v prvním shluku.
            B: Seznam indexů PŮVODNÍCH objektů v druhém shluku.
            D: Původní (n x n) matice vzdáleností.

        Vrací: Číslo (float) reprezentující vzdálenost mezi shluky.
        Implementuje potomek — viz single.py, complete.py, average.py.
        """
        raise NotImplementedError(
            "Metoda '_cluster_distance' ještě nebyla implementována!"
        )

    def fit(self, d: np.ndarray) -> np.ndarray:
        """
        Úkol: Naprogramujte aglomerativní smyčku a sestavte linkage matici Z.

        Vstup:
            D (np.ndarray): Čtvercová symetrická matice vzdáleností tvaru (n×n).
                            Diagonála = 0. Výstup Distance.create_distance_matrix().

        Výstup:
            Z (np.ndarray): Linkage matice tvaru (n-1, 4). Každý řádek popisuje
                            jedno sloučení:
                            [ID_A, ID_B, vzdálenost, velikost_nového_shluku]

        Postup (implementujte krok za krokem):

        1.  n = D.shape[0]
            Inicializujte slovník aktivních shluků:
                active = {i: [i] for i in range(n)}
            Klíč = ID shluku, hodnota = seznam indexů původních objektů.

        2.  Předalokujte výstup:
                Z = np.zeros((n - 1, 4))

        3.  Smyčka  for step in range(n - 1):

            a) Projděte všechny PÁRY aktivních shluků. Pro každý pár (id_a, id_b)
               zavolejte self._cluster_distance(active[id_a], active[id_b], D).

            b) Najděte pár s MINIMÁLNÍ vzdáleností.
               id_i = min(id_a, id_b)   ← menší ID do sloupce 0
               id_j = max(id_a, id_b)   ← větší ID do sloupce 1

            c) Zapište řádek linkage matice:
                Z[step, 0] = id_i
                Z[step, 1] = id_j
                Z[step, 2] = minimální vzdálenost
                Z[step, 3] = len(active[id_i]) + len(active[id_j])

            d) Přidejte nový shluk a odstraňte sloučené:
                active[n + step] = active[id_i] + active[id_j]
                del active[id_i]
                del active[id_j]

        4.  self.z = z.astype(float)
            return self.z

        POZOR — tři časté chyby:
        • ID nového shluku je  n + step  (číslo), ne pořadové číslo ve slovníku.
        • z musí být dtype=float — scipy odmítne dendrogram jinak.
        • Při shodě vzdáleností (tie) se vaše pořadí fúzí může lišit od scipy;
          to je správně. Testujte výšky a velikosti, ne konkrétní ID.
        """
        # assert  Doplňte podmínku: vstupní data musí být numpy array (typ np.ndarray)
        # assert  Doplňte podmínku: matice musí být 2D (počet dimenzí == 2)
        # assert  Doplňte podmínku: matice musí být čtvercová (d.shape[0] == d.shape[1])
        raise NotImplementedError("Metoda 'fit' ještě nebyla implementována!")

    def fit_predict(self, d: np.ndarray, k: int) -> np.ndarray:
        """
        Úkol: Spusťte shlukování, zastavte se při k shlucích, vraťte labely.

        Vstup:
            d (np.ndarray): Matice vzdáleností (n x n).
            k (int):        Požadovaný počet výsledných shluků  (1 ≤ k ≤ n).

        Výstup:
            labels (np.ndarray): Pole délky n. labels[i] = číslo shluku objektu i.
                                  Labely jsou 0, 1, ..., k-1 (v libovolném pořadí).

        Postup:
            Stejná smyčka jako ve fit(), ale provedená pouze (n - k)-krát.
            Po skončení zbývá v active přesně k shluků.
            Přiřaďte každému zbývajícímu shluku label 0, 1, ..., k-1
            a vyplňte labels[i] = label toho shluku, jehož je objekt i členem.

        Hint: Nejjednodušší implementace vůbec nevolá fit() — jen zkopíruje smyčku
              a zastaví ji po (n - k) krocích. Alternativně: zavolejte self.fit(d)
              a pak přečtěte finální shluky z posledních (k-1) řádků matice z.
        """
        # assert  Doplňte podmínku: k musí být >= 1 a <= počtu objektů (d.shape[0])
        raise NotImplementedError("Metoda 'fit_predict' ještě nebyla implementována!")
