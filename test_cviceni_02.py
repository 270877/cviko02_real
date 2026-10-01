# -*- coding: utf-8 -*-
"""
Smoke testy pro Cvičení 02 — Hierarchické shlukování.

Každá implementace studenta je porovnána s referenčním výsledkem scipy.
Testy jsou navrženy tak, aby fungovaly i tehdy, když student ještě nemá
implementovanou třídu Distance z Cvičení 01 (DummyDistance to obchází).

Spuštění:
    python -m pytest test_cviceni_02.py -v
"""

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# ======================================================================
# DummyDistance — izoluje testy od src/distance.py (Cvičení 01)
# ======================================================================

class DummyDistance:
    """
    Plně implementovaná Euklidovská vzdálenost, nezávislá na src/distance.py.
    Slouží výhradně pro izolované testování shlukování — studenti ji nepíší.
    """
    def create_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Vytvoří matici vzdáleností pro daná data."""
        n = data.shape[0]
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.sqrt(np.sum((data[i] - data[j]) ** 2)))
                d[i, j] = dist
                d[j, i] = dist
        return d


# ======================================================================
# Fixní testovací matice vzdáleností (4×4)
# ======================================================================

#       0     1     2     3
d_test = np.array([
    [0.0, 1.0, 4.0, 5.0],
    [1.0, 0.0, 2.0, 3.0],
    [4.0, 2.0, 0.0, 1.0],
    [5.0, 3.0, 1.0, 0.0],
], dtype=float)

# Referenční výsledky scipy pro single/complete/average
_condensed       = squareform(d_test)
z_single_ref     = linkage(_condensed, method="single")
z_complete_ref   = linkage(_condensed, method="complete")
z_average_ref    = linkage(_condensed, method="average")


# ======================================================================
# Pomocné funkce pro porovnávání Z matic
# ======================================================================

def sorted_heights(z: np.ndarray) -> list[float]:
    """Výšky fúzí seřazené vzestupně — porovnání nezávislé na pořadí při shodách."""
    return sorted(z[:, 2].tolist())


def sorted_sizes(z: np.ndarray) -> list[int]:
    """Velikosti nových shluků seřazené vzestupně."""
    return sorted(z[:, 3].astype(int).tolist())


# ======================================================================
# TESTY — SingleLinkage
# ======================================================================

class TestSingleLinkage:
    """
    Testy pro SingleLinkage.
    Používají DummyDistance — nevyžadují implementaci z Cvičení 01.
    """

    def test_import(self):
        """Testuje import třídy SingleLinkage."""
        from src.single import SingleLinkage  # noqa: F401

    def test_fit_shape(self):
        """Testuje, že fit() vrací linkage matici správného tvaru (n-1, 4)."""
        from src.single import SingleLinkage
        z = SingleLinkage().fit(d_test)
        assert z.shape == (3, 4), f"Tvar Z by měl být (3, 4), dostali jste {z.shape}"

    def test_fit_dtype_float(self):
        """Testuje, že fit() vrací linkage matici typu float."""
        from src.single import SingleLinkage
        z = SingleLinkage().fit(d_test)
        assert np.issubdtype(z.dtype, np.floating), \
            "Z musí být dtype=float (scipy jinak odmítne dendrogram)"

    def test_fit_heights_match_scipy(self):
        """Testuje, že výšky fúzí odpovídají referenčnímu výsledku scipy."""
        from src.single import SingleLinkage
        z = SingleLinkage().fit(d_test)
        assert sorted_heights(z) == pytest.approx(sorted_heights(z_single_ref), abs=1e-9), (
            f"Výšky fúzí single linkage neodpovídají scipy.\n"
            f"  Student: {sorted_heights(z)}\n"
            f"  Scipy:   {sorted_heights(z_single_ref)}"
        )

    def test_fit_sizes_match_scipy(self):
        """Testuje, že velikosti nových shluků odpovídají referenčnímu výsledku scipy."""
        from src.single import SingleLinkage
        z = SingleLinkage().fit(d_test)
        assert sorted_sizes(z) == sorted_sizes(z_single_ref), (
            f"Velikosti shluků single linkage neodpovídají scipy.\n"
            f"  Student: {sorted_sizes(z)}\n"
            f"  Scipy:   {sorted_sizes(z_single_ref)}"
        )

    def test_fit_predict_k2_label_count(self):
        """Testuje, že fit_predict() vrací správný počet labelů pro k=2."""
        from src.single import SingleLinkage
        labels = SingleLinkage().fit_predict(d_test, k=2)
        assert labels.shape == (4,)
        assert len(set(labels.tolist())) == 2, "Pro k=2 musí být právě 2 různé labely"

    def test_fit_predict_k1_one_cluster(self):
        """Testuje, že fit_predict() vrací jeden shluk pro k=1."""
        from src.single import SingleLinkage
        labels = SingleLinkage().fit_predict(d_test, k=1)
        assert len(set(labels.tolist())) == 1, \
            "Pro k=1 musí být všechny objekty v jednom shluku"

    def test_fit_predict_k_equals_n_singletons(self):
        """Testuje, že fit_predict() vrací každý objekt ve vlastním shluku pro k=n."""
        from src.single import SingleLinkage
        labels = SingleLinkage().fit_predict(d_test, k=4)
        assert len(set(labels.tolist())) == 4, \
            "Pro k=n musí mít každý objekt vlastní shluk"


# ======================================================================
# TESTY — CompleteLinkage
# ======================================================================

class TestCompleteLinkage:
    """
    Testy pro CompleteLinkage.
    Používají DummyDistance — nevyžadují implementaci z Cvičení 01.
    """

    def test_fit_heights_match_scipy(self):
        """Testuje, že výšky fúzí odpovídají referenčnímu výsledku scipy."""
        from src.complete import CompleteLinkage
        z = CompleteLinkage().fit(d_test)
        assert sorted_heights(z) == pytest.approx(sorted_heights(z_complete_ref), abs=1e-9), (
            f"Výšky fúzí complete linkage neodpovídají scipy.\n"
            f"  Student: {sorted_heights(z)}\n"
            f"  Scipy:   {sorted_heights(z_complete_ref)}"
        )

    def test_fit_sizes_match_scipy(self):
        """Testuje, že velikosti nových shluků odpovídají referenčnímu výsledku scipy."""
        from src.complete import CompleteLinkage
        z = CompleteLinkage().fit(d_test)
        assert sorted_sizes(z) == sorted_sizes(z_complete_ref)


# ======================================================================
# TESTY — AverageLinkage
# ======================================================================

class TestAverageLinkage:
    """
    Testy pro AverageLinkage.
    Používají DummyDistance — nevyžadují implementaci z Cvičení 01.
    """

    def test_fit_heights_match_scipy(self):
        """Testuje, že výšky fúzí odpovídají referenčnímu výsledku scipy."""
        from src.average import AverageLinkage
        z = AverageLinkage().fit(d_test)
        assert sorted_heights(z) == pytest.approx(sorted_heights(z_average_ref), abs=1e-9), (
            f"Výšky fúzí average linkage neodpovídají scipy.\n"
            f"  Student: {sorted_heights(z)}\n"
            f"  Scipy:   {sorted_heights(z_average_ref)}"
        )

    def test_fit_sizes_match_scipy(self):
        """Testuje, že velikosti nových shluků odpovídají referenčnímu výsledku scipy."""
        from src.average import AverageLinkage
        z = AverageLinkage().fit(d_test)
        assert sorted_sizes(z) == sorted_sizes(z_average_ref)


# ======================================================================
# TESTY — within_cluster_inertia
# ======================================================================

class TestInertia:
    """Inertia testy — nevyžadují implementaci Distance z Cvičení 01."""

    def test_two_tight_clusters(self):
        """Testuje výpočet inertia pro dva těsné shluky."""
        from src.inertia import within_cluster_inertia
        data   = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        # Shluk 0: těžiště=(0.5, 0) -> 0.25 + 0.25 = 0.5
        # Shluk 1: těžiště=(10.5, 0) -> 0.25 + 0.25 = 0.5
        assert within_cluster_inertia(data, labels) == pytest.approx(1.0, abs=1e-9)

    def test_single_cluster(self):
        """Testuje výpočet inertia pro jediný shluk."""
        from src.inertia import within_cluster_inertia
        data   = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        labels = np.array([0, 0, 0])
        # těžiště=(2, 0) -> 4 + 0 + 4 = 8
        assert within_cluster_inertia(data, labels) == pytest.approx(8.0, abs=1e-9)

    def test_all_singletons_zero_inertia(self):
        """Testuje, že každý bod ve vlastním shluku -> inertia=0."""
        from src.inertia import within_cluster_inertia
        data   = np.array([[0.0], [1.0], [2.0]])
        labels = np.array([0, 1, 2])
        assert within_cluster_inertia(data, labels) == pytest.approx(0.0, abs=1e-9), \
            "Každý bod je ve svém shluku -> inertia musí být 0"


# ======================================================================
# Testovací data pro Ward (surové souřadnice, ne matice vzdáleností)
# ======================================================================

#  Dva těsné páry daleko od sebe:
#    0-1          2-3
#  [0,0][1,0]  [4,0][5,0]
#
# Očekávané sloučení: (0,1) a (2,3) ve výšce 1.0,
# pak výsledné páry ve výšce 4*sqrt(2) ~ 5.657

data_ward = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [4.0, 0.0],
    [5.0, 0.0],
], dtype=float)

z_ward_ref = linkage(data_ward, method="ward")


# ======================================================================
# TESTY — WardLinkage
# ======================================================================

class TestWardLinkage:
    """
    Testy pro WardLinkage.
    Vstupem jsou surové souřadnice (data_ward), ne matice vzdáleností.
    Referenční výsledky jsou vypočítány pomocí scipy.
    """

    def test_import(self):
        """Testuje import třídy WardLinkage."""
        from src.ward import WardLinkage  # noqa: F401

    def test_fit_shape(self):
        """Testuje, že fit() vrací linkage matici správného tvaru (n-1, 4)."""
        from src.ward import WardLinkage
        z = WardLinkage().fit(data_ward)
        assert z.shape == (3, 4), f"Tvar Z by měl být (3, 4), dostali jste {z.shape}"

    def test_fit_dtype_float(self):
        """Testuje, že fit() vrací matici dtype=float (scipy jinak odmítne dendrogram)."""
        from src.ward import WardLinkage
        z = WardLinkage().fit(data_ward)
        assert np.issubdtype(z.dtype, np.floating), "Z musí být dtype=float"

    def test_fit_heights_match_scipy(self):
        """Testuje, že výšky fúzí odpovídají referenčnímu výsledku scipy."""
        from src.ward import WardLinkage
        z = WardLinkage().fit(data_ward)
        assert sorted_heights(z) == pytest.approx(sorted_heights(z_ward_ref), abs=1e-9), (
            f"Výšky fúzí Ward linkage neodpovídají scipy.\n"
            f"  Student: {sorted_heights(z)}\n"
            f"  Scipy:   {sorted_heights(z_ward_ref)}"
        )

    def test_fit_sizes_match_scipy(self):
        """Testuje, že velikosti nových shluků odpovídají referenčnímu výsledku scipy."""
        from src.ward import WardLinkage
        z = WardLinkage().fit(data_ward)
        assert sorted_sizes(z) == sorted_sizes(z_ward_ref), (
            f"Velikosti shluků Ward linkage neodpovídají scipy.\n"
            f"  Student: {sorted_sizes(z)}\n"
            f"  Scipy:   {sorted_sizes(z_ward_ref)}"
        )

    def test_fit_predict_k2_separates_pairs(self):
        """
        Testuje, že fit_predict(k=2) správně oddělí oba těsné páry.
        Objekty 0 a 1 musí být ve stejném shluku; objekty 2 a 3 ve druhém.
        """
        from src.ward import WardLinkage
        labels = WardLinkage().fit_predict(data_ward, k=2)
        assert labels.shape == (4,), f"labels musí mít délku 4, dostali jste {labels.shape}"
        assert len(set(labels.tolist())) == 2, "Pro k=2 musí být právě 2 různé labely"
        assert labels[0] == labels[1], "Objekty 0 a 1 (blízký pár) musí být ve stejném shluku"
        assert labels[2] == labels[3], "Objekty 2 a 3 (blízký pár) musí být ve stejném shluku"
        assert labels[0] != labels[2], "Oba páry musí být v různých shlucích"

    def test_fit_predict_k1_one_cluster(self):
        """Testuje, že fit_predict(k=1) vrátí všechny objekty v jednom shluku."""
        from src.ward import WardLinkage
        labels = WardLinkage().fit_predict(data_ward, k=1)
        assert len(set(labels.tolist())) == 1, \
            "Pro k=1 musí být všechny objekty v jednom shluku"
