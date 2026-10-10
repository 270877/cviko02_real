# -*- coding: utf-8 -*-
"""Metriky kvality shlukování: intra-cluster inertia a křivka lokte."""

import numpy as np


def within_cluster_inertia(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Úkol: Vypočtěte součet kvadrátů vzdáleností každého bodu od těžiště jeho shluku.

    Vstup:
        data   (np.ndarray): Matice surových dat (n_objektů x n_příznaků).
        labels (np.ndarray): Pole délky n s číslem shluku pro každý objekt
                             (výstup fit_predict).

    Výstup:
        inertia (float): Skalár — celková intra-cluster inertia.

    Vzorec:  inertia = Σ_k  Σ_{i ∈ C_k}  ||x_i - μ_k||²
             kde μ_k je těžiště (centroid) shluku C_k.

    POZOR: Tato funkce potřebuje SUROVÁ DATA (souřadnice), ne matici vzdáleností!
    Shlukování samo probíhá pouze na základě matice vzdáleností — ale k výpočtu
    těžiště musíme znát polohy bodů v prostoru příznaků. Toto je záměrný výukový
    kontrast: pro shlukování stačí vzdálenosti, pro inertii potřebujeme souřadnice.

    Postup:
        for každý unikátní shluk k in np.unique(labels):
            X_k    = data[labels == k]                  # body v tomto shluku
            centroid = np.mean(X_k, axis=0)             # těžiště
            inertia += np.sum(np.linalg.norm(X_k - centroid, axis=1)**2)
            # součet kvadrátů vzdáleností
    """
    inertia = 0.0
    for k in np.unique(labels):
        X_k = data[labels == k]
        centroid = np.mean(X_k, axis=0)
        inertia += float(np.sum(np.linalg.norm(X_k - centroid, axis=1) ** 2))
    return inertia

    raise NotImplementedError("Funkce 'within_cluster_inertia' ještě nebyla implementována!")


def inertia_curve(
    model,
    d: np.ndarray,
    data: np.ndarray,
    k_max: int | None = None,
) -> tuple[list[int], list[float]]:
    """
    PRE-FILLED: Pro k = 1 .. k_max zavolá model.fit_predict(d, k),
    spočítá within_cluster_inertia a vrátí (k_values, inertias).

    Args:
        model:  Instance HierarchicalClustering (SingleLinkage apod.).
        d:      Matice vzdáleností (n x n).
        data:   Surová data (n x p) — předávána do within_cluster_inertia.
        k_max:  Maximální počet shluků (výchozí: n).

    Vrací:
        k_values  (list[int]):   [1, 2, ..., k_max]
        inertias  (list[float]): Odpovídající hodnoty inercie.
    """
    n = d.shape[0]
    if k_max is None:
        k_max = n
    k_values = list(range(1, k_max + 1))
    inertias = []
    for k in k_values:
        labels = model.fit_predict(d, k)
        inertias.append(within_cluster_inertia(data, labels))
    return k_values, inertias

def find_elbow(k_values: list[int], inertias: list[float]) -> int:
    """
    Najde bod záhybu (elbow) v křivce inercie.

    Args:
        k_values (list[int]): Hodnoty počtu shluků.
        inertias (list[float]): Odpovídající hodnoty inercie.

    Returns:
        int: Počet shluků odpovídající záhybu v křivce.
    """
    if len(k_values) < 3:
        return k_values[0]

        # Výpočet poklesu inercie medzi susednými bodmi (prvé diferencie)
    inertias_arr = np.array(inertias)
    diffs = np.diff(inertias_arr)  # záporné hodnoty, pretože inercia klesá
    diffs_drop = -diffs  # kladné hodnoty veľkosti poklesu

    # Loket zodpovedá mieste s najväčším poklesom (maximum prvej diferencie)
    # Index v diffs zodpovedá prechodu z k_values[i] na k_values[i+1]
    elbow_idx = int(np.argmax(diffs_drop))
    return k_values[elbow_idx]

    raise NotImplementedError("Funkce 'find_elbow' ještě nebyla implementována!")
