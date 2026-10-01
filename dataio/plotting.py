# -*- coding: utf-8 -*-
"""Vizualizační nástroje: dendrogram a křivka inercie."""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(
    z: np.ndarray,
    labels: list[str],
    title: str = "Dendrogram",
    save_path: str | None = None,
) -> None:
    """Vykreslí dendrogram z linkage matice z pomocí scipy.

    Slouží také jako validátor: pokud je z nesprávně sestavena,
    scipy zde vyvolá chybu — tak odlišíte chybu shlukování od chyby vykreslování.
    """
    _, ax = plt.subplots(figsize=(12, 5))
    dendrogram(z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Slovo")
    ax.set_ylabel("Vzdálenost")
    plt.tight_layout()
    if save_path:
        abs_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        plt.savefig(abs_path, dpi=150)
        print(f">>> Dendrogram uložen: {save_path}")
    plt.show()


def plot_inertia_curve(
    k_values: list[int],
    inertias: list[float],
    elbow: int | None = None,
    save_path: str | None = None,
) -> None:
    """Vykreslí vývoj inercie vs. počet shluků; volitelně zvýrazní loket."""
    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, inertias, marker="o", linewidth=2, color="steelblue")
    if elbow is not None and elbow in k_values:
        idx = k_values.index(elbow)
        ax.scatter(
            [elbow], [inertias[idx]], color="red", zorder=5, s=120,
            label=f"Loket (k={elbow})",
        )
        ax.legend()
    ax.set_xlabel("Počet shluků k")
    ax.set_ylabel("Intra-cluster inertia")
    ax.set_title("Křivka inercie")
    ax.set_xticks(k_values)
    plt.tight_layout()
    if save_path:
        abs_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        plt.savefig(abs_path, dpi=150)
        print(f">>> Křivka inercie uložena: {save_path}")
    plt.show()
