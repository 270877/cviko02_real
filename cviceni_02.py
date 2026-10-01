# -*- coding: utf-8 -*-

"""
Created on 24. 06. 2026

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Cvičení 2 Umělá inteligence v medicíně — Hierarchické shlukování
"""

import os

from dataio import (
    load_words,
    ascii_vectorize,
    keyboard_vectorize,
    plot_dendrogram,
    plot_inertia_curve
)

from src import (
    EuclideanDistance,
    SingleLinkage,
    CompleteLinkage,
    AverageLinkage,
    WardLinkage,
    inertia_curve,
    find_elbow,
)

# Dostupné datové sady — změňte dle potřeby:
#   "words_list_len5.csv" — 20 slov, délka 5 znaků
#   "words_list.csv"      — 40 slov, délka 3 znaky
DATA_PATH  = os.path.join("data", "words_list.csv")
GRAPHS_DIR = "graphs"


def main() -> None:
    """
    Hlavní funkce pro spuštění cvičení 2.
    """
    print("=" * 60)
    print(" CVIČENÍ 2: HIERARCHICKÉ SHLUKOVÁNÍ")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. NAČTENÍ DAT
    # ------------------------------------------------------------------
    print("\n[1] Načítání dat...")
    words = load_words(DATA_PATH)
    print(f"    Načtena slova ({len(words)}): {words}")

    # ------------------------------------------------------------------
    # 2. VEKTORIZACE
    # ------------------------------------------------------------------
    print("\n[2] Vektorizace slov...")

    data_ascii = None
    try:
        data_ascii = ascii_vectorize(words)
        print(f"    ASCII vektorizace OK. Tvar: {data_ascii.shape}")
    except NotImplementedError as e:
        print(f"    [INFO] {e}")

    data_keyboard = None
    try:
        data_keyboard = keyboard_vectorize(words)
        print(f"    Keyboard vektorizace OK. Tvar: {data_keyboard.shape}")
    except NotImplementedError as e:
        print(f"    [INFO] {e}")

    # Primární datová sada pro shlukování — ASCII
    data = data_ascii
    if data is None:
        print("\n[PŘESKOČENO] Implementujte ascii_vectorize() pro pokračování.")
        return

    # ------------------------------------------------------------------
    # 3. MATICE VZDÁLENOSTÍ
    # ------------------------------------------------------------------
    print("\n[3] Výpočet matice vzdáleností (Euklidovská)...")
    try:
        d = EuclideanDistance().create_distance_matrix(data)
        print(f"    Matice vzdáleností OK. Tvar: {d.shape}")
    except NotImplementedError as e:
        print(f"    [INFO] {e}")
        print("    [INFO] Dokopírujte metody Distance z Cvičení 01 do src/distance.py")
        return

    # ------------------------------------------------------------------
    # 4. SHLUKOVÁNÍ — SINGLE / COMPLETE / AVERAGE LINKAGE
    # ------------------------------------------------------------------
    print("\n[4] Hierarchické shlukování a dendrogramy...")

    methods = [
        ("Single linkage",   SingleLinkage()),
        ("Complete linkage", CompleteLinkage()),
        ("Average linkage",  AverageLinkage()),
    ]

    for name, model in methods:
        print(f"\n  --- {name} ---")
        try:
            z = model.fit(d)
            print(f"    fit() OK. z.shape = {z.shape}")
            save_path = os.path.join(
                GRAPHS_DIR, f"dendrogram_{name.split()[0].lower()}.png"
            )
            plot_dendrogram(z, labels=words, title=f"Dendrogram — {name}",
                            save_path=save_path)
        except NotImplementedError as e:
            print(f"    [INFO] {e}")

    # ------------------------------------------------------------------
    # 4b. SHLUKOVÁNÍ — WARD LINKAGE (přímé souřadnice)
    # ------------------------------------------------------------------
    print("\n[4b] Wardovo shlukování...")
    print("     (Ward nepotřebuje matici vzdáleností — pracuje přímo se souřadnicemi)")
    try:
        z_ward = WardLinkage().fit(data)
        print(f"    fit() OK. z.shape = {z_ward.shape}")
        plot_dendrogram(
            z_ward, labels=words,
            title="Dendrogram — Ward linkage",
            save_path=os.path.join(GRAPHS_DIR, "dendrogram_ward.png"),
        )
    except NotImplementedError as e:
        print(f"    [INFO] {e}")

    # ------------------------------------------------------------------
    # 5. BONUS: KEYBOARD VEKTORIZACE (druhý průchod)
    # ------------------------------------------------------------------
    if data_keyboard is not None:
        print("\n[5] BONUS: Shlukování na keyboard vektorizaci (single linkage)...")
        try:
            d_kb = EuclideanDistance().create_distance_matrix(data_keyboard)
            z_kb = SingleLinkage().fit(d_kb)
            plot_dendrogram(
                z_kb, labels=words,
                title="Dendrogram — Single linkage (keyboard)",
                save_path=os.path.join(GRAPHS_DIR, "dendrogram_keyboard.png"),
            )
        except NotImplementedError as e:
            print(f"    [INFO] {e}")

    # ------------------------------------------------------------------
    # 6. BONUS: KŘIVKA INERCIE
    # ------------------------------------------------------------------
    print("\n[6] BONUS: Křivka inercie (single linkage, ASCII)...")
    try:
        single = SingleLinkage()
        k_max = len(words) - 1
        k_values, inertias = inertia_curve(single, d, data, k_max=k_max)
        print(f"    Inertia pro k=1..{k_max}: {[round(v, 1) for v in inertias]}")
        elbow = find_elbow(k_values, inertias)
        plot_inertia_curve(
            k_values, inertias, elbow=elbow,
            save_path=os.path.join(GRAPHS_DIR, "inertia_curve.png"),
        )
        print("    Tip: Hledejte 'loket' křivky — místo, kde pokles inercie zpomalí.")
        print("    Očekávaný loket: k ≈ 4")
        print(f"    Detekovaný loket: k = {elbow}")
    except NotImplementedError as e:
        print(f"    [INFO] {e}")

    print("\n" + "=" * 60)
    print(" Hotovo!")
    print("=" * 60)


if __name__ == "__main__":
    main()
