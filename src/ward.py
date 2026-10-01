# -*- coding: utf-8 -*-
"""
Ward linkage (Wardova metoda) — aglomerativní shlukování s minimalizací rozptylu.

Ward je záměrně samostatná třída (nededí od HierarchicalClustering), protože
pracuje přímo se surovými souřadnicemi, ne s maticí vzdáleností. Tento rozdíl
je klíčová vlastnost algoritmu, ne implementační detail.
"""

import numpy as np


class WardLinkage:
    """
    Hierarchické shlukování Wardovou metodou.

    -----------------------------------------------------------------------
    Proč je Ward jiný než Single/Complete/Average?
    -----------------------------------------------------------------------
    Single, Complete a Average linkage potřebují jen matici vzdáleností D — bez
    ohledu na to, jak vzdálenosti vznikly. Ward naopak pracuje s těžišti shluků
    (centroidy), která se po každém sloučení přepočítávají. Těžiště lze spočítat
    pouze ze surových souřadnic, ne z matice vzdáleností. Proto má Ward jiný
    vstup a není možné ho jednoduše vložit do společného rozhraní.

    -----------------------------------------------------------------------
    Wardovo kritérium
    -----------------------------------------------------------------------
    Ward minimalizuje nárůst celkového vnitroshlukového součtu čtverců (WCSS)
    při každém sloučení. WCSS shluku A je definován jako:

        WCSS_A = Σ_{i ∈ A} ||x_i - c_A||²

    kde c_A = průměr souřadnic shluku A (těžiště).

    Nárůst WCSS při sloučení A a B:

        ΔSS(A, B) = WCSS_{A∪B} - WCSS_A - WCSS_B
                  = (n_A · n_B) / (n_A + n_B) · ||c_A - c_B||²

    -----------------------------------------------------------------------
    Wardova vzdálenost (výška v Z matici)
    -----------------------------------------------------------------------
    Jako výška fúze se ukládá odmocnina z dvojnásobku nárůstu WCSS:

        d_Ward(A, B) = sqrt(2 · n_A · n_B / (n_A + n_B)) · ||c_A - c_B||

    Faktor 2 zajišťuje, že při slučování dvou singletonů dostaneme přesně jejich
    Euklidovskou vzdálenost (konvence scipy). Výsledky odpovídají:
        scipy.cluster.hierarchy.linkage(data, method='ward')

    -----------------------------------------------------------------------
    Aktualizace těžiště po sloučení
    -----------------------------------------------------------------------
    Po sloučení A a B do nového shluku C se těžiště přepočítá jako vážený průměr:

        c_C = (n_A · c_A + n_B · c_B) / (n_A + n_B)

    Tím se vyhne nutnosti ukládat všechny původní souřadnice členů shluku
    — stačí si pamatovat těžiště a velikost.

    -----------------------------------------------------------------------
    Formát výstupu Z (stejný jako HierarchicalClustering)
    -----------------------------------------------------------------------
    Z má tvar (n-1, 4), každý řádek = jedno sloučení:
        [ID_A, ID_B, d_Ward, velikost_nového_shluku]

    Nový shluk v kroku step dostane ID = n + step  (stejná konvence jako scipy).
    """

    def __init__(self) -> None:
        self.z: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Úkol: Naprogramujte Wardovu aglomerativní smyčku a sestavte linkage matici Z.

        Vstup:
            data (np.ndarray): Matice surových souřadnic tvaru (n x p).
                               Každý řádek je jeden objekt, každý sloupec jeden příznak.
                               POZOR: Toto NENÍ matice vzdáleností — jsou to přímo vektory!

        Výstup:
            Z (np.ndarray): Linkage matice tvaru (n-1, 4). Formát identický s
                            HierarchicalClustering.fit() — lze předat přímo do
                            plot_dendrogram() nebo scipy.cluster.hierarchy.dendrogram().

        Postup (implementujte krok za krokem):

        1.  Inicializace:
                n = data.shape[0]
                active    = {i: [i]              for i in range(n)}   # členové shluku
                centroids = {i: data[i].copy()   for i in range(n)}   # těžiště
                sizes     = {i: 1                for i in range(n)}   # velikosti
                Z = np.zeros((n - 1, 4))

        2.  Smyčka  for step in range(n - 1):

            a) Pro každý pár aktivních shluků (id_a, id_b) spočítejte Wardovu vzdálenost:
                    nA, nB = sizes[id_a], sizes[id_b]
                    diff   = centroids[id_a] - centroids[id_b]
                    d_ward = np.sqrt(2 * nA * nB / (nA + nB)) * np.linalg.norm(diff)

            b) Najděte pár (id_a, id_b) s MINIMÁLNÍ Wardovou vzdáleností.
               Uložte ho jako:
                    id_i = min(id_a, id_b)
                    id_j = max(id_a, id_b)

            c) Zapište řádek linkage matice:
                    Z[step, 0] = id_i
                    Z[step, 1] = id_j
                    Z[step, 2] = minimální d_ward
                    Z[step, 3] = sizes[id_i] + sizes[id_j]

            d) Vytvořte nový shluk a odstraňte sloučené:
                    new_id = n + step
                    sizes[new_id]     = sizes[id_i] + sizes[id_j]
                    centroids[new_id] = (sizes[id_i] * centroids[id_i]
                                       + sizes[id_j] * centroids[id_j]) / sizes[new_id]
                    active[new_id]    = active[id_i] + active[id_j]
                    del active[id_i],    active[id_j]
                    del centroids[id_i], centroids[id_j]
                    del sizes[id_i],     sizes[id_j]

        3.  Uložte a vraťte výsledek:
                self.z = Z.astype(float)
                return self.z

        KLÍČOVÉ ROZDÍLY oproti fit() v HierarchicalClustering:
        • Vstup je data (n x p), ne matice vzdáleností D (n x n).
        • Vzdálenost se počítá z těžišť, ne z původní matice D.
        • Těžiště se aktualizuje po každém kroku — proto Ward nelze redukovat
          na jedinou metodu _cluster_distance pracující s původní maticí D.
        """
        raise NotImplementedError("Metoda 'fit' v WardLinkage ještě nebyla implementována!")

    def fit_predict(self, data: np.ndarray, k: int) -> np.ndarray:
        """
        Úkol: Spusťte Wardovo shlukování, zastavte se při k shlucích, vraťte labely.

        Vstup:
            data (np.ndarray): Matice souřadnic (n x p) — stejný vstup jako fit().
            k (int):           Požadovaný počet výsledných shluků (1 ≤ k ≤ n).

        Výstup:
            labels (np.ndarray): Pole délky n. labels[i] = číslo shluku objektu i.
                                 Čísla shluků jsou 0, 1, ..., k-1 (v libovolném pořadí).

        Postup:
            Stejná smyčka jako ve fit(), ale provedená pouze (n - k)-krát.
            Po skončení zůstane v active přesně k shluků.
            Přiřaďte každému shluku label 0, 1, ..., k-1 a vyplňte labels[i].

        Hint: Nejjednodušší implementace zkopíruje smyčku z fit() a zastaví ji
              po (n - k) krocích. Alternativně: zavolejte self.fit(data) a přečtěte
              výsledné shluky z posledních (k-1) řádků matice Z.
        """
        # assert  Doplňte podmínku: k musí být >= 1 a <= počtu objektů (data.shape[0])
        raise NotImplementedError("Metoda 'fit_predict' v WardLinkage ještě nebyla implementována!")
