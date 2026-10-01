# -*- coding: utf-8 -*-
"""Vektorizace slov pro výpočet vzdáleností a shlukování."""

import numpy as np


def ascii_vectorize(words: list[str]) -> np.ndarray:
    """
    Úkol: Převeďte každé slovo na vektor ASCII kódů jeho znaků.
    Vrací matici (n_slov x délka_slova).

    Hint: ord('a') == 97. Každý znak slova se převede na číslo pomocí ord().
    Předpokládejte, že všechna slova mají stejnou délku.

    Příklad: "abc" → [97, 98, 99]
    """
    raise NotImplementedError("Funkce 'ascii_vectorize' ještě nebyla implementována!")


def keyboard_vectorize(words: list[str]) -> np.ndarray:
    """
    Úkol: Převeďte každý znak na souřadnice (řádek, sloupec) na QWERTZ klávesnici.
    Každý znak → 2 příznaky (řádek, sloupec). Vrací matici (n_slov x 2·délka_slova).

    Rozložení kláves (pouze písmena, řádek po řádku):
        layout = "qwertzuiopasdfghjklyxcvbnm"
        Řádek 0:  q w e r t z u i o p   (indexy  0 - 9 v layout)
        Řádek 1:   a s d f g h j k l    (indexy 10 - 18 v layout)
        Řádek 2:    y x c v b n m       (indexy 19 - 25 v layout)

    Postup pro každý znak ch:
        1. idx = layout.index(ch.lower())
        2. row: 0 pokud idx < 10, 1 pokud 10 ≤ idx < 19, jinak 2
        3. col: idx          pro řádek 0
                idx - 10     pro řádek 1
                idx - 18     pro řádek 2
               (řádek 2 je fyzicky posunut ~1 klávesu doprava → col = idx - 18, ne idx - 19)
        4. Vektor slova: [row_0, col_0, row_1, col_1, ...]

    Příklad: 'a' → idx=10 → row=1, col=0  →  příznaky [1, 0]
             'q' → idx=0  → row=0, col=0  →  příznaky [0, 0]
    """
    raise NotImplementedError("Funkce 'keyboard_vectorize' ještě nebyla implementována!")
