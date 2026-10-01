# -*- coding: utf-8 -*-
"""Načítání dat ze souborů CSV."""

import csv


def load_words(filepath: str) -> list[str]:
    """Načte CSV se slovy, přeskočí hlavičku, vrátí plochý seznam stringů."""
    words = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # přeskočení hlavičky
        for row in reader:
            words.extend(row)
    ######################################################
    # assert Doplňte podmínku: Soubor musí obsahovat alespoň tři slova
    # assert Doplňte podmínku: Všechna slova musí mít stejnou délku
    ######################################################
    return words
