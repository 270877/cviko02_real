# -*- coding: utf-8 -*-
"""
Balíček dataio — načítání dat, vektorizace a vizualizace.

Proč tento soubor existuje:
    Prázdný __init__.py označuje složku dataio/ jako Python balíček.
    Bez něj by import 'from dataio.loader import load_words' selhal
    s chybou ModuleNotFoundError.

Co tu je navíc:
    Re-exporty níže umožňují kratší zápis:
        from dataio import load_words          ← díky __init__.py
    namísto:
        from dataio.loader import load_words   ← přímý import modulu
"""

from dataio.loader import load_words
from dataio.vectorizer import ascii_vectorize, keyboard_vectorize
from dataio.plotting import plot_dendrogram, plot_inertia_curve

# __all__ říká Pythonu, co je veřejné API tohoto balíčku:
#   - potlačuje varování linteru "not accessed" u re-exportů
#   - řídí, co se importuje při  from dataio import *
__all__ = [
    "load_words",
    "ascii_vectorize", "keyboard_vectorize",
    "plot_dendrogram", "plot_inertia_curve",
]
