# -*- coding: utf-8 -*-
"""
Balíček src — hierarchické shlukování.

Proč tento soubor existuje:
    Prázdný __init__.py označuje složku src/ jako Python balíček.
    Bez něj by import 'from src.single import SingleLinkage' selhal
    s chybou ModuleNotFoundError.

Co tu je navíc:
    Re-exporty níže umožňují kratší zápis:
        from src import SingleLinkage          ← díky __init__.py
    namísto:
        from src.single import SingleLinkage   ← přímý import modulu
    Obě formy fungují; re-export je konvence pro čistší veřejné API balíčku.
"""

from src.distance import EuclideanDistance, ManhattanDistance, CosineCoeficient
from src.single import SingleLinkage
from src.complete import CompleteLinkage
from src.average import AverageLinkage
from src.base import HierarchicalClustering
from src.ward import WardLinkage
from src.inertia import inertia_curve, within_cluster_inertia, find_elbow

# __all__ říká Pythonu, co je veřejné API tohoto balíčku:
#   - potlačuje varování linteru "not accessed" u re-exportů
#   - řídí, co se importuje při  from src import *
__all__ = [
    "EuclideanDistance", "ManhattanDistance", "CosineCoeficient",
    "HierarchicalClustering",
    "SingleLinkage", "CompleteLinkage", "AverageLinkage", "WardLinkage",
    "inertia_curve", "within_cluster_inertia", "find_elbow"
]
