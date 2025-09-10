#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compito 2 ? Analisi di griglie (contesto, complemento, distanza libera)
Algoritmi e Strutture Dati (a.a. 2024/25)

- Input: griglia salvata dal Compito 1 (CSV o JSON).
- Output: contesto e complemento di una cella O, distanza libera tra O e D.

Definizioni:
- Cammino libero = sequenza di mosse diagonali in un quadrante + mosse orizzontali/verticali.
- Contesto(O) = celle raggiungibili con cammino libero di tipo 1.
- Complemento(O) = celle raggiungibili con cammino libero di tipo 2.
- Distanza libera dlib(O,D) = sqrt(2)*?min + (?max - ?min).
"""

import argparse, json, csv, math
from typing import List, Tuple, Set
from pathlib import Path

# Importiamo la classe Grid dal file precedente
from grid_generator import Grid

Cell = Tuple[int, int]

# ----------------- distanza libera -----------------
def dlib(o: Cell, d: Cell) -> float:
    """Calcola la distanza libera tra due celle O e D (se esiste un cammino libero)."""
    dx = abs(o[1] - d[1])
    dy = abs(o[0] - d[0])
    dmin = min(dx, dy)
    dmax = max(dx, dy)
    return math.sqrt(2) * dmin + (dmax - dmin)

# ----------------- contesto e complemento -----------------
def compute_context_and_complement(g: Grid, O: Cell) -> Tuple[Set[Cell], Set[Cell]]:
    """
    Calcola contesto e complemento di O.
    Restituisce due insiemi: (contesto, complemento).
    """
    context: Set[Cell] = set()
    complement: Set[Cell] = set()

    rows, cols = g.h, g.w
    r0, c0 = O

    for r in range(rows):
        for c in range(cols):
            if not g.is_free(r, c):
                continue
            dx, dy = abs(c - c0), abs(r - r0)

            if dx == 0 and dy == 0:
                continue
            # Cammino libero tipo 1 (obliquo + orizz/vert)
            # oppure tipo 2 (inverso)
            d1 = math.sqrt(2)*min(dx,dy) + abs(dx-dy)
            # Qui distinguiamo in modo semplice: se dx>=dy -> tipo 1, se dy>dx -> tipo 2
            if dx >= dy:
                context.add((r,c))
            else:
                complement.add((r,c))

    return context, complement

# ----------------- caricamento griglia -----------------
def load_grid_from_csv(path: Path) -> Grid:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        cells = [[int(x) for x in row] for row in reader]
    g = Grid(len(cells), len(cells[0]))
    g.cells = cells
    return g

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Compito 2: Analisi griglie (contesto, complemento, distanza libera)")
    ap.add_argument("--grid", required=True, help="file CSV della griglia (generato dal Compito 1)")
    ap.add_argument("--origin", type=int, nargs=2, metavar=("R","C"), required=True, help="cella origine O (riga colonna)")
    ap.add_argument("--dest", type=int, nargs=2, metavar=("R","C"), help="cella destinazione D (opzionale)")
    args = ap.parse_args()

    g = load_grid_from_csv(Path(args.grid))
    O = tuple(args.origin)

    context, complement = compute_context_and_complement(g, O)

    print(f"Origine O = {O}")
    print(f"Contesto(O): {len(context)} celle")
    print(f"Complemento(O): {len(complement)} celle")

    if args.dest:
        D = tuple(args.dest)
        dist = dlib(O, D)
        print(f"Distanza libera dlib(O,D) = {dist:.3f}")

if __name__ == "__main__":
    main()
