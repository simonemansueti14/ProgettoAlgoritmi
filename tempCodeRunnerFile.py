#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Compito 2 - Analisi di griglie (contesto, complemento, distanza libera)
Algoritmi e Strutture Dati (a.a. 2024/25)

- Input: griglia salvata dal Compito 1 (CSV o JSON).
- Output: contesto e complemento di una cella O, distanza libera tra O e D.

Definizioni:
- Cammino libero = sequenza di mosse diagonali in un quadrante + mosse orizzontali/verticali.
- Contesto(O) = celle raggiungibili con cammino libero di tipo 1.
- Complemento(O) = celle raggiungibili con cammino libero di tipo 2.
- Distanza libera dlib(O,D) = \sqrt2 * \Delta_{min} + \Delta_{max} - \Delta_{min} 
"""

#argparse importa la libreria standard argparse che serve per leggere gli argomenti dalla riga di comando
import argparse, csv, math
from typing import Tuple, Set
from pathlib import Path
from grid_generator import Grid

Cell = Tuple[int, int]

# ---------------------------------- DISTANZA LIBERA ----------------------------------
#calcola la distanza libera tra due celle O origine e D destinazione
def dlib(o: Cell, d: Cell) -> float:
    dx = abs(o[1] - d[1])
    dy = abs(o[0] - d[0])
    dmin = min(dx, dy)
    dmax = max(dx, dy)
    return math.sqrt(2) * dmin + (dmax - dmin)



# ---------------------------------- CONTESTO E COMPLEMENTO ----------------------------------
#data una griglia g e una cella origine O, calcola il contesto e il complemento di O
def compute_context_and_complement(g: Grid, O: Cell) -> Tuple[Set[Cell], Set[Cell]]:
    context: Set[Cell] = set()
    complement: Set[Cell] = set()

    rows, cols = g.h, g.w
    r0, c0 = O
    
    if g.in_bounds(r0, c0) is False:
        raise ValueError(f"Cella origine O={O} fuori dai limiti della griglia {rows}x{cols}")

    #scorre tutte le celle della griglia
    #se una cella (r,c) è un ostacolo la salta con continue
    for r in range(rows):
        for c in range(cols):
            if not g.is_free(r, c):
                continue
            dx, dy = abs(c - c0), abs(r - r0)

            if dx == 0 and dy == 0:
                continue

            d1 = math.sqrt(2)*min(dx,dy) + abs(dx-dy)
            if dx >= dy:
                context.add((r,c))
            else:
                complement.add((r,c))

    return context, complement



# ---------------------------------- CARICAMENTO GRIGLIA ----------------------------------
#carica una grilia salvata in formato csv
def load_grid_from_csv(path: Path) -> Grid:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        cells = [[int(x) for x in row] for row in reader]
    g = Grid(len(cells), len(cells[0]))
    g.cells = cells
    return g



# ---------------------------------- MAIN ----------------------------------
#crea un parser di argomenti con 3 comandi
def main():
    ap = argparse.ArgumentParser(description="Compito 2: Analisi griglie (contesto, complemento, distanza libera)")
    ap.add_argument("--grid", required=True, help="file CSV della griglia (generato dal Compito 1)")
    ap.add_argument("--origin", type=int, nargs=2, metavar=("R","C"), required=True, help="cella origine O (riga colonna)")
    ap.add_argument("--dest", type=int, nargs=2, metavar=("R","C"), default=None, help="cella destinazione D (opzionale)")
    args = ap.parse_args()

    g = load_grid_from_csv(Path(args.grid))
    O = tuple(args.origin)
    if args.dest is not None:
        D = tuple(args.dest)

    #calcola contesto e complemento di O e stampa a video
    context, complement = compute_context_and_complement(g, O)

    print(f"Origine O = {O}")
    if args.dest is not None:
        print(f"Destinazione D = {D}")
    print(f"Contesto(O): {len(context)} celle")
    print(f"Complemento(O): {len(complement)} celle")

    #se l'utente ha fornito anche --dest allora legge le coordinate di D, calcola la distanza libera e la stampa con 3 cifre decimali
    if args.dest:
        D = tuple(args.dest)
        dist = dlib(O, D)
        print(f"Distanza libera dlib(O,D) = {dist:.3f}")

if __name__ == "__main__":
    main()
