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
from _1_grid_generator import Grid

Cell = Tuple[int, int]

# ---------------------------------- DISTANZA LIBERA ----------------------------------
#calcola la distanza libera tra due celle O origine e D destinazione
def dlib(o: Cell, d: Cell, cont: Set[Cell]=None, comp: Set[Cell]=None) -> float:
    #parte della dlib che serve per questo esercizio (non so ancora a priori se esiste o no una dist libera)
    if cont is not None and comp is not None:
        if d not in cont and d not in comp: raise ValueError("Dist. libera non calcolabile - Destinazione inserita non presente nella chiusura di O")
    #parte comune
    dx = abs(o[1] - d[1])
    dy = abs(o[0] - d[0])
    dmin = min(dx, dy)
    dmax = max(dx, dy)
    return math.sqrt(2) * dmin + (dmax - dmin)



# ---------------------------------- CONTESTO E COMPLEMENTO ----------------------------------
#data una griglia g e una cella origine O, calcola il contesto e il complemento di O
def compute_context_and_complement(
    g: Grid,
    O: Cell,
    blocked: Set[Cell] = None
) -> Tuple[Set[Cell], Set[Cell]]:

    if blocked is None:
        blocked = set()

    context: Set[Cell] = set()
    complement: Set[Cell] = set()

    r0, c0 = O

    if not g.in_bounds(r0, c0) or not g.is_free(r0, c0):
        raise ValueError(f"Origine {O} non valida")

    for r in range(g.h):
        for c in range(g.w):

            if (r, c) == O:
                continue

            if not g.is_free(r, c):
                continue

            dr, dc = r - r0, c - c0

            if is_type1_reachable(g, r0, c0, r, c, dr, dc, blocked):
                context.add((r, c))

            elif is_type2_reachable(g, r0, c0, r, c, dr, dc, blocked):
                complement.add((r, c))

    return context, complement



def is_type1_reachable(
    g: Grid,
    r0: int, c0: int,
    r: int, c: int,
    dr: int, dc: int,
    blocked: Set[Cell] = None
) -> bool:
    """Tipo 1: Diagonale → Ortogonale (o solo uno dei due)"""

    if blocked is None:
        blocked = set()

    # Caso 1: Solo ortogonale
    if dr == 0:  # stessa riga
        return is_path_free(g, r0, c0, 0, sign(dc), abs(dc), blocked)

    if dc == 0:  # stessa colonna
        return is_path_free(g, r0, c0, sign(dr), 0, abs(dr), blocked)

    # Caso 2: Solo diagonale
    if abs(dr) == abs(dc):
        if is_path_free(g, r0, c0, sign(dr), sign(dc), abs(dr), blocked):
            return True

    # Caso 3: Diagonale + Ortogonale
    diag_steps = min(abs(dr), abs(dc))
    ddr, ddc = sign(dr), sign(dc)

    for k in range(1, diag_steps + 1):

        # Punto dopo k passi diagonali
        mid_r = r0 + k * ddr
        mid_c = c0 + k * ddc   # <<< corretto (era r0 prima!)

        # Verifica diagonale fino a mid
        if not is_path_free(g, r0, c0, ddr, ddc, k, blocked):
            break

        rem_dr = r - mid_r
        rem_dc = c - mid_c

        # Orizzontale coerente
        if rem_dr == 0 and rem_dc != 0:
            if sign(rem_dc) == ddc:
                if is_path_free(g, mid_r, mid_c, 0, sign(rem_dc), abs(rem_dc), blocked):
                    return True

        # Verticale coerente
        elif rem_dc == 0 and rem_dr != 0:
            if sign(rem_dr) == ddr:
                if is_path_free(g, mid_r, mid_c, sign(rem_dr), 0, abs(rem_dr), blocked):
                    return True

    return False



def is_type2_reachable(
    g: Grid,
    r0: int, c0: int,
    r: int, c: int,
    dr: int, dc: int,
    blocked: Set[Cell] = None
) -> bool:
    """Tipo 2: Ortogonale → Diagonale (coerente col quadrante)"""

    if blocked is None:
        blocked = set()

    # Non può essere solo ortogonale
    if dr == 0 or dc == 0:
        return False

    # ----- Orizzontale + diagonale -----
    for hor_steps in range(1, abs(dc) + 1):

        mid_r = r0
        mid_c = c0 + hor_steps * sign(dc)

        if not is_path_free(g, r0, c0, 0, sign(dc), hor_steps, blocked):
            break

        rem_dr = r - mid_r
        rem_dc = c - mid_c

        if abs(rem_dr) == abs(rem_dc) and abs(rem_dr) > 0:

            if sign(rem_dc) == sign(dc) and sign(rem_dr) == sign(dr):

                if is_path_free(
                    g,
                    mid_r, mid_c,
                    sign(rem_dr), sign(rem_dc),
                    abs(rem_dr),
                    blocked
                ):
                    return True

    # ----- Verticale + diagonale -----
    for ver_steps in range(1, abs(dr) + 1):

        mid_r = r0 + ver_steps * sign(dr)
        mid_c = c0

        if not is_path_free(g, r0, c0, sign(dr), 0, ver_steps, blocked):
            break

        rem_dr = r - mid_r
        rem_dc = c - mid_c

        if abs(rem_dr) == abs(rem_dc) and abs(rem_dr) > 0:

            if sign(rem_dr) == sign(dr) and sign(rem_dc) == sign(dc):

                if is_path_free(
                    g,
                    mid_r, mid_c,
                    sign(rem_dr), sign(rem_dc),
                    abs(rem_dr),
                    blocked
                ):
                    return True

    return False



def is_path_free(
    g: Grid,
    r: int, c: int,
    dr: int, dc: int,
    steps: int,
    blocked: Set[Cell] = None
) -> bool:
    """Verifica se un percorso rettilineo è libero
       considerando anche le celle dinamicamente bloccate.
    """

    if blocked is None:
        blocked = set()

    for _ in range(steps):
        r += dr
        c += dc

        if (
            not g.in_bounds(r, c)
            or not g.is_free(r, c)
            or (r, c) in blocked
        ):
            return False

    return True



def sign(x: int) -> int:
    """Ritorna il segno di x (-1, 0, 1)"""
    return (x > 0) - (x < 0)

# ---------------------------------- CARICAMENTO GRIGLIA ----------------------------------
#carica una grilia salvata in formato csv
def load_grid_from_csv(path: Path) -> Grid:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        cells = [[int(x) for x in row] for row in reader]
    g = Grid(len(cells), len(cells[0]))
    g.cells = cells
    return g

# --- stampa contesto e complemento ---
def print_cont_comp(cont : Set[Cell], comp: Set[Cell]):
    res="contesto: {"
    count=0
    for cell in cont:
        count+=1
        res+=f"({cell[0]},{cell[1]})"
        if count<len(cont)-1: res+=","
    print(res+"}")
    res="complemento: {"
    count=0
    for cell in comp:
        res+=f"({cell[0]},{cell[1]})"
        if count<len(comp)-1: res+=","
    print(res+"}")

# ---------------------------------- MAIN ----------------------------------
def main():

    print("=== Compito 2: Analisi griglie (contesto, complemento, distanza libera) ===\n")
    grid_path = input("Inserisci il percorso del file CSV della griglia: ").strip()
    g = load_grid_from_csv(Path(grid_path))
    r0 = int(input("Inserisci riga origine O: "))
    c0 = int(input("Inserisci colonna origine O: "))
    O = (r0,c0)
    calc_dest = input("Vuoi inserire una destinazione D per calcolare la distanza libera? (s/n): ").strip().lower()
    if calc_dest == "s":
        r1 = int(input("Inserisci riga destinazione D: "))
        c1 = int(input("Inserisci colonna destinazione D: "))
        D = (r1, c1)
    else:
        D = None

    #calcola contesto e complemento di O e stampa a video
    context, complement = compute_context_and_complement(g, O)

    print(f"Origine O = {O}")
    if D is not None:
        print(f"Destinazione D = {D}")
    print(f"Contesto(O): {len(context)} celle")
    print(f"Complemento(O): {len(complement)} celle")

    print_cont_comp(context,complement)

    #se l'utente ha fornito anche --dest allora legge le coordinate di D, calcola la distanza libera e la stampa con 3 cifre decimali
    if D:
        dist = dlib(O, D, context, complement)
        print(f"Distanza libera dlib(O,D) = {dist:.3f}")

if __name__ == "__main__":
    main()
