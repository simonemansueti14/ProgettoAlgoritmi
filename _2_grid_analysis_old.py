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
def compute_context_and_complement(g: Grid, O: Cell) -> Tuple[Set[Cell], Set[Cell]]:
 
    context: Set[Cell] = set()
    complement: Set[Cell] = set()

    rows, cols = g.h, g.w
    r0, c0 = O

    if not g.in_bounds(r0, c0):
        raise ValueError(f"Cella origine O={O} fuori dai limiti della griglia {rows}x{cols}")
    if not g.is_free(r0, c0):
        raise ValueError(f"Cella origine O={O} impostata su un ostacolo!")

    # Direzioni: (dr, dc)
    diagonals = [(-1, 1), (-1, -1), (1, -1), (1, 1)]  # NE, NW, SW, SE
    horizontals = [(0, 1), (0, -1)]                   # E, W
    verticals = [(-1, 0), (1, 0)]                     # N, S

    # funzione ausiliaria per verificare se un percorso dritto o obliquo è libero
    #Ritorna la cella finale se tutti i passi sono liberi, altrimenti None
    def free_path(r, c, dr, dc, steps) -> Tuple[int, int] | None:
        for _ in range(steps):
            r += dr
            c += dc
            if not g.in_bounds(r, c) or not g.is_free(r, c):
                return None
        return (r, c)

    # Itera su tutte le celle della griglia
    for r in range(rows):
        for c in range(cols):
            if (r, c) == O or not g.is_free(r, c):
                continue

            found_type1 = False
            found_type2 = False

            # --- TIPO 1: oblique poi rette ---
            for ddr, ddc in diagonals:
                # prova k passi diagonali
                for k in range(0, max(rows, cols)):
                    #Se free_path su k passi ritorna None => la diagonale con k passi è bloccata, non proviamo k più grandi in quella direzione (break)
                    start = free_path(r0, c0, ddr, ddc, k)
                    if start is None:
                        break
                    sr, sc = start
                    # se la cella raggiunta è già la destinazione => tipo 1 (caso particolare tutto obliquo senza rette)
                    if (sr, sc) == (r, c):
                        found_type1 = True
                        break
                    # poi prova, partendo da quella cella, le rette ammesse per quel quadrante (orizzontale o verticale allontanandomi dall'origine)
                    for dr, dc in [(0, ddc), (ddr, 0)]:
                        for m in range(1, max(rows, cols)):
                            end = free_path(sr, sc, dr, dc, m)
                            if end is None:
                                break
                            if end == (r, c):
                                found_type1 = True
                                break
                        if found_type1:
                            break
                    if found_type1:
                        break
                if found_type1:
                    break

            # includi anche i cammini puri (solo retti o solo diagonali)
            if not found_type1:
                # solo orizzontale o verticale
                for dr, dc in horizontals + verticals:
                    #uso free_path con un numero di passi pari alla distanza
                    end = free_path(r0, c0, dr, dc, max(abs(r - r0), abs(c - c0)))
                    if end == (r, c):
                        found_type1 = True
                        break
                # solo diagonale (se a sua volta già non è stato raggiungibile con oriz/vert)
                if not found_type1:
                    for dr, dc in diagonals:
                        end = free_path(r0, c0, dr, dc, abs(r - r0))
                        if end == (r, c):
                            found_type1 = True
                            break

            # --- TIPO 2: dritte poi oblique (coerenti col quadrante) ---
            for dr, dc in horizontals + verticals:
                for m in range(1, max(rows, cols)):
                    start = free_path(r0, c0, dr, dc, m)
                    if start is None:
                        break
                    sr, sc = start

                    # determina il quadrante di (sr,sc) rispetto all'origine (r0,c0)
                    possible_diagonals = []

                    if sr < r0 and sc > c0:           # Quadrante I
                        possible_diagonals = [(-1, 1)]    # NE
                    elif sr < r0 and sc < c0:         # Quadrante II
                        possible_diagonals = [(-1, -1)]   # NW
                    elif sr > r0 and sc < c0:         # Quadrante III
                        possible_diagonals = [(1, -1)]    # SW
                    elif sr > r0 and sc > c0:         # Quadrante IV
                        possible_diagonals = [(1, 1)]     # SE
                    else:
                        # Caso bordo: stessa riga o stessa colonna
                        if sr == r0:
                            # stessa riga: determinato dalla direzione orizzontale
                            if sc > c0:
                                possible_diagonals = [(-1, 1), (1, 1)]   # NE, SE
                            elif sc < c0:
                                possible_diagonals = [(-1, -1), (1, -1)] # NW, SW
                        elif sc == c0:
                            # stessa colonna: determinato dalla direzione verticale
                            if sr < r0:
                                possible_diagonals = [(-1, 1), (-1, -1)] # NE, NW
                            elif sr > r0:
                                possible_diagonals = [(1, 1), (1, -1)]   # SE, SW

                    # ora prova solo le diagonali consentite
                    for ddr, ddc in possible_diagonals:
                        # pruning: la diagonale deve effettivamente puntare verso la destinazione
                        if (r - sr) * ddr < 0 or (c - sc) * ddc < 0:
                            continue
                        for k in range(1, max(rows, cols)):
                            end = free_path(sr, sc, ddr, ddc, k)
                            if end is None:
                                break
                            if end == (r, c):
                                found_type2 = True
                                break
                        if found_type2:
                            break
                    if found_type2:
                        break
                if found_type2:
                    break

            if found_type1:
                context.add((r, c))
            elif found_type2:
                complement.add((r, c))

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
