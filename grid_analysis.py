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

#argparse importa la libreria standard argparse che serve per leggere gli argomenti dalla riga di comando
import argparse, json, csv, math
from typing import List, Tuple, Set  #tipi di dato usati per annotare le funzioni 
from pathlib import Path #gestisce i percorsi dei file

#importo la classe Grid dal file precedente
from grid_generator import Grid

Cell = Tuple[int, int]

# ---------------------------------- DISTANZA LIBERA ----------------------------------
#calcola la distanza libera tra due celle O origine e D destinazione
def dlib(o: Cell, d: Cell) -> float:
    dx = abs(o[1] - d[1]) #differenza in colonne
    dy = abs(o[0] - d[0]) #differenza in righe
    dmin = min(dx, dy) #prendo minore e maggiore delle due distanze
    dmax = max(dx, dy)
    return math.sqrt(2) * dmin + (dmax - dmin)



# ---------------------------------- CONTESTO E COMPLEMENTO ----------------------------------
#data una griglia g e una cella origine O, calcola il contesto e il complemento di O
def compute_context_and_complement(g: Grid, O: Cell) -> Tuple[Set[Cell], Set[Cell]]:
    context: Set[Cell] = set() #usa due insiemi per memorizzare le celle trovate
    complement: Set[Cell] = set()

    rows, cols = g.h, g.w #dimensioni della griglia
    r0, c0 = O #coordinate della cella di origine

    #scorre tutte le celle della griglia
    #se una cella (r,c) è un ostacolo la salta con continue
    for r in range(rows):
        for c in range(cols):
            if not g.is_free(r, c):
                continue
            #calcola la distanza (in righe e colonne) della cella origine O
            #se la cella è proprio O la ignora, non ha senso includerla
            dx, dy = abs(c - c0), abs(r - r0)

            if dx == 0 and dy == 0:
                continue
            #cammino libero tipo 1 (obliquo + orizz/vert)
            # oppure tipo 2 (inverso)
            d1 = math.sqrt(2)*min(dx,dy) + abs(dx-dy)
            #qui distinguiamo in modo semplice: se dx>=dy -> tipo 1, se dy>dx -> tipo 2
            #divide le celle in due categorie:
            #contesto (tipo 1) se la distanza orizzontael dx è maggiore o uguale a quella verticale dy
            #complemento (tipo 2) se invece dy è maggiore di dx
            #è un modo semplice per dire se il cammino libero è di tipo 1 o tipo 2
            if dx >= dy:
                context.add((r,c))
            else:
                complement.add((r,c))

    return context, complement



# ---------------------------------- CARICAMENTO GRIGLIA ----------------------------------
#carica una grilia salvata in formato csv
def load_grid_from_csv(path: Path) -> Grid:
    with open(path, newline="", encoding="utf-8") as f: #apre il file csv in lettura
        reader = csv.reader(f) #usa il modulo csv per leggere le righe
        cells = [[int(x) for x in row] for row in reader] #costruisce una lista di liste di interi (ogni valore "0" e "1" stringa diventa 0 e 1 intero)
    g = Grid(len(cells), len(cells[0])) #crea un nuovo oggetto Grid con le dimensioni del file letto
    g.cells = cells
    return g



# ---------------------------------- MAIN ----------------------------------
#crea un parser di argomenti con 3 comandi
def main():
    ap = argparse.ArgumentParser(description="Compito 2: Analisi griglie (contesto, complemento, distanza libera)")
    ap.add_argument("--grid", required=True, help="file CSV della griglia (generato dal Compito 1)")
    ap.add_argument("--origin", type=int, nargs=2, metavar=("R","C"), required=True, help="cella origine O (riga colonna)")
    ap.add_argument("--dest", type=int, nargs=2, metavar=("R","C"), help="cella destinazione D (opzionale)")
    args = ap.parse_args() #legge i valori dati dal terminale

    g = load_grid_from_csv(Path(args.grid)) #carica la griglia dal csv
    O = tuple(args.origin) #converte le due coordinate (--origin) in una tupla

    #calcola contesto e complemento di O e stampa a video
    context, complement = compute_context_and_complement(g, O)

    print(f"Origine O = {O}")
    print(f"Contesto(O): {len(context)} celle")
    print(f"Complemento(O): {len(complement)} celle")

    #se l'utente ha fornito anche --dest allora legge le coordinate di D, calcola la distanza libera e la stampa con 3 cifre decimali
    if args.dest:
        D = tuple(args.dest)
        dist = dlib(O, D)
        print(f"Distanza libera dlib(O,D) = {dist:.3f}")

if __name__ == "__main__":
    main()
