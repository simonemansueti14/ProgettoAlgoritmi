#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compito 3 ? Cammini minimi con CAMMINOMIN
Algoritmi e Strutture Dati (a.a. 2024/25)

- Input: griglia dal Compito 1 (CSV)
- Algoritmo: CAMMINOMIN (ricorsivo, come da traccia)
- Output: 
  * sequenza dei landmark di un cammino minimo (se esiste)
  * lunghezza del cammino minimo
  * (per gruppi da 3) sequenza completa delle celle del cammino
"""

import argparse, csv, math, json, os
from typing import List, Tuple, Set, Dict
from pathlib import Path
from grid_generator import Grid
from grid_analysis import dlib, compute_context_and_complement

Cell = Tuple[int, int]

# ---------------------------------- FRONTIERA ----------------------------------
#metodo per determinare la frontiera di una cella di origine O
#g: Grid è la griglia, context sono le celle del contesto di O, complement del complemento, e ritorna
#una lista di coppie ((r,c), tipo) dove tipo vale 1 se la cella era nel contesto e 2 se era nel complemento
def compute_frontier(g: Grid, context: Set[Cell], complement: Set[Cell]) -> List[Tuple[Cell,int]]:
    """
    Determina la frontiera di O (celle al confine con l'esterno).
    Restituisce una lista di coppie (F, tipo) dove tipo=1 se appartiene al contesto, 2 se al complemento.
    """
    frontier: List[Tuple[Cell,int]] = [] #lista vuota che verrà riempita con le celle di frontiera trovate
    closure = context.union(complement) #unione di contesto e complemento

    #per ogni cella (r,c) della chiusura, guarda tutti i suoi 8 vicini
    for (r,c) in closure:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]: #spostamento su giù ecc
            nr, nc = r+dr, c+dc #coordinate della cella vicina
            #se la cella vicina è dentro i confini della grilia, se è libera, ma non fa parte della chiusura
            if g.in_bounds(nr,nc) and g.is_free(nr,nc) and (nr,nc) not in closure: 
                if (r,c) in context: #in quel caso è sul bordo quindi appartiene alla frontiera, e andrò a vedere se è tipo 1 o 2
                    frontier.append(((r,c),1)) 
                else:
                    frontier.append(((r,c),2))
                break
    return frontier



# ---------------------------------- CAMMINOMIN ricorsivo ----------------------------------
def cammino_minimo(g: Grid, O: Cell, D: Cell, blocked: Set[Cell]=None) -> Tuple[float,List[Tuple[Cell,int]]]:
    """
    Algoritmo CAMMINOMIN:
    - Ritorna una tupla (lunghezza, sequenza_landmark)
    - blocked = insieme di celle da trattare come ostacoli (per ricorsione)
    """
    if blocked is None: #alla prima chiamata blocked è none, viene inizializzato come insieme vuoto
        blocked = set() #poi ad ogni passo si aggiungeranno nuove celle bloccate

    #se O o D non sono libere, cammino impossibile, ritorna infinito e sequenza vuota
    if not g.is_free(*O) or not g.is_free(*D):
        return math.inf, []

    #calcolo contesto e complemento di O, ignorando celle "blocked"
    context, complement = compute_context_and_complement(g, O)
    context = {c for c in context if c not in blocked}
    complement = {c for c in complement if c not in blocked}
    closure = context.union(complement) #chiusura di O

    #caso base: se la destinazione D è già raggiungibile con un cammino libero diretto da O 
    #(perché  sta nel contesto del complemento), allora il cammino minimo è proprio quello
    if D in closure:
        t = 1 if D in context else 2
        return dlib(O,D), [(O,0),(D,t)] #restituisce la distanza libera e la sequenza di landmark

    #altrimenti calcolo la frontiera
    frontier = compute_frontier(g, context, complement)

    lunghezzaMin = math.inf #inizializza la migliore lunghezza a infinito e la miglior sequenza vuota
    seqMin: List[Tuple[Cell,int]] = []

    for F,t in frontier:
        #per ogni cella calcola distanza libera  da O a F (cammino libero)
        lF = dlib(O,F)
        # richiama camminomin ricorsivamente da F a D, ma con le celle della chiusura attuale aggiunte a blocked cosi non le ricalcola all'infinito
        lFD, seqFD = cammino_minimo(g, F, D, blocked.union(closure))
        if lFD == math.inf: #se infinito salta
            continue
        lTot = lF + lFD #somma le lunghezze
        if lTot < lunghezzaMin: #se è la migliore trovata aggiunge e costruisce la sequenza
            lunghezzaMin = lTot
            #compatta, unisce la sequenza di O?F con quella di F?D
            seqMin = [(O,0),(F,t)] + seqFD[1:]

    return lunghezzaMin, seqMin



# ---------------------------------- RICOSTRUZIONE CAMMINO ----------------------------------
#punto per gruppi da 3
#data una sequenza di landmark ricostruisce tutte le celle intermedie del cammino
#prima ci si muove in diagonale finché è possibile, poi con mosse orizzontali/verticali
def build_path_from_landmarks(g: Grid, seq: List[Tuple[Cell,int]]) -> List[Cell]:
    """
    Costruisce la sequenza completa delle celle a partire dai landmark.
    NB: qui semplifichiamo: fra due landmark usiamo mosse step-by-step
    seguendo prima la diagonale poi l'asse, come da definizione di cammino libero.
    """
    path: List[Cell] = [] #lista finale con tutte le celle del cammino
    #scorre la sequenza di landmark a coppie consecutive (A,B)
    for i in range(len(seq)-1):
        A,_ = seq[i] #landmark di partenza
        B,_ = seq[i+1] #landmark di arrivo
        r,c = A #estrae coordinate riga colonna di A
        rB,cB = B #estrae coordinate riga colonna di B
        dr = 1 if rB>r else -1 if rB<r else 0 #calcolo direzione in cui bisogna muoversi, dr verso rb
        dc = 1 if cB>c else -1 if cB<c else 0 #dc verso cb
        #finché entrambe le coordinate differiscono muove in diagonale, incrementa r e c insieme
        #ogni cella attraversata viene aggiunta al cammino
        while r!=rB and c!=cB:
            r+=dr; c+=dc
            path.append((r,c))
        #quando non può più muovere in diagonale completa il percorso, se resta la differenza sulle righe fa mosse verticali
        #se resta differenza sulle colonne fa mosse orizzontali
        while r!=rB:
            r+=dr
            path.append((r,c))
        while c!=cB:
            c+=dc
            path.append((r,c))
    return path



# ---------------------------------- VALIDAZIONE CAMMINO ----------------------------------
#verifica se la sequeza di celle path calcolata dal cammino passa solo per celle libere (.) e non da ostacoli (#)
def validate_path(g: Grid, path: List[Cell]) -> bool:
    """
    Controlla che tutte le celle del cammino siano libere.
    Ritorna True se il cammino è valido, False se passa sopra a un ostacolo.
    """
    #per ogni cella del cammino controlla che sia dentro i confini della griglia e che sia libera
    for (r, c) in path:
        if not g.in_bounds(r, c) or not g.is_free(r, c): #se fallisce stampa messaggio e ritorna false
            print(f"[ERRORE] Cammino invalido: cella {(r,c)} non è libera")
            return False
    return True


# ---------------------------------- CARICAMENTO GRIGLIA ----------------------------------
def load_grid_from_csv(path: Path) -> Grid:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        cells = [[int(x) for x in row] for row in reader]
    g = Grid(len(cells), len(cells[0]))
    g.cells = cells
    return g



# ---------------------------------- MAIN ----------------------------------
def main():
    #lettura parametri da terminale
    ap = argparse.ArgumentParser(description="Compito 3: Cammino minimo con CAMMINOMIN")
    ap.add_argument("--grid", required=True, help="file CSV della griglia (generato dal Compito 1)")
    ap.add_argument("--origin", type=int, nargs=2, metavar=("R","C"), required=True, help="cella origine O (riga colonna)")
    ap.add_argument("--dest", type=int, nargs=2, metavar=("R","C"), required=True, help="cella destinazione D (riga colonna)")
    args = ap.parse_args()

    g = load_grid_from_csv(Path(args.grid))
    O = tuple(args.origin) #converte le coordinate in tuple
    D = tuple(args.dest)

    #chiama funzione ricorsiva 
    length, seq = cammino_minimo(g, O, D)
    if length == math.inf:
        print("Nessun cammino minimo trovato (celle non raggiungibili)")
        return

    print(f"Cammino minimo O={O} ? D={D}")
    print(f"Sequenza landmark = {seq}")

    # gruppo da 3: costruzione cammino completo
    #ricostruisce la sequenza completa di celle, poi verifica se il percorso passa davvero solo da 
    #celle libere, se tutto va bene stampa il cammino completo 
    full_path = build_path_from_landmarks(g, seq)
    if validate_path(g, full_path):
        print(f"Cammino completo ({len(full_path)} celle): {full_path}")
    else:
        print("Cammino non valido (passa sopra a ostacoli)")

    #salvataggio opzionale
    out_json = {
        "origin": O, "dest": D,
        "length": length,
        "landmarks": seq,
        "path": full_path
    }
    with open("cammino_output.json","w",encoding="utf-8") as f:
        json.dump(out_json,f,indent=2)
    print("\nRisultato salvato in cammino_output.json")

if __name__ == "__main__":
    main()
