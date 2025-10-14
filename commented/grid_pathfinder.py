#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compito 3 – Cammini minimi con CAMMINOMIN
Algoritmi e Strutture Dati (a.a. 2024/25)

- Input: griglia dal Compito 1 (CSV)
- Algoritmo: CAMMINOMIN (ricorsivo, come da traccia)
- Output: 
  * sequenza dei landmark di un cammino minimo (se esiste)
  * lunghezza del cammino minimo
  * (per gruppi da 3) sequenza completa delle celle del cammino
  * (requisito funzionale 2) possibilità di interrompere il calcolo con timeout
"""

import argparse, csv, math, json, os, time
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
def cammino_minimo(g: Grid, O: Cell, D: Cell, blocked: Set[Cell]=None, stats: Dict[str,int]=None, deadline: float=None, best: Tuple[float,List[Tuple[Cell,int]]]=None) -> Tuple[float,List[Tuple[Cell,int]],Dict[str,int],bool]:
    """
    Algoritmo CAMMINOMIN con statistiche (requisito funzionale 1) e timeout (requisito funzionale 2):
    - Ritorna una tupla (lunghezza, sequenza_landmark, stats, completed)
    - blocked = insieme di celle da trattare come ostacoli (per ricorsione)
    - stats = dizionario con i contatori:
        * frontier_count = numero totale di celle di frontiera visitate
        * tipo1_count   = quante volte è stato scelto un landmark di tipo1 (contesto)
        * tipo2_count   = quante volte è stato scelto un landmark di tipo2 (complemento)
    - deadline = istante (in secondi, da time.perf_counter) oltre il quale interrompere
    - best = miglior soluzione trovata finora (lunghezza, sequenza)
    - completed = True se il calcolo è arrivato a termine, False se interrotto
    """
    if blocked is None: 
        blocked = set()  # alla prima chiamata blocked è None, viene inizializzato come insieme vuoto
    if stats is None:
        # alla prima chiamata inizializza i contatori a zero
        stats = {"frontier_count": 0, "tipo1_count": 0, "tipo2_count": 0}
    if best is None:
        best = (math.inf, [])

    # controllo timeout: se esiste una deadline e l'orologio ha superato quel tempo → interrompo e ritorno best_so_far
    if deadline is not None and time.perf_counter() > deadline:
        return best[0], best[1], stats, False  # False = non completato

    # se O o D non sono libere, cammino impossibile → ritorna infinito, sequenza vuota e stats invariato
    if not g.is_free(*O) or not g.is_free(*D):
        return math.inf, [], stats, True

    # calcolo contesto e complemento di O, ignorando le celle già bloccate
    context, complement = compute_context_and_complement(g, O)
    context = {c for c in context if c not in blocked}
    complement = {c for c in complement if c not in blocked}
    closure = context.union(complement)  # chiusura = insieme di tutte le celle raggiungibili da O

    # caso base: se la destinazione D è già nella chiusura di O
    if D in closure:
        t = 1 if D in context else 2  # assegna il tipo in base a dove si trova D
        if t == 1:
            stats["tipo1_count"] += 1
        else:
            stats["tipo2_count"] += 1
        # ritorna la distanza libera, la sequenza di landmark e i contatori aggiornati
        return dlib(O,D), [(O,0),(D,t)], stats, True

    # altrimenti calcolo la frontiera della chiusura
    frontier = compute_frontier(g, context, complement)
    stats["frontier_count"] += len(frontier)  # aggiorno il contatore con le celle di frontiera trovate

    lunghezzaMin = math.inf  # inizializza la migliore lunghezza a infinito
    seqMin: List[Tuple[Cell,int]] = []  # inizializza la miglior sequenza vuota
    completed = True  # se non scatta timeout rimane True

    # scorro tutte le celle di frontiera
    for F,t in frontier:
        # controllo timeout ad ogni passo
        if deadline is not None and time.perf_counter() > deadline:
            return best[0], best[1], stats, False

        # aggiorno i contatori in base al tipo della cella di frontiera
        if t == 1:
            stats["tipo1_count"] += 1
        else:
            stats["tipo2_count"] += 1

        # calcolo distanza libera da O a F (cammino diretto)
        lF = dlib(O,F)
        # richiamo cammino_minimo ricorsivamente da F a D,
        # bloccando anche tutte le celle della chiusura per evitare cicli
        lFD, seqFD, stats, sub_completed = cammino_minimo(g, F, D, blocked.union(closure), stats, deadline, best)
        if not sub_completed:
            return best[0], best[1], stats, False
        if lFD == math.inf:  # se la ricorsione non trova cammino → salta
            continue
        lTot = lF + lFD  # lunghezza totale da O a F + da F a D
        if lTot < lunghezzaMin:  # se è la migliore trovata finora
            lunghezzaMin = lTot
            # costruisco la sequenza finale compattando O→F e F→D
            seqMin = [(O,0),(F,t)] + seqFD[1:]
            best = (lunghezzaMin, seqMin)

    # ritorno la miglior lunghezza, la miglior sequenza, i contatori aggiornati e completed
    return lunghezzaMin, seqMin, stats, completed


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
    ap.add_argument("--timeout", type=float, default=None, help="tempo massimo (in secondi) per il calcolo (opzionale)")  # nuovo parametro
    args = ap.parse_args()

    g = load_grid_from_csv(Path(args.grid))
    O = tuple(args.origin) #converte le coordinate in tuple
    D = tuple(args.dest)

    deadline = time.perf_counter() + args.timeout if args.timeout else None

    #chiama funzione ricorsiva con deadline
    length, seq, stats, completed = cammino_minimo(g, O, D, deadline=deadline)
    if not completed:
        print("Calcolo interrotto (timeout raggiunto) – risultato parziale:")
    elif length == math.inf:
        print("Nessun cammino minimo trovato (celle non raggiungibili)")
        return



    print(f"Cammino minimo O={O} -> D={D}")
    print(f"Sequenza landmark = {seq}")

    # gruppo da 3: costruzione cammino completo
    #ricostruisce la sequenza completa di celle, poi verifica se il percorso passa davvero solo da 
    #celle libere, se tutto va bene stampa il cammino completo 
    full_path = build_path_from_landmarks(g, seq)
    if validate_path(g, full_path):
        print(f"Cammino completo ({len(full_path)} celle): {full_path}")
    else:
        print("Cammino non valido (passa sopra a ostacoli)")

    #riepilogo statistico
    print("\n-- Riepilogo istanza --")
    print(f"Dimensioni griglia: {g.h} x {g.w}")
    print(f"Celle di frontiera esplorate: {stats['frontier_count']}")
    print(f"Scelte tipo1: {stats['tipo1_count']}")
    print(f"Scelte tipo2: {stats['tipo2_count']}")
    print(f"Calcolo completato: {completed}")

    #salvataggio opzionale
    out_json = {
        "origin": O, "dest": D,
        "length": length,
        "landmarks": seq,
        "path": full_path,
        "completed": completed,
        "summary": {
            "grid_size": (g.h, g.w),
            "frontier_count": stats["frontier_count"],
            "tipo1_count": stats["tipo1_count"],
            "tipo2_count": stats["tipo2_count"]
        }
    }
    with open("cammino_output.json","w",encoding="utf-8") as f:
        json.dump(out_json,f,indent=2)
    print("\nRisultato salvato in cammino_output.json")

if __name__ == "__main__":
    main()
