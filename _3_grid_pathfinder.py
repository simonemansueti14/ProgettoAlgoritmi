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
from _1_grid_generator import Grid
from _2_grid_analysis import dlib, compute_context_and_complement
Cell = Tuple[int, int]

# ---------------------------------- FRONTIERA ----------------------------------
#metodo per determinare la frontiera di una cella di origine O
#g: Grid è la griglia, context sono le celle del contesto di O, complement del complemento, e ritorna
#una lista di coppie ((r,c), tipo) dove tipo vale 1 se la cella era nel contesto e 2 se era nel complemento
def compute_frontier(g: Grid, context: Set[Cell], complement: Set[Cell], O: Cell) -> List[Tuple[Cell,int]]:
    """
    Determina la frontiera di O (celle al confine con l'esterno).
    Restituisce una lista di coppie (F, tipo) dove tipo=1 se appartiene al contesto, 2 se al complemento.
    """
    frontier: List[Tuple[Cell,int]] = [] #lista vuota che verrà riempita con le celle di frontiera trovate
    closure = context.union(complement) #unione di contesto e complemento

    #debug
    #print(f"Chiusura {closure}")

    #per ogni cella (r,c) della chiusura, guarda tutti i suoi 8 vicini
    for (r,c) in closure:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]: #spostamento su giù ecc
            nr, nc = r+dr, c+dc #coordinate della cella vicina
            #se la cella vicina è dentro i confini della grilia, se è libera, ma non fa parte della chiusura e non è il punto di origine
            if g.in_bounds(nr,nc) and g.is_free(nr,nc) and (nr,nc) not in closure and (nr,nc) != O:
                if (r,c) in context: #in quel caso è sul bordo quindi appartiene alla frontiera, e andrò a vedere se è tipo 1 o 2
                    frontier.append(((r,c),1)) 
                else:
                    frontier.append(((r,c),2))
                break
    return frontier

#helper
def path_cost_by_steps(path: List[Cell]) -> float:
    """Calcola il costo (usando costi 1 per ortogonale, sqrt(2) per diagonale)
       path è lista di celle in ordine (include origine e destinazione)."""
    if not path or len(path) == 1:
        return 0.0
    total = 0.0
    for i in range(len(path)-1):
        r1,c1 = path[i]
        r2,c2 = path[i+1]
        dr = abs(r2-r1)
        dc = abs(c2-c1)
        if dr == 1 and dc == 1:
            total += math.sqrt(2)
        elif (dr == 1 and dc == 0) or (dr == 0 and dc == 1):
            total += 1.0
        else:
            # passo non consentito come cella non adiacente: fallback ad infinito
            return math.inf
    return total

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
    #se O==D
    if O==D: return 0,[],stats,True

    # calcolo contesto e complemento di O, ignorando le celle già bloccate
    context, complement = compute_context_and_complement(g, O)
    context = {c for c in context if c not in blocked}
    complement = {c for c in complement if c not in blocked}
    closure = context.union(complement)  # chiusura = insieme di tutte le celle raggiungibili da O

    #debug
    #print(f"contesto di {O}: {context}")
    #print(f"complemento di {O}: {complement}")

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
    frontier = compute_frontier(g, context, complement, O)
    #debug
    #print(f"Frontiera trovata da {O}: {[f for f,_ in frontier]}")
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
def build_path_from_landmarks(g: Grid, seq: List[Tuple[Cell,int]], blocked: Set[Cell]=None) -> List[Cell]:
    """
    Ricostruisce la sequenza completa delle celle a partire dai landmark.
    Per ogni coppia consecutiva (A, typeA) -> (B, typeB) ricava un segmento VALIDO
    che sia un cammino libero del tipo corrispondente a B:
      - se typeB == 1 -> cerca un segmento di TIPO1 (diagonali (>=0) poi rette (>=0))
      - se typeB == 2 -> cerca un segmento di TIPO2 (rette (>=1) poi diagonali (>=1))
      - se typeB == 0 -> assume TIPO1 (origine)
    blocked: insieme di celle da trattare come ostacoli (opzionale)
    Ritorna la lista delle celle del path (escluse le ripetizioni tra segmenti).
    Se non è possibile ricostruire, ritorna una lista vuota.
    """
    if blocked is None:
        blocked = set()

    def free_path(r, c, dr, dc, steps):
        """Se tutti i passi sono liberi e dentro griglia ritorna la cella finale, altrimenti None."""
        for _ in range(steps):
            r += dr; c += dc
            if not g.in_bounds(r, c) or not g.is_free(r, c) or (r,c) in blocked:
                return None
        return (r, c)

    diagonals = [(-1, 1), (-1, -1), (1, -1), (1, 1)]
    horizontals = [(0, 1), (0, -1)]
    verticals = [(-1, 0), (1, 0)]

    def find_type1_segment(A, B):
        """Cerca un segmento di tipo1 da A a B. Ritorna lista di celle (escluse A, include B) o None."""
        r0, c0 = A
        r, c = B
        maxs = max(g.h, g.w)

        # provare oblique k>=0 poi rette m>=0
        for ddr, ddc in diagonals:
            # opzionale: ottimizzazione evita diagonali contrarie, ma non necessaria
            # proviamo k da 0 in su
            for k in range(0, maxs):
                start = free_path(r0, c0, ddr, ddc, k)
                if start is None:
                    break
                sr, sc = start
                if (sr, sc) == (r, c):
                    # segmento puramente diagonale (k passi)
                    # ricostruisco le celle percorse (escluse A)
                    seg = []
                    tr, tc = r0, c0
                    for _ in range(k):
                        tr += ddr; tc += ddc
                        seg.append((tr, tc))
                    return seg
                # poi rette ammesse: (0, ddc) e (ddr, 0) come nella definizione
                for dr, dc in [(0, ddc), (ddr, 0)]:
                    for m in range(1, maxs):
                        end = free_path(sr, sc, dr, dc, m)
                        if end is None:
                            break
                        if end == (r, c):
                            # ricostruisco le celle percorse: k diagonali poi m rette
                            seg = []
                            tr, tc = r0, c0
                            for _ in range(k):
                                tr += ddr; tc += ddc
                                seg.append((tr, tc))
                            for _ in range(m):
                                tr += dr; tc += dc
                                seg.append((tr, tc))
                            return seg
        # prova cammini puri rettilinei (k=0,m>=1) o puri diagonali già coperti
        for dr, dc in horizontals + verticals:
            dist = abs(r - r0) if dr != 0 else abs(c - c0)
            if dist == 0:
                continue
            end = free_path(r0, c0, dr, dc, dist)
            if end == (r, c):
                seg = []
                tr, tc = r0, c0
                for _ in range(dist):
                    tr += dr; tc += dc
                    seg.append((tr, tc))
                return seg
        return None

    def find_type2_segment(A, B):
        """Cerca un segmento di tipo2: rette (>=1) poi diagonali (>=1). Ritorna lista di celle (escluse A, include B) o None."""
        r0, c0 = A
        r, c = B
        maxs = max(g.h, g.w)

        for dr0, dc0 in horizontals + verticals:
            # primi m passi rettilinei, m>=1
            for m in range(1, maxs):
                start = free_path(r0, c0, dr0, dc0, m)
                if start is None:
                    break
                sr, sc = start
                # poi diagonali k>=1
                for ddr, ddc in diagonals:
                    for k in range(1, maxs):
                        end = free_path(sr, sc, ddr, ddc, k)
                        if end is None:
                            break
                        if end == (r, c):
                            # ricostruisco: m rettilinei poi k diagonali
                            seg = []
                            tr, tc = r0, c0
                            for _ in range(m):
                                tr += dr0; tc += dc0
                                seg.append((tr, tc))
                            for _ in range(k):
                                tr += ddr; tc += ddc
                                seg.append((tr, tc))
                            return seg
        return None

    # costruzione path iterando sui landmark
    full_path: List[Cell] = []
    if not seq:
        return full_path

    # inizio: includo l'origine
    full_path.append(seq[0][0])

    for i in range(len(seq)-1):
        A, typeA = seq[i]
        B, typeB = seq[i+1]
        # scegli il tipo da usare basandoti sul tipo del landmark B (come nel codice CAMMINOMIN)
        typ = 1 if typeB == 1 or typeB == 0 else 2

        if typ == 1:
            seg = find_type1_segment(A, B)
        else:
            seg = find_type2_segment(A, B)

        if seg is None:
            print(f"[ERRORE] Impossibile ricostruire segmento {A} -> {B} di tipo {typ}")
            return []  # fallimento: sequence inconsistente con le chiusure scelte

        # append segment cells ma evitando ripetizione della prima cella (A)
        for cell in seg:
            if cell != full_path[-1]:
                full_path.append(cell)

    return full_path



# ---------------------------------- VALIDAZIONE CAMMINO ----------------------------------

def validate_path(g: Grid, path: List[Cell]=None) -> bool:
    """
    Controlla che non ci siano ostacoli in mezzo al percorso
    """
    for (r, c) in path:
        if not g.in_bounds(r, c) or not g.is_free(r, c): #se fallisce stampa messaggio e ritorna false
            print(f"[ERRORE] Cammino invalido: cella {(r,c)} non è libera")
            return True
    return False


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
    grid = input("Inserisci il percorso del file CSV della griglia: ").strip()
    r0 = int(input("Inserisci riga origine O: "))
    c0 = int(input("Inserisci colonna origine O: "))
    r1 = int(input("Inserisci riga destinazione D: "))
    c1 = int(input("Inserisci colonna destinazione D: "))
    checkTimeOut=input("Inserisci timeout (s) per il calcolo (opzionale): ")
    timeout=None
    if checkTimeOut!='':
        timeout=float(checkTimeOut)

    save=input("salvataggio Json? (s/n): ").strip().lower()

    g = load_grid_from_csv(Path(grid))
    O = (r0,c0)
    D =(r1,c1)

    if timeout==None: 
        deadline = None
    else:
        deadline = time.perf_counter() + timeout

    #chiama funzione ricorsiva con deadline
    length, seq, stats, completed = cammino_minimo(g, O, D, deadline=deadline)
    if not completed:
        print("Calcolo interrotto (timeout raggiunto) risultato parziale:")
    elif length == math.inf:
        print("Nessun cammino minimo trovato (celle non raggiungibili)")
        #return



    print(f"Calcolo del cammino minimo da O={O} a D={D}")
    print(f"Sequenza landmark = {seq}")

    # gruppo da 3: costruzione cammino completo
    #ricostruisce la sequenza completa di celle, poi verifica se il percorso passa davvero solo da 
    #celle libere, se tutto va bene stampa il cammino completo 

    full_path = build_path_from_landmarks(g, seq)
    
    print(f"Cammino completo ({len(full_path)} celle): {full_path}")
    print(f"distanza minima: {length}")

    
    if validate_path(g,full_path): print("Errore! Il percorso intermedio passa sopra ad ostacoli!")

    #riepilogo statistico
    print("\n-- Riepilogo istanza --")
    print(f"Dimensioni griglia: {g.h} x {g.w}")
    print(f"Celle di frontiera esplorate: {stats['frontier_count']}")
    print(f"Scelte tipo1: {stats['tipo1_count']}")
    print(f"Scelte tipo2: {stats['tipo2_count']}")
    print(f"Calcolo completato: {completed}")

    #salvataggio opzionale
    #print(f"Parametro per salvare cammino su file: {save}")
    if save=="s":
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
