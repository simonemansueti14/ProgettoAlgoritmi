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

import csv, math, json, time
from typing import List, Tuple, Set, Dict
from pathlib import Path
from _1_grid_generator import Grid
from datetime import datetime
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

# ---------------------------------- CAMMINOMIN ricorsivo ----------------------------------
def cammino_minimo(
    g: Grid, O: Cell, D: Cell, deadline: float = None,
    blocked: Set[Cell] = None, stats: Dict[str, int] = None, best=None
):
    if blocked is None:
        blocked = set()
    if stats is None:
        stats = {"frontier_count": 0, "tipo1_count": 0, "tipo2_count": 0, "valorefalsoriga16": 0}
    if best is None:
        best = (math.inf, [])

    #Stop se scaduto il tempo massimo
    if deadline and time.perf_counter() > deadline:
        return best[0], best[1], stats, False

    #Celle non valide
    if not g.is_free(*O) or not g.is_free(*D):
        return math.inf, [], stats, True

    #Caso base: origine e destinazione coincidono
    if O == D:
        return 0, [], stats, True

    #Calcola contesto e complemento, escludendo celle già bloccate
    context, complement = compute_context_and_complement(g, O, blocked)
    
    closure = context.union(complement)

    #Se la destinazione è già nel contesto o complemento → cammino diretto
    if D in closure:
        t = 1 if D in context else 2
        stats[f"tipo{t}_count"] += 1
        return dlib(O, D), [(O, 0), (D, t)], stats, True

    #Calcola la frontiera e aggiorna le statistiche
    frontier = compute_frontier(g, context, complement, O)
    stats["frontier_count"] += len(frontier)

    if not frontier:
        # Vicolo cieco: nessuna frontiera disponibile
        return math.inf, [], stats, True

    lunghezzaMin, seqMin, completed = math.inf, [], True

    # Ordinamento frontiera per f(n) = g(n) + h(n)

    frontier_sorted = sorted(
        frontier,
        key=lambda x: dlib(O, x[0]) + dlib(x[0], D)
    )

    #Esplora ogni cella di frontiera
    for F, t in frontier_sorted:
        #Controllo deadline ad ogni iterazione
        if deadline and time.perf_counter() > deadline:
            return best[0], best[1], stats, False

        stats[f"tipo{t}_count"] += 1
        lF = dlib(O, F)

        #Condizione euristica (riga 16 pseudocodice):
        #Se il cammino parziale + distanza stimata al target
        #è già peggiore del migliore trovato, evita la ricorsione
        if lF >= best[0]:
            break

        #Ricorsione sul sottoproblema (F → D)
        stats["valorefalsoriga16"]+=1
        lFD, seqFD, stats, sub_completed = cammino_minimo(
            g, F, D, deadline, blocked.union(closure), stats, best
        )

        #---- MODIFICA: AGGIORNA BEST ANCHE SE LA SUB.RICORSIONE NON E' COMPLETATA ---- => SODDISFA ALLA SLIDE 71 PUNTO 2
        if lFD != math.inf and lF + lFD < best[0]:
            best = (lF + lFD, [(O,0),(F,t)] + seqFD[1:])

        if not sub_completed:
            return best[0], best[1], stats, False
        if lFD == math.inf:
            continue

        #Calcolo della lunghezza totale
        lTot = lF + lFD

        if lTot < lunghezzaMin:
            lunghezzaMin = lTot
            seqMin = [(O, 0), (F, t)] + seqFD[1:]
            best = (lunghezzaMin, seqMin)

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
        """
        Tipo1: diagonali (>=0) poi rette (>=0)
        """
        r0, c0 = A
        r1, c1 = B

        dr = r1 - r0
        dc = c1 - c0

        seg = []
        r, c = r0, c0

        # ---------
        # FASE DIAGONALE
        # ---------
        diag_steps = min(abs(dr), abs(dc))
        step_r = 1 if dr > 0 else -1
        step_c = 1 if dc > 0 else -1

        for _ in range(diag_steps):
            r += step_r
            c += step_c
            if not g.in_bounds(r, c) or not g.is_free(r, c):
                return None
            seg.append((r, c))

        # aggiorno residui
        dr_res = r1 - r
        dc_res = c1 - c

        # ---------
        # FASE RETTILINEA
        # ---------
        if dr_res != 0:
            step = 1 if dr_res > 0 else -1
            for _ in range(abs(dr_res)):
                r += step
                if not g.in_bounds(r, c) or not g.is_free(r, c):
                    return None
                seg.append((r, c))

        if dc_res != 0:
            step = 1 if dc_res > 0 else -1
            for _ in range(abs(dc_res)):
                c += step
                if not g.in_bounds(r, c) or not g.is_free(r, c):
                    return None
                seg.append((r, c))

        if (r, c) != (r1, c1):
            return None

        return seg


    def find_type2_segment(A, B):
        """
        Tipo2: rette (>=1) poi diagonali (>=1)
        """
        r0, c0 = A
        r1, c1 = B

        dr = r1 - r0
        dc = c1 - c0

        seg = []
        r, c = r0, c0

        # Caso degenerato (non valido per tipo2)
        if dr == 0 and dc == 0:
            return None

        # ---------
        # FASE RETTILINEA (almeno 1 passo)
        # ---------
        if abs(dr) >= abs(dc):
            # muovo prima verticalmente
            step = 1 if dr > 0 else -1
            first_steps = abs(dr) - min(abs(dr), abs(dc))
            if first_steps == 0:
                first_steps = 1
            for _ in range(first_steps):
                r += step
                if not g.in_bounds(r, c) or not g.is_free(r, c):
                    return None
                seg.append((r, c))
        else:
            # muovo prima orizzontalmente
            step = 1 if dc > 0 else -1
            first_steps = abs(dc) - min(abs(dr), abs(dc))
            if first_steps == 0:
                first_steps = 1
            for _ in range(first_steps):
                c += step
                if not g.in_bounds(r, c) or not g.is_free(r, c):
                    return None
                seg.append((r, c))

        # ---------
        # FASE DIAGONALE (almeno 1 passo)
        # ---------
        dr_res = r1 - r
        dc_res = c1 - c

        diag_steps = min(abs(dr_res), abs(dc_res))
        if diag_steps == 0:
            return None  # tipo2 richiede almeno una diagonale

        step_r = 1 if dr_res > 0 else -1
        step_c = 1 if dc_res > 0 else -1

        for _ in range(diag_steps):
            r += step_r
            c += step_c
            if not g.in_bounds(r, c) or not g.is_free(r, c):
                return None
            seg.append((r, c))

        if (r, c) != (r1, c1):
            return None

        return seg



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
        - True: se il path è valido
        - False: altrimenti
    """
    for (r, c) in path:
        if not g.in_bounds(r, c) or not g.is_free(r, c): #se fallisce stampa messaggio e ritorna false
            print(f"[ERRORE] Cammino invalido: cella {(r,c)} non è libera")
            return False
    return True

# ---------------------------------- VALIDAZIONE INPUT ----------------------------------
def get_valid_cell(prompt: str, g: Grid) -> Tuple[int,int]:
    """
        Input sicuro con validazione.
    """
    while True:
        try:
            r = int(input(f"{prompt} - Riga (0-{g.h-1}): "))
            c = int(input(f"{prompt} - Colonna (0-{g.w-1}): "))
            
            if not g.in_bounds(r, c):
                print(f" Cella ({r},{c}) fuori dai limiti!")
                continue
            if not g.is_free(r, c):
                print(f" Cella ({r},{c}) è un ostacolo!")
                continue
            
            return (r, c)
        except ValueError:
            print(" Inserisci numeri validi!")


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
    g = load_grid_from_csv(Path(grid))
    O = get_valid_cell("Origine O ", g)
    D = get_valid_cell("Destinazione D ", g)
    checkTimeOut=input("Inserisci timeout (s) per il calcolo (opzionale): ")
    timeout=None
    if checkTimeOut!='':
        timeout=float(checkTimeOut)

    save=input("salvataggio Json? (s/n): ").strip().lower()

    if timeout==None: 
        deadline = None
    else:
        deadline = time.perf_counter() + timeout

    start = time.perf_counter()

    #chiama funzione ricorsiva con deadline
    length, seq, stats, completed = cammino_minimo(g, O, D, deadline=deadline)

    cam_comp_info = None
    d_min_info = None
    land_info = None

    if not completed:
        print("Calcolo interrotto (timeout raggiunto) risultato parziale:")
        cam_comp_info="cammino ricavato finora: "
        d_min_info="distanza minima ricavata finora: "
        land_info = "Sequenza landmark ottenuta finora: "

    elif length == math.inf:
        print("Nessun cammino minimo trovato (celle non raggiungibili)")
        #return
    else: 
        cam_comp_info="cammino completo: "
        d_min_info = "distanza minima: "
        land_info = "Sequenza landmark finale: "



    print(f"Calcolo del cammino minimo da O={O} a D={D}")

    print(f"{land_info} = {seq}")

    # gruppo da 3: costruzione cammino completo
    #ricostruisce la sequenza completa di celle, poi verifica se il percorso passa davvero solo da 
    #celle libere, se tutto va bene stampa il cammino completo 

    full_path = build_path_from_landmarks(g, seq)
    
    print(f"{cam_comp_info} ({len(full_path)} celle): {full_path}")
    print(f"{d_min_info} {length}")

    
    if not validate_path(g,full_path): 
        print("Errore! Il percorso intermedio passa sopra ad ostacoli!")

    #riepilogo statistico
    print("\n-- Riepilogo istanza --")
    print(f"Dimensioni griglia: {g.h} x {g.w}")
    print(f"Celle di frontiera considerate: {stats['frontier_count']}")
    print(f"Ricorsioni effettuate: {stats['valorefalsoriga16']}")
    print(f"Scelte tipo1: {stats['tipo1_count']}")
    print(f"Scelte tipo2: {stats['tipo2_count']}")
    print(f"Calcolo completato: {completed}")

    #salvataggio opzionale
    #print(f"Parametro per salvare cammino su file: {save}")
    elapsed = time.perf_counter() - start
    print(f"Tempo: {elapsed:.4f}s")
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

        res_dir = Path(__file__).parent / "output_es_3"
        res_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = res_dir / f"cammino_output_{timestamp}.json"

        with open(output_file,"w",encoding="utf-8") as f:
            json.dump(out_json,f,indent=2)
        print(f"\nRisultato salvato in {output_file}")

if __name__ == "__main__":
    main()
