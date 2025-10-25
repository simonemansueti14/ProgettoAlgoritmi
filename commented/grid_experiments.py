#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compito 4 - Esperimenti su CAMMINOMIN
Algoritmi e Strutture Dati (a.a. 2024/25)

- Esegue CAMMINOMIN su diverse griglie generate dal Compito 1.
- Confronta O?D con D?O (correttezza).
- Misura tempi di esecuzione e numero di nodi visitati (prestazioni).
- (Gruppo da 3) consente di confrontare la condizione alternativa per tipo1/tipo2.
"""

import argparse, csv, math, time, json, statistics
from pathlib import Path
from typing import Tuple, List, Dict, Set

from grid_generator import Grid
from _2_grid_analysis import dlib, compute_context_and_complement
from _3_grid_pathfinder import cammino_minimo, build_path_from_landmarks, validate_path

Cell = Tuple[int,int]

# ---------------------------------- CARICAMENTO GRIGLIA ----------------------------------
def load_grid_from_csv(path: Path) -> Grid:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        cells = [[int(x) for x in row] for row in reader]
    g = Grid(len(cells), len(cells[0]))
    g.cells = cells
    return g



# ---------------------------------- VARIANTE compute_context_and_complement ----------------------------------
#variant se 0 indica l'utilizzo della versione standard, 1 versione alternativa
def compute_context_and_complement_variant(g: Grid, O: Cell, variant:int=0):
    """
    Variante di compute_context_and_complement per testare la condizione alternativa.
    variant=0 -> standard (dx >= dy => tipo1)
    variant=1 -> alternativa (dy >= dx => tipo1)
    """
    context: Set[Cell] = set() #crea due insiemi vuoti
    complement: Set[Cell] = set()

    rows, cols = g.h, g.w
    r0, c0 = O

    #scorre tutte le celle della griglia, se la cella non � libera la salta
    for r in range(rows):
        for c in range(cols):
            if not g.is_free(r,c):
                continue
            #dx differenza orizzontale tra cella corrente e origine, dy idem verticale
            dx, dy = abs(c-c0), abs(r-r0)
            if dx==0 and dy==0: #se differenza � 0 allora la cella corrente � O, quindi non deve essere considerata
                continue
            if variant==0: #caso standard
                if dx >= dy:
                    context.add((r,c))
                else:
                    complement.add((r,c))
            else: # caso variante
                if dy >= dx:
                    context.add((r,c))
                else:
                    complement.add((r,c))
    return context, complement



# ---------------------------------- ESPERIMENTO SINGOLO ----------------------------------
#lancia camminomin su una coppia di celle sia in direzione O->D che D->O
#trials � quante volte ripetere esperimento
#l'output � un dizionario con i risultati
def experiment(g: Grid, O: Cell, D: Cell, trials:int=1, variant:int=0) -> Dict:
    """
    Lancia CAMMINOMIN su (O,D) e (D,O), misura tempi ed esiti.
    Usa la versione standard o la variante (a seconda del flag).
    """
    results = {}

    #patch temporaneo: sostituiamo compute_context_and_complement
    #salva la funzione originale in orig_func e la sostituisce con una lambda che chiama la variante con il parametro variant
    #in questo modo quando cammino_minimo chiamer� compute_context_and_complement user� la versione scelta
    import _2_grid_analysis
    orig_func = _2_grid_analysis.compute_context_and_complement
    _2_grid_analysis.compute_context_and_complement = lambda g_,O_: compute_context_and_complement_variant(g_,O_,variant)

    #fa due esperimenti, in entrambi i versi
    #per ogni direzione inizializza le liste per le lunghezze e i tempi e ripete per trials volte
    #misura il tempo con timer,perf_counter() e salva tutto 
    for direction,label in [((O,D),"OtoD"), ((D,O),"DtoO")]:
        O_,D_ = direction
        lengths = []
        times = []
        for _ in range(trials):
            start = time.perf_counter()
            length, seq = cammino_minimo(g, O_, D_)
            elapsed = time.perf_counter() - start
            lengths.append(length)
            times.append(elapsed)
        results[label] = { #per ogni direzione salva un dizionario con tutti i parametri
            "avg_length": statistics.mean(lengths),
            "avg_time": statistics.mean(times),
            "last_seq": seq,
            "valid": (lengths[-1] != math.inf),
            "variant": variant
        }

    #ripristina la funzione originale per non lasciare modifiche permanenti
    _2_grid_analysis.compute_context_and_complement = orig_func

    return results

#lancia pi� esperimenti su una lista di griglie
#grids � la lista di percorsi a file .csv delle griglie
def run_experiments(grids: List[Path], origin: Cell, dest: Cell, trials:int=3, variant:int=0) -> Dict:
    summary = {}
    #per ogni griglia la carica con load_grid_from_csv e lancia experiment, salva poi i risultati in summary
    for gpath in grids:
        g = load_grid_from_csv(gpath)
        res = experiment(g, origin, dest, trials=trials, variant=variant)
        summary[str(gpath)] = res
    return summary





# ---------------------------------- MAIN ----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compito 4: Esperimenti su CAMMINOMIN")
    ap.add_argument("--grids", nargs="+", required=True, help="lista di file CSV di griglie da testare")
    ap.add_argument("--origin", type=int, nargs=2, metavar=("R","C"), required=True, help="origine O")
    ap.add_argument("--dest", type=int, nargs=2, metavar=("R","C"), required=True, help="destinazione D")
    ap.add_argument("--trials", type=int, default=3, help="numero di ripetizioni per ogni esperimento")
    ap.add_argument("--variant", type=int, choices=[0,1], default=0, help="0=standard, 1=condizione alternativa riga 16")
    args = ap.parse_args()

    grids = [Path(p) for p in args.grids]
    O = tuple(args.origin)
    D = tuple(args.dest)

    #lancia run_experiments per fare esperimenti su ogni griglia e raccoglie tutti i dati in un summary
    summary = run_experiments(grids, O, D, trials=args.trials, variant=args.variant)

    print(json.dumps(summary, indent=2))
    with open("experiments_output.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print("\nRisultati salvati in experiments_output.json")


if __name__ == "__main__":
    main()
