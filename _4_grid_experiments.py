#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compito 4 - Esperimenti su CAMMINOMIN (versione full automatica)
Algoritmi e Strutture Dati (a.a. 2024/25)

- Legge automaticamente parametri da experimental_params.json
- Carica tutte le griglie in experimental_grid/
- Esegue esperimenti CAMMINOMIN e sua variante
- Confronta direzioni O→D e D→O
- Produce riepilogo a terminale + grafico + file JSON
"""

import csv, math, time, json, statistics
from pathlib import Path
from typing import Tuple, List, Dict, Set
import matplotlib.pyplot as plt

from _1_grid_generator import Grid
from _2_grid_analysis import dlib, compute_context_and_complement
from _3_grid_pathfinder import cammino_minimo, build_path_from_landmarks, compute_frontier, validate_path

Cell = Tuple[int,int]

# ---------------------------------- CARICAMENTO GRIGLIA ----------------------------------
def load_grid_from_csv(path: Path) -> Grid:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        cells = [[int(x) for x in row] for row in reader]
    g = Grid(len(cells), len(cells[0]))
    g.cells = cells
    return g

# ---------------------------------- VARIANTE CAMMINO MINIMO ----------------------------------
def cammino_minimo_variant(g: Grid, O: Cell, D: Cell, variant: int=0,
                           blocked: Set[Cell]=None, stats: Dict[str,int]=None,
                           deadline: float=None, best=None):
    if blocked is None: blocked = set()
    if stats is None: stats = {"frontier_count": 0, "tipo1_count": 0, "tipo2_count": 0}
    if best is None: best = (math.inf, [])

    if deadline and time.perf_counter() > deadline:
        return best[0], best[1], stats, False
    if not g.is_free(*O) or not g.is_free(*D):
        return math.inf, [], stats, True
    if O == D:
        return 0, [], stats, True

    context, complement = compute_context_and_complement(g, O)
    context = {c for c in context if c not in blocked}
    complement = {c for c in complement if c not in blocked}
    closure = context.union(complement)

    if D in closure:
        t = 1 if D in context else 2
        stats[f"tipo{t}_count"] += 1
        return dlib(O, D), [(O,0),(D,t)], stats, True

    frontier = compute_frontier(g, context, complement, O)
    stats["frontier_count"] += len(frontier)

    lunghezzaMin, seqMin, completed = math.inf, [], True
    for F, t in frontier:
        if deadline and time.perf_counter() > deadline:
            return best[0], best[1], stats, False
        stats[f"tipo{t}_count"] += 1
        lF = dlib(O, F)
        lFD, seqFD, stats, sub_completed = cammino_minimo(g, F, D, blocked.union(closure), stats, deadline, best)
        if not sub_completed: return best[0], best[1], stats, False
        if lFD == math.inf: continue
        lTot = lF + lFD
        toConfront = lTot if variant == 0 else lTot + dlib(F, D)
        if toConfront < lunghezzaMin:
            lunghezzaMin = lTot
            seqMin = [(O,0),(F,t)] + seqFD[1:]
            best = (lunghezzaMin, seqMin)
    return lunghezzaMin, seqMin, stats, completed

# ---------------------------------- SUPPORTO STAMPA ----------------------------------
def checkDistanzeUguali(g: Grid, O: Cell, D: Cell):
    lOD, _, _, _ = cammino_minimo(g,O,D)
    lDO, _, _, _ = cammino_minimo(g,D,O)
    return lOD == lDO, lOD, lDO

def printStatistiche(g:Grid, length:float, seq, stats:Dict[str,int], completed:bool):
    print(f"  Lunghezza: {length}")
    print(f"  Frontiere: {stats['frontier_count']} | Tipo1: {stats['tipo1_count']} | Tipo2: {stats['tipo2_count']}")
    print(f"  Completato: {completed}")
    full_path = build_path_from_landmarks(g, seq)
    if validate_path(g, full_path):
        print("⚠️ Percorso non valido (passa su ostacoli)")

# ---------------------------------- ESPERIMENTI AUTOMATICI ----------------------------------
def experiment(g: Grid, O: Cell, D: Cell, trials:int=3, variant:int=0) -> Dict:
    results = {}
    for direction,label in [((O,D),"OtoD"), ((D,O),"DtoO")]:
        O_,D_ = direction
        lengths, times = [], []
        i=1
        print(label)
        for _ in range(trials):
            start = time.perf_counter()
            length, _, _, _ = cammino_minimo_variant(g, O_, D_, variant)
            elapsed = time.perf_counter() - start
            print(f"trial #{i} - tempo: {elapsed}")
            lengths.append(length)
            times.append(elapsed)
            i+=1
        results[label] = {
            "avg_length": statistics.mean(lengths),
            "avg_time": statistics.mean(times),
            "valid": (lengths[-1] != math.inf),
            "variant": variant
        }
    return results

def run_experiments(param_map: Dict[str, Dict], grid_dir: Path, variant:int) -> Dict:
    summary = {}
    for grid_name, config in param_map.items():
        grid_path = grid_dir / grid_name
        if not grid_path.exists():
            print(f"⚠️ Griglia {grid_name} non trovata")
            continue

        g = load_grid_from_csv(grid_path)
        origin = tuple(config.get("origin", [0,0]))
        dest = tuple(config.get("dest", [1,1]))
        trials = int(config.get("trials", 3))

        #debug
        print(f"\nEseguo {grid_name}: O={origin}, D={dest}, var={variant}, trials={trials}")
        res = experiment(g, origin, dest, trials=trials, variant=variant)
        summary[grid_name] = res
    return summary

# ---------------------------------- VISUALIZZAZIONE RISULTATI ----------------------------------
def summarize_results(summary: Dict):
    print("\n=== RIEPILOGO ESPERIMENTI ===")
    for gname, res in summary.items():
        print(f"\n🧩 {gname}")
        for direction, vals in res.items():
            print(f"  {direction}: distanza min={vals['avg_length']:.2f}, tempo medio={vals['avg_time']:.3f}s, valid={vals['valid']}, variant={vals['variant']}")

def plot_results(summary: Dict, variant:int, dim:int):
    labels = list(summary.keys())
    times0, times1 = [], []
    for gname, res in summary.items():
        t0 = res["OtoD"]["avg_time"]
        t1 = res["DtoO"]["avg_time"]
        times0.append(t0)
        times1.append(t1)

    x = range(len(labels))
    plt.figure(figsize=(8,5))
    plt.bar(x, times0, width=0.4, label="OtoD")
    plt.bar([i+0.4 for i in x], times1, width=0.4, label="DtoO")
    plt.xticks([i+0.2 for i in x], labels, rotation=45, ha="right")
    plt.ylabel("Tempo medio (s)")
    plt.title(f"Confronto tempi medi OtoD vs DtoO - variante: {variant} - dimensione griglie: {dim}x{dim}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # converte math.inf in "inf" per errori JSON
def make_json_safe(obj):
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    else:
        return obj

# ---------------------------------- MAIN AUTOMATICO ----------------------------------
def main():
    base_dir = Path(__file__).parent
    grid_dir = base_dir / "experimental_grids"
    param_file = base_dir / "experimental_params.json"

    if not grid_dir.exists():
        print("❌ Cartella experimental_grids/ non trovata.")
        return
    if not param_file.exists():
        print("❌ File experimental_params.json non trovato.")
        return

    with open(param_file, encoding="utf-8") as f:
        param_map = json.load(f)

    combined_summary = {
        "variant_0": {},
        "variant_1": {}
    }

    # === SCORRI TUTTE LE SOTTOCARTELLE (dimensioni griglia) ===
    for size_folder, grids in param_map.items():
        subdir = grid_dir / size_folder
        if not subdir.exists():
            print(f"⚠️ Sottocartella {size_folder}/ mancante, salto.")
            continue

        print(f"\n=== 🔍 ESECUZIONE ESPERIMENTI SU {size_folder}/ ===")

        # === VARIANTE 0 ===
        print("\n--- VARIANTE = 0 ---")
        summary_var0 = run_experiments(grids, subdir, 0)
        summarize_results(summary_var0)
        plot_results(summary_var0, 0, size_folder[0])
        combined_summary["variant_0"][size_folder] = make_json_safe(summary_var0)

        # === VARIANTE 1 ===
        print("\n--- VARIANTE = 1 ---")
        summary_var1 = run_experiments(grids, subdir, 1)
        summarize_results(summary_var1)
        plot_results(summary_var1, 1, size_folder[0])
        combined_summary["variant_1"][size_folder] = make_json_safe(summary_var1)

    # === SCRIVE I RISULTATI FINALI ===
    output_file = base_dir / "experiments_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)

    print(f"\n💾 Risultati salvati in {output_file.name}")

if __name__ == "__main__":
    main()
