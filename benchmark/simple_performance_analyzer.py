#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Analyzer - Versione Semplificata
Analizza prestazioni spaziali e temporali generando:
- 1 file CSV con tutte le metriche
- 1 file JSON con i dati strutturati
- 1-2 grafici PNG di confronto

Uso:
    python simple_performance_analyzer.py
"""

import os
import sys
import csv
import math
import time
import json
import tracemalloc
import gc
from pathlib import Path
from typing import Tuple, List, Dict, Set
from datetime import datetime

# Aggiungi la directory parent al path per importare i moduli
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from _1_grid_generator import Grid
    from _2_grid_analysis import dlib, compute_context_and_complement
    from _3_grid_pathfinder import compute_frontier
except ImportError as e:
    print(f"ERRORE: Impossibile importare i moduli richiesti: {e}")
    print("   Assicurati che _1_grid_generator.py, _2_grid_analysis.py e _3_grid_pathfinder.py")
    print("   siano nella directory parent di questo script.")
    sys.exit(1)

# Matplotlib per i grafici
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    print("⚠️  AVVISO: matplotlib non installato. I grafici non verranno generati.")
    print("   Installa con: pip install matplotlib")
    HAS_MATPLOTLIB = False

Cell = Tuple[int, int]

# ================================= CARICAMENTO GRIGLIA =================================

def load_grid_from_csv(path: Path) -> Grid:
    """Carica griglia da CSV"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        cells = [[int(x) for x in row] for row in reader]
    g = Grid(len(cells), len(cells[0]))
    g.cells = cells
    return g

def choose_origin_and_dest(g: Grid) -> Tuple[Cell, Cell]:
    """Trova origine e destinazione ottimali"""
    h, w = g.h, g.w
    origin = [0, 0]
    dest = [h - 1, w - 1]
    
    max_iters = h * w * 2
    iters = 0
    
    while (not g.is_free(origin[0], origin[1])) or (not g.is_free(dest[0], dest[1])):
        if not g.is_free(origin[0], origin[1]):
            origin[1] += 1
            if origin[1] >= w:
                origin[1] = 0
                origin[0] = (origin[0] + 1) % h
        
        if not g.is_free(dest[0], dest[1]):
            dest[1] -= 1
            if dest[1] < 0:
                dest[1] = w - 1
                dest[0] = (dest[0] - 1) % h
        
        iters += 1
        if iters > max_iters:
            raise RuntimeError("Impossibile trovare celle libere")
    
    return tuple(origin), tuple(dest)

# ================================= ALGORITMO CON PROFILING =================================

def cammino_minimo_profiled(
    g: Grid,
    O: Cell,
    D: Cell,
    variant: int = 0,
    deadline: float = None,
    blocked: Set[Cell] = None,
    stats: Dict[str, int] = None,
    best = None
):
    """Versione algoritmo con profiling"""
    
    if blocked is None:
        blocked = set()
    if stats is None:
        stats = {"frontier_count": 0, "tipo1_count": 0, "tipo2_count": 0, "recursions": 0}
    if best is None:
        best = (math.inf, [])
    
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
        return dlib(O, D), [(O, 0), (D, t)], stats, True
    
    frontier = compute_frontier(g, context, complement, O)
    stats["frontier_count"] += len(frontier)
    
    if not frontier:
        return math.inf, [], stats, True
    
    lunghezzaMin, seqMin, completed = math.inf, [], True
    
    for F, t in frontier:
        if deadline and time.perf_counter() > deadline:
            return best[0], best[1], stats, False
        
        stats[f"tipo{t}_count"] += 1
        lF = dlib(O, F)
        
        if variant == 0:
            pruning_threshold = lF
        else:
            pruning_threshold = lF + dlib(F, D)
        
        if pruning_threshold >= best[0]:
            continue
        
        stats["recursions"] += 1
        lFD, seqFD, stats, sub_completed = cammino_minimo_profiled(
            g, F, D, variant, deadline, blocked.union(closure), stats, best
        )
        
        if lFD != math.inf and lF + lFD < best[0]:
            best = (lF + lFD, [(O, 0), (F, t)] + seqFD[1:])
        
        if not sub_completed:
            return best[0], best[1], stats, False
        
        if lFD == math.inf:
            continue
        
        lTot = lF + lFD
        if lTot < lunghezzaMin:
            lunghezzaMin = lTot
            seqMin = [(O, 0), (F, t)] + seqFD[1:]
            best = (lunghezzaMin, seqMin)
    
    return lunghezzaMin, seqMin, stats, completed

# ================================= MISURAZIONE PRESTAZIONI =================================

def measure_performance(g: Grid, O: Cell, D: Cell, variant: int, deadline: float, trials: int = 3):
    """Esegue più trial e calcola statistiche"""
    
    print(f"\n  Variante {variant}:", end=" ")
    
    all_metrics = []
    
    for trial in range(1, trials + 1):
        # Garbage collection
        gc.collect()
        
        # Avvia tracciamento memoria
        tracemalloc.start()
        
        # Esecuzione
        start = time.perf_counter()
        deadline_abs = start + deadline
        
        length, path, stats, completed = cammino_minimo_profiled(
            g, O, D, variant, deadline_abs
        )
        
        elapsed = time.perf_counter() - start
        
        # Memoria
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        all_metrics.append({
            "trial": trial,
            "time": elapsed,
            "memory_mb": peak / 1024 / 1024,
            "recursions": stats["recursions"],
            "frontiers": stats["frontier_count"],
            "tipo1": stats["tipo1_count"],
            "tipo2": stats["tipo2_count"],
            "length": length if length != math.inf else -1,
            "completed": completed
        })
        
        print(f"T{trial}={elapsed:.3f}s", end=" ")
    
    # Calcola medie
    times = [m["time"] for m in all_metrics]
    memories = [m["memory_mb"] for m in all_metrics]
    recursions = [m["recursions"] for m in all_metrics]
    
    avg_metrics = {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "avg_memory_mb": sum(memories) / len(memories),
        "min_memory_mb": min(memories),
        "max_memory_mb": max(memories),
        "avg_recursions": sum(recursions) / len(recursions),
        "all_trials": all_metrics
    }
    
    print(f"→ Media: {avg_metrics['avg_time']:.3f}s, {avg_metrics['avg_memory_mb']:.2f}MB")
    
    return avg_metrics

# ================================= ELABORAZIONE RISULTATI =================================

def analyze_grid(grid_path: Path, trials: int, deadline: float):
    """Analizza una singola griglia con entrambe le varianti"""
    
    grid_name = grid_path.stem
    print(f"\n{'='*60}")
    print(f"Griglia: {grid_name}")
    print(f"{'='*60}")
    
    g = load_grid_from_csv(grid_path)
    print(f"Dimensione: {g.h}×{g.w}")
    
    O, D = choose_origin_and_dest(g)
    print(f"Origine: {O}, Destinazione: {D}")
    
    results = {
        "grid_name": grid_name,
        "grid_size": g.h,
        "origin": O,
        "destination": D,
        "variants": {}
    }
    
    # Variante 0
    print(f"\nTest Variante 0 (pruning BASE):")
    v0_otod = measure_performance(g, O, D, 0, deadline, trials)
    v0_dtoo = measure_performance(g, D, O, 0, deadline, trials)
    
    results["variants"]["variant_0"] = {
        "OtoD": v0_otod,
        "DtoO": v0_dtoo
    }
    
    # Variante 1
    print(f"\nTest Variante 1 (pruning FORTE):")
    v1_otod = measure_performance(g, O, D, 1, deadline, trials)
    v1_dtoo = measure_performance(g, D, O, 1, deadline, trials)
    
    results["variants"]["variant_1"] = {
        "OtoD": v1_otod,
        "DtoO": v1_dtoo
    }
    
    return results

# ================================= GENERAZIONE OUTPUT =================================

def save_to_csv(all_results: List[Dict], output_path: Path):
    """Salva tutti i risultati in CSV"""
    
    csv_path = output_path / "performance_results.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Griglia", "Dimensione", "Variante", "Direzione",
            "Tempo_Medio(s)", "Tempo_Min(s)", "Tempo_Max(s)",
            "Memoria_Media(MB)", "Memoria_Min(MB)", "Memoria_Max(MB)",
            "Ricorsioni_Medie"
        ])
        
        # Dati
        for result in all_results:
            grid_name = result["grid_name"]
            size = result["grid_size"]
            
            for variant_name, variant_data in result["variants"].items():
                var_num = variant_name.split('_')[1]
                
                for direction, metrics in variant_data.items():
                    arrow = "O→D" if direction == "OtoD" else "D→O"
                    
                    writer.writerow([
                        grid_name,
                        size,
                        var_num,
                        arrow,
                        f"{metrics['avg_time']:.6f}",
                        f"{metrics['min_time']:.6f}",
                        f"{metrics['max_time']:.6f}",
                        f"{metrics['avg_memory_mb']:.4f}",
                        f"{metrics['min_memory_mb']:.4f}",
                        f"{metrics['max_memory_mb']:.4f}",
                        f"{metrics['avg_recursions']:.2f}"
                    ])
    
    print(f"\nCSV salvato: {csv_path.name}")
    return csv_path

def save_to_json(all_results: List[Dict], output_path: Path, metadata: Dict):
    """Salva tutti i risultati in JSON"""
    
    json_path = output_path / "performance_results.json"
    
    output = {
        "metadata": metadata,
        "results": all_results
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"JSON salvato: {json_path.name}")
    return json_path

def generate_plots(all_results: List[Dict], output_path: Path):
    """Genera grafici di confronto"""
    
    if not HAS_MATPLOTLIB:
        print("Matplotlib non disponibile, grafici non generati")
        return []
    
    # Estrai dati
    grids = []
    v0_time_otod, v0_time_dtoo = [], []
    v1_time_otod, v1_time_dtoo = [], []
    v0_mem_otod, v0_mem_dtoo = [], []
    v1_mem_otod, v1_mem_dtoo = [], []
    
    for result in all_results:
        grids.append(result["grid_name"])
        
        v0 = result["variants"]["variant_0"]
        v1 = result["variants"]["variant_1"]
        
        v0_time_otod.append(v0["OtoD"]["avg_time"])
        v0_time_dtoo.append(v0["DtoO"]["avg_time"])
        v1_time_otod.append(v1["OtoD"]["avg_time"])
        v1_time_dtoo.append(v1["DtoO"]["avg_time"])
        
        v0_mem_otod.append(v0["OtoD"]["avg_memory_mb"])
        v0_mem_dtoo.append(v0["DtoO"]["avg_memory_mb"])
        v1_mem_otod.append(v1["OtoD"]["avg_memory_mb"])
        v1_mem_dtoo.append(v1["DtoO"]["avg_memory_mb"])
    
    saved_plots = []
    
    # ===== GRAFICO 1: Confronto Tempi =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(grids))
    width = 0.35
    
    # Variante 0
    ax1.bar(x - width/2, v0_time_otod, width, label='O→D', alpha=0.8)
    ax1.bar(x + width/2, v0_time_dtoo, width, label='D→O', alpha=0.8)
    ax1.set_xlabel('Griglia', fontsize=11)
    ax1.set_ylabel('Tempo (s)', fontsize=11)
    ax1.set_title('Variante 0 - Tempo di Esecuzione', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(grids, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Variante 1
    ax2.bar(x - width/2, v1_time_otod, width, label='O→D', alpha=0.8, color='orange')
    ax2.bar(x + width/2, v1_time_dtoo, width, label='D→O', alpha=0.8, color='red')
    ax2.set_xlabel('Griglia', fontsize=11)
    ax2.set_ylabel('Tempo (s)', fontsize=11)
    ax2.set_title('Variante 1 - Tempo di Esecuzione', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(grids, rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    time_plot = output_path / "comparison_time.png"
    plt.savefig(time_plot, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(time_plot)
    print(f"Grafico tempi: {time_plot.name}")
    
    # ===== GRAFICO 2: Confronto Memoria =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Variante 0
    ax1.bar(x - width/2, v0_mem_otod, width, label='O→D', alpha=0.8, color='green')
    ax1.bar(x + width/2, v0_mem_dtoo, width, label='D→O', alpha=0.8, color='lightgreen')
    ax1.set_xlabel('Griglia', fontsize=11)
    ax1.set_ylabel('Memoria (MB)', fontsize=11)
    ax1.set_title('Variante 0 - Utilizzo Memoria', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(grids, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Variante 1
    ax2.bar(x - width/2, v1_mem_otod, width, label='O→D', alpha=0.8, color='purple')
    ax2.bar(x + width/2, v1_mem_dtoo, width, label='D→O', alpha=0.8, color='violet')
    ax2.set_xlabel('Griglia', fontsize=11)
    ax2.set_ylabel('Memoria (MB)', fontsize=11)
    ax2.set_title('Variante 1 - Utilizzo Memoria', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(grids, rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    mem_plot = output_path / "comparison_memory.png"
    plt.savefig(mem_plot, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(mem_plot)
    print(f"Grafico memoria: {mem_plot.name}")
    
    return saved_plots

# ================================= MAIN =================================

def main():
    print("\n" + "="*70)
    print("SIMPLE PERFORMANCE ANALYZER")
    print("   Analisi Prestazioni Spaziali e Temporali")
    print("="*70 + "\n")
    
    # ===== INPUT =====
    print("Inserisci il percorso della cartella con le griglie CSV:")
    print("   (es: ../input_es4/experimental_grids_es4_20250128/20x20)")
    print(f"   Directory corrente: {Path.cwd()}")
    #grid_dir = input("   Percorso: ").strip()
    grid_dir = "./50x50"
    
    if not grid_dir:
        print("❌ Percorso non valido!")
        return
    
    # Prova sia percorso assoluto che relativo
    grid_path = Path(grid_dir)
    
    # Se è relativo, prova anche dalla parent directory dello script
    if not grid_path.exists() and not grid_path.is_absolute():
        # Prova dalla directory dello script
        script_dir = Path(__file__).parent
        grid_path_alt1 = script_dir / grid_dir
        grid_path_alt2 = script_dir.parent / grid_dir
        
        if grid_path_alt1.exists():
            grid_path = grid_path_alt1
            print(f"✓ Trovato in: {grid_path_alt1}")
        elif grid_path_alt2.exists():
            grid_path = grid_path_alt2
            print(f"✓ Trovato in: {grid_path_alt2}")
    
    if not grid_path.exists():
        print(f"\n❌ Cartella non trovata: {grid_path}")
        print(f"\nPercorsi provati:")
        print(f"   1. {Path(grid_dir).absolute()}")
        if not Path(grid_dir).is_absolute():
            print(f"   2. {(Path(__file__).parent / grid_dir).absolute()}")
            print(f"   3. {(Path(__file__).parent.parent / grid_dir).absolute()}")
        print(f"\n Suggerimento: usa un percorso assoluto o verifica la cartella")
        return
    
    grid_path = grid_path.absolute()
    
    # Trials
    trials = 3
    trials_input = input(f"\n Numero di trial per test [default: {trials}]: ").strip()
    if trials_input:
        try:
            trials = int(trials_input)
        except:
            pass
    
    # Deadline
    deadline = 30.0
    deadline_input = input(f"\n Deadline per esecuzione in secondi [default: {deadline}]: ").strip()
    if deadline_input:
        try:
            deadline = float(deadline_input)
        except:
            pass
    
    print(f"\n{'='*70}")
    print(f"Configurazione:")
    print(f"   Cartella: {grid_path}")
    print(f"   Trial: {trials}")
    print(f"   Deadline: {deadline}s")
    print(f"{'='*70}")
    
    # ===== CREAZIONE OUTPUT DIRECTORY =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid = "50x50"

    # ===== ANALISI =====
    file = "agglomerates_05.csv"
    grid_files = list(grid_path.glob(file))
    
    # Se lo script è in una sottocartella, salva i risultati nella parent
    script_dir = Path(__file__).parent
    if script_dir.name.startswith("performance") or script_dir.name == "analysis":
        output_base = script_dir.parent / "performance_results"
    else:
        output_base = script_dir / "performance_results"
    
    output_dir = output_base / grid / file
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Risultati salvati in: {output_dir.relative_to(Path.cwd())}\n")
    
    if not grid_files:
        print(f"\nNessun file CSV trovato in {grid_path}")
        print(f"\nContenuto della cartella:")
        try:
            all_files = list(grid_path.iterdir())
            if not all_files:
                print("   (cartella vuota)")
            else:
                for item in sorted(all_files)[:10]:  # Mostra max 10 file
                    tipo = "📁" if item.is_dir() else "📄"
                    print(f"   {tipo} {item.name}")
                if len(all_files) > 10:
                    print(f"   ... e altri {len(all_files) - 10} elementi")
        except Exception as e:
            print(f"   Errore lettura cartella: {e}")
        return
    
    print(f"📋 Trovate {len(grid_files)} griglie da analizzare\n")
    
    all_results = []
    
    for i, grid_file in enumerate(grid_files, 1):
        print(f"\n[{i}/{len(grid_files)}] Elaborazione in corso...")
        
        try:
            result = analyze_grid(grid_file, trials, deadline)
            all_results.append(result)
        except Exception as e:
            print(f"Errore durante l'analisi di {grid_file.name}: {e}")
            continue
    
    if not all_results:
        print("\nNessun risultato disponibile!")
        return
    
    # ===== GENERAZIONE OUTPUT =====
    print(f"\n{'='*70}")
    print("Generazione file di output...")
    print(f"{'='*70}")
    
    metadata = {
        "timestamp": timestamp,
        "generated_at": datetime.now().isoformat(),
        "grid_directory": str(grid_path),
        "trials_per_test": trials,
        "deadline_seconds": deadline,
        "total_grids": len(all_results)
    }
    
    # Salva CSV
    csv_file = save_to_csv(all_results, output_dir)
    
    # Salva JSON
    json_file = save_to_json(all_results, output_dir, metadata)
    
    # Genera grafici
    plots = generate_plots(all_results, output_dir)
    
    # ===== RIEPILOGO =====
    print(f"\n{'='*70}")
    print("ANALISI COMPLETATA!")
    print(f"{'='*70}\n")
    
    print(f"Risultati disponibili in: {output_dir.relative_to(Path.cwd())}/")
    print(f"   CSV: {csv_file.name}")
    print(f"   JSON: {json_file.name}")
    if plots:
        for plot in plots:
            print(f"   🖼️  PNG: {plot.name}")

if __name__ == "__main__":
    main()