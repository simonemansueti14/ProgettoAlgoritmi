#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark CAMMINOMIN: Versione VECCHIA vs OTTIMIZZATA

Confronta le prestazioni su diverse configurazioni di griglia.
"""


import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Set, Callable
from dataclasses import dataclass
from statistics import mean
import math
import math, time
from typing import Tuple, Set
import os
import sys
import time
import random
import csv


# Trova la cartella dove si trova lo script attuale
current_dir = os.path.dirname(os.path.abspath(__file__))
# Risale di un livello
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# importing
from _1_grid_generator import Grid
from _2_grid_analysis import dlib, compute_context_and_complement


# ========================= MOCK CLASSES (sostituisci con i tuoi import) =========================
Cell = Tuple[int, int]

class Grid:
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.cells = [[0] * w for _ in range(h)]
    
    def in_bounds(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w
    
    def is_free(self, r, c):
        return self.in_bounds(r, c) and self.cells[r][c] == 0
    
    def set_obstacle(self, r, c):
        if self.in_bounds(r, c):
            self.cells[r][c] = 1
    
    def generate_random_obstacles(self, density: float = 0.2):
        for r in range(self.h):
            for c in range(self.w):
                if random.random() < density:
                    self.cells[r][c] = 1

def dlib(A: Cell, B: Cell) -> float:
    """Distanza euclidea (mock)"""
    return math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)


# ========================= FRONTIERA (comune) =========================
def compute_frontier(g: Grid, context: Set[Cell], complement: Set[Cell], O: Cell):
    frontier = []
    closure = context.union(complement)
    directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    for (r, c) in closure:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (g.in_bounds(nr, nc) and g.is_free(nr, nc) and 
                (nr, nc) not in closure and (nr, nc) != O):
                tipo = 1 if (r, c) in context else 2
                frontier.append(((r, c), tipo))
                break
    return frontier


# ========================= VERSIONE VECCHIA (NON OTTIMIZZATA) =========================
def cammino_minimo_OLD(g, O, D, deadline=None, blocked=None, stats=None, best=None):
    """VERSIONE ORIGINALE (senza ordinamento euristico)"""
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

        if lF + dlib(F, D) >= best[0]:
            continue 

        stats["recursions"] += 1
        lFD, seqFD, stats, sub_completed = cammino_minimo_OLD(
            g, F, D, deadline, blocked.union(closure), stats, best
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


# ========================= VERSIONE NUOVA (OTTIMIZZATA) =========================
def cammino_minimo_NEW(g, O, D, deadline=None, blocked=None, stats=None, best=None):
    """VERSIONE OTTIMIZZATA (con ordinamento euristico A*)"""
    if blocked is None:
        blocked = set()
    if stats is None:
        stats = {"frontier_count": 0, "tipo1_count": 0, "tipo2_count": 0, 
                 "recursions": 0, "pruned_nodes": 0}
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


    frontier_sorted = sorted(
        frontier,
        key=lambda x: dlib(O, x[0]) + dlib(x[0], D)
    )

    lunghezzaMin, seqMin, completed = math.inf, [], True

    for F, t in frontier_sorted:
        if deadline and time.perf_counter() > deadline:
            return best[0], best[1], stats, False

        stats[f"tipo{t}_count"] += 1
        lF = dlib(O, F)
        h_FD = dlib(F, D)

        if lF + h_FD >= best[0]:
            stats["pruned_nodes"] += 1
            break

        stats["recursions"] += 1
        lFD, seqFD, stats, sub_completed = cammino_minimo_NEW(
            g, F, D, deadline, blocked.union(closure), stats, best
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


# ========================= BENCHMARK =========================
@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    recursions: int
    frontier_count: int
    pruned_nodes: int
    path_length: float
    completed: bool


def benchmark_camminomin(func: Callable, g: Grid, O: Cell, D: Cell, timeout: float = 30.0) -> BenchmarkResult:
    """Esegue benchmark di una funzione CAMMINOMIN"""
    deadline = time.perf_counter() + timeout
    
    start = time.perf_counter()
    length, seq, stats, completed = func(g, O, D, deadline=deadline)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    return BenchmarkResult(
        name=func.__name__,
        time_ms=elapsed,
        recursions=stats.get("recursions", 0),
        frontier_count=stats["frontier_count"],
        pruned_nodes=stats.get("pruned_nodes", 0),
        path_length=length,
        completed=completed
    )


def save_results_to_csv(filename, configs, results_old, results_new):
    """Salva i risultati del benchmark in formato CSV"""
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "config_desc",
            "size",
            "density",
            "version",
            "time_ms",
            "recursions",
            "frontier_count",
            "pruned_nodes",
            "path_length",
            "completed"
        ])
        
        for config, res_old, res_new in zip(configs, results_old, results_new):
            # OLD
            writer.writerow([
                config["desc"],
                config["size"],
                config["density"],
                "OLD",
                f"{res_old.time_ms:.3f}",
                res_old.recursions,
                res_old.frontier_count,
                0,  # OLD non ha pruning
                f"{res_old.path_length:.3f}",
                res_old.completed
            ])
            
            # NEW
            writer.writerow([
                config["desc"],
                config["size"],
                config["density"],
                "NEW",
                f"{res_new.time_ms:.3f}",
                res_new.recursions,
                res_new.frontier_count,
                res_new.pruned_nodes,
                f"{res_new.path_length:.3f}",
                res_new.completed
            ])


def run_comprehensive_benchmark():
    """Esegue benchmark completo su diverse configurazioni"""
    print("=" * 80)
    print("BENCHMARK: CAMMINOMIN - VECCHIA vs OTTIMIZZATA")
    print("=" * 80)
    
    configs = [
        {"size": 15, "density": 0.15, "desc": "Piccola (15x15, 15%)"},
        {"size": 20, "density": 0.15, "desc": "Media (20x20, 15%)"},
        {"size": 25, "density": 0.15, "desc": "Grande (25x25, 15%)"},
        {"size": 20, "density": 0.25, "desc": "Densa (20x20, 25%)"},
        {"size": 30, "density": 0.15, "desc": "Molto Grande (30x30, 15%)"},
        {"size": 40, "density": 0.20, "desc": "Molto Grande (40x40, 20%)"},
        {"size": 50, "density": 0.30, "desc": "Molto Grande (50x50, 30%)"},
    ]
    
    results_old = []
    results_new = []
    
    for config in configs:
        size = config["size"]
        density = config["density"]
        desc = config["desc"]
        
        print(f"\n{'='*80}")
        print(f"Test: {desc}")
        print(f"{'='*80}")
        
        # Genera griglia
        g = Grid(size, size)
        g.generate_random_obstacles(density)
        
        # Trova origine e destinazione valide
        free_cells = [(r, c) for r in range(size) for c in range(size) if g.is_free(r, c)]
        if len(free_cells) < 2:
            print("Troppi ostacoli, skip")
            continue
        
        random.shuffle(free_cells)
        O, D = free_cells[0], free_cells[-1]
        
        print(f"Origine: {O}, Destinazione: {D}")
        print(f"Distanza euclidea: {dlib(O, D):.2f}")
        
        # Benchmark VECCHIA
        print("\nEsecuzione VECCHIA versione...", end=" ", flush=True)
        res_old = benchmark_camminomin(cammino_minimo_OLD, g, O, D)
        res_old_d_to_o = benchmark_camminomin(cammino_minimo_OLD, g, D, O)
        results_old.append(res_old)
        print(f"✓")
        print(f"   Tempo O-D: {res_old.time_ms:.1f} ms")
        print(f"   Tempo D-O: {res_old_d_to_o.time_ms:.1f} ms")
        print(f"   Ricorsioni: {res_old.recursions}")
        
        # Benchmark NUOVA
        print("\nEsecuzione NUOVA versione...", end=" ", flush=True)
        res_new = benchmark_camminomin(cammino_minimo_NEW, g, O, D)
        res_new_d_to_o = benchmark_camminomin(cammino_minimo_NEW, g, O, D)
        results_new.append(res_new)
        print(f"✓")
        print(f"   Tempo O-D: {res_new.time_ms:.1f} ms")
        print(f"   Tempo D-O: {res_new_d_to_o.time_ms:.1f} ms")
        print(f"   Ricorsioni: {res_new.recursions}")
        print(f"   Nodi potati: {res_new.pruned_nodes}")
        
        # Confronto
        speedup = res_old.time_ms / res_new.time_ms if res_new.time_ms > 0 else 0
        rec_reduction = (1 - res_new.recursions / res_old.recursions) * 100 if res_old.recursions > 0 else 0
        
        print(f"\nRISULTATI:")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Riduzione ricorsioni: {rec_reduction:.1f}%")
        print(f"   Efficienza pruning: {res_new.pruned_nodes} nodi evitati")
    
    # Riepilogo generale
    print("\n" + "=" * 80)
    print("RIEPILOGO GENERALE")
    print("=" * 80)
    
    if not results_old or not results_new:
        print("Nessun risultato valido")
        return
    
    times_old = [r.time_ms for r in results_old]
    times_new = [r.time_ms for r in results_new]
    recs_old = [r.recursions for r in results_old]
    recs_new = [r.recursions for r in results_new]
    
    avg_speedup = mean([old/new for old, new in zip(times_old, times_new) if new > 0])
    avg_rec_reduction = mean([(1 - new/old) * 100 for old, new in zip(recs_old, recs_new) if old > 0])
    
    print(f"\nSpeedup medio: {avg_speedup:.2f}x")
    print(f"Riduzione media ricorsioni: {avg_rec_reduction:.1f}%")
    print(f"Tempo risparmiato medio: {mean([old-new for old, new in zip(times_old, times_new)]):.1f} ms")
    
    # Grafici
    labels = [c["desc"] for c in configs[:len(results_old)]]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Grafico 1: Tempi
    ax1.bar(x - width/2, times_old, width, label='Vecchia', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, times_new, width, label='Ottimizzata', color='#4ECDC4', alpha=0.8)
    ax1.set_xlabel('Configurazione')
    ax1.set_ylabel('Tempo (ms)')
    ax1.set_title('Confronto Tempi di Esecuzione')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Grafico 2: Ricorsioni
    ax2.bar(x - width/2, recs_old, width, label='Vecchia', color='#FF6B6B', alpha=0.8)
    ax2.bar(x + width/2, recs_new, width, label='Ottimizzata', color='#4ECDC4', alpha=0.8)
    ax2.set_xlabel('Configurazione')
    ax2.set_ylabel('Numero Ricorsioni')
    ax2.set_title('Confronto Numero Ricorsioni')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Grafico 3: Speedup
    speedups = [old/new for old, new in zip(times_old, times_new) if new > 0]
    colors = ['#51CF66' if s > 1 else '#FF6B6B' for s in speedups]
    ax3.bar(x, speedups, color=colors, alpha=0.8)
    ax3.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Parità (1x)')
    ax3.set_xlabel('Configurazione')
    ax3.set_ylabel('Speedup (x)')
    ax3.set_title('Speedup della Versione Ottimizzata')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Grafico 4: Nodi potati
    pruned = [r.pruned_nodes for r in results_new]
    ax4.bar(x, pruned, color='#FFA500', alpha=0.8)
    ax4.set_xlabel('Configurazione')
    ax4.set_ylabel('Nodi Potati')
    ax4.set_title('Efficacia del Pruning (solo versione ottimizzata)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_camminomin.png', dpi=150, bbox_inches='tight')
    print("\nGrafici salvati in: benchmark_camminomin.png")

    csv_filename = "benchmark_camminomin.csv"
    save_results_to_csv(csv_filename, configs[:len(results_old)], results_old, results_new)

    print(f"Risultati salvati in CSV: {csv_filename}")


if __name__ == "__main__":
    random.seed(42)
    run_comprehensive_benchmark()