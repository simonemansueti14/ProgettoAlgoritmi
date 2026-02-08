#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark per confrontare le prestazioni di compute_context_and_complement
Versione VECCHIA vs NUOVA
"""

import time
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Set, List, Callable
from dataclasses import dataclass
from statistics import mean, stdev

Cell = Tuple[int, int]

# ========================= MOCK GRID CLASS =========================
class Grid:
    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.cells = [[0] * w for _ in range(h)]
    
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w
    
    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.cells[r][c] == 0
    
    def set_obstacle(self, r: int, c: int):
        if self.in_bounds(r, c):
            self.cells[r][c] = 1
    
    def generate_random_obstacles(self, density: float = 0.2):
        """Genera ostacoli casuali con una certa densità (0.0 - 1.0)"""
        for r in range(self.h):
            for c in range(self.w):
                if random.random() < density:
                    self.cells[r][c] = 1


# ========================= UTILITY =========================
def sign(x: int) -> int:
    return (x > 0) - (x < 0)


# ========================= VERSIONE VECCHIA =========================
def compute_context_and_complement_OLD(g: Grid, O: Cell) -> Tuple[Set[Cell], Set[Cell]]:
    """VERSIONE ORIGINALE (non ottimizzata)"""
    context: Set[Cell] = set()
    complement: Set[Cell] = set()

    rows, cols = g.h, g.w
    r0, c0 = O

    if not g.in_bounds(r0, c0) or not g.is_free(r0, c0):
        raise ValueError(f"Origine {O} non valida")

    diagonals = [(-1, 1), (-1, -1), (1, -1), (1, 1)]
    horizontals = [(0, 1), (0, -1)]
    verticals = [(-1, 0), (1, 0)]

    def free_path(r, c, dr, dc, steps) -> Tuple[int, int] | None:
        for _ in range(steps):
            r += dr
            c += dc
            if not g.in_bounds(r, c) or not g.is_free(r, c):
                return None
        return (r, c)

    # Itera su tutte le celle
    for r in range(rows):
        for c in range(cols):
            if (r, c) == O or not g.is_free(r, c):
                continue

            found_type1 = False
            found_type2 = False

            # TIPO 1: oblique poi rette
            for ddr, ddc in diagonals:
                for k in range(0, max(rows, cols)):
                    start = free_path(r0, c0, ddr, ddc, k)
                    if start is None:
                        break
                    sr, sc = start
                    if (sr, sc) == (r, c):
                        found_type1 = True
                        break
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

            # Cammini puri
            if not found_type1:
                for dr, dc in horizontals + verticals:
                    end = free_path(r0, c0, dr, dc, max(abs(r - r0), abs(c - c0)))
                    if end == (r, c):
                        found_type1 = True
                        break
                if not found_type1:
                    for dr, dc in diagonals:
                        end = free_path(r0, c0, dr, dc, abs(r - r0))
                        if end == (r, c):
                            found_type1 = True
                            break

            # TIPO 2: dritte poi oblique
            for dr, dc in horizontals + verticals:
                for m in range(1, max(rows, cols)):
                    start = free_path(r0, c0, dr, dc, m)
                    if start is None:
                        break
                    sr, sc = start

                    possible_diagonals = []
                    if sr < r0 and sc > c0:
                        possible_diagonals = [(-1, 1)]
                    elif sr < r0 and sc < c0:
                        possible_diagonals = [(-1, -1)]
                    elif sr > r0 and sc < c0:
                        possible_diagonals = [(1, -1)]
                    elif sr > r0 and sc > c0:
                        possible_diagonals = [(1, 1)]
                    else:
                        if sr == r0:
                            if sc > c0:
                                possible_diagonals = [(-1, 1), (1, 1)]
                            elif sc < c0:
                                possible_diagonals = [(-1, -1), (1, -1)]
                        elif sc == c0:
                            if sr < r0:
                                possible_diagonals = [(-1, 1), (-1, -1)]
                            elif sr > r0:
                                possible_diagonals = [(1, 1), (1, -1)]

                    for ddr, ddc in possible_diagonals:
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


# ========================= VERSIONE NUOVA =========================
def is_path_free(g: Grid, r: int, c: int, dr: int, dc: int, steps: int) -> bool:
    for _ in range(steps):
        r += dr
        c += dc
        if not g.in_bounds(r, c) or not g.is_free(r, c):
            return False
    return True


def is_type1_reachable(g: Grid, r0: int, c0: int, r: int, c: int, dr: int, dc: int) -> bool:
    # Solo ortogonale
    if dr == 0 and dc != 0:
        return is_path_free(g, r0, c0, 0, sign(dc), abs(dc))
    if dc == 0 and dr != 0:
        return is_path_free(g, r0, c0, sign(dr), 0, abs(dr))
    
    # Solo diagonale
    if abs(dr) == abs(dc):
        if is_path_free(g, r0, c0, sign(dr), sign(dc), abs(dr)):
            return True
    
    # Diagonale + Ortogonale
    diag_steps = min(abs(dr), abs(dc))
    ddr, ddc = sign(dr), sign(dc)
    
    for k in range(1, diag_steps + 1):
        mid_r, mid_c = r0 + k * ddr, c0 + k * ddc
        
        if not is_path_free(g, r0, c0, ddr, ddc, k):
            break
        
        rem_dr, rem_dc = r - mid_r, c - mid_c
        
        if rem_dr == 0 and rem_dc != 0:
            if sign(rem_dc) == ddc:
                if is_path_free(g, mid_r, mid_c, 0, sign(rem_dc), abs(rem_dc)):
                    return True
        elif rem_dc == 0 and rem_dr != 0:
            if sign(rem_dr) == ddr:
                if is_path_free(g, mid_r, mid_c, sign(rem_dr), 0, abs(rem_dr)):
                    return True
    
    return False


def is_type2_reachable(g: Grid, r0: int, c0: int, r: int, c: int, dr: int, dc: int) -> bool:
    if dr == 0 or dc == 0:
        return False
    
    # Orizzontale + Diagonale
    for hor_steps in range(1, abs(dc) + 1):
        mid_r, mid_c = r0, c0 + hor_steps * sign(dc)
        
        if not is_path_free(g, r0, c0, 0, sign(dc), hor_steps):
            break
        
        rem_dr, rem_dc = r - mid_r, c - mid_c
        
        if abs(rem_dr) == abs(rem_dc) and abs(rem_dr) > 0:
            if sign(rem_dc) == sign(dc) and sign(rem_dr) * dr > 0:
                if is_path_free(g, mid_r, mid_c, sign(rem_dr), sign(rem_dc), abs(rem_dr)):
                    return True
    
    # Verticale + Diagonale
    for ver_steps in range(1, abs(dr) + 1):
        mid_r, mid_c = r0 + ver_steps * sign(dr), c0
        
        if not is_path_free(g, r0, c0, sign(dr), 0, ver_steps):
            break
        
        rem_dr, rem_dc = r - mid_r, c - mid_c
        
        if abs(rem_dr) == abs(rem_dc) and abs(rem_dr) > 0:
            if sign(rem_dr) == sign(dr) and sign(rem_dc) * dc > 0:
                if is_path_free(g, mid_r, mid_c, sign(rem_dr), sign(rem_dc), abs(rem_dr)):
                    return True
    
    return False


def compute_context_and_complement_NEW(g: Grid, O: Cell) -> Tuple[Set[Cell], Set[Cell]]:
    """VERSIONE OTTIMIZZATA"""
    context: Set[Cell] = set()
    complement: Set[Cell] = set()
    
    r0, c0 = O
    
    if not g.in_bounds(r0, c0) or not g.is_free(r0, c0):
        raise ValueError(f"Origine {O} non valida")
    
    for r in range(g.h):
        for c in range(g.w):
            if (r, c) == O or not g.is_free(r, c):
                continue
            
            dr, dc = r - r0, c - c0
            
            if is_type1_reachable(g, r0, c0, r, c, dr, dc):
                context.add((r, c))
            elif is_type2_reachable(g, r0, c0, r, c, dr, dc):
                complement.add((r, c))
    
    return context, complement


# ========================= BENCHMARK =========================
@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    context_size: int
    complement_size: int


def benchmark_function(
    func: Callable,
    grid: Grid,
    origin: Cell,
    runs: int = 5
) -> BenchmarkResult:
    """Esegue un benchmark di una funzione con più run"""
    times = []
    
    for _ in range(runs):
        start = time.perf_counter()
        context, complement = func(grid, origin)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Converti in ms
    
    return BenchmarkResult(
        name=func.__name__,
        time_ms=mean(times),
        context_size=len(context),
        complement_size=len(complement)
    )


def run_comprehensive_benchmark():
    """Esegue benchmark completo su diverse configurazioni"""
    print("=" * 70)
    print("BENCHMARK: compute_context_and_complement - VECCHIA vs NUOVA")
    print("=" * 70)
    
    # Configurazioni di test
    configs = [
        {"size": 10, "density": 0.1, "runs": 10},
        {"size": 20, "density": 0.1, "runs": 10},
        {"size": 30, "density": 0.1, "runs": 5},
        {"size": 50, "density": 0.1, "runs": 3},
        {"size": 20, "density": 0.3, "runs": 10},
        {"size": 30, "density": 0.3, "runs": 5},
    ]
    
    results_old = []
    results_new = []
    
    for config in configs:
        size = config["size"]
        density = config["density"]
        runs = config["runs"]
        
        print(f"\n--- Test: Griglia {size}x{size}, Densità ostacoli {density:.0%} ---")
        
        # Genera griglia
        g = Grid(size, size)
        g.generate_random_obstacles(density)
        
        # Trova origine valida (cella libera al centro)
        origin = (size // 2, size // 2)
        while not g.is_free(*origin):
            origin = (random.randint(0, size-1), random.randint(0, size-1))
        
        print(f"Origine: {origin}")
        
        # Benchmark VECCHIA
        print("  Esecuzione VECCHIA versione...", end=" ")
        res_old = benchmark_function(compute_context_and_complement_OLD, g, origin, runs)
        results_old.append(res_old)
        print(f"✓ {res_old.time_ms:.2f} ms")
        
        # Benchmark NUOVA
        print("  Esecuzione NUOVA versione...", end=" ")
        res_new = benchmark_function(compute_context_and_complement_NEW, g, origin, runs)
        results_new.append(res_new)
        print(f"✓ {res_new.time_ms:.2f} ms")
        
        # Speedup
        speedup = res_old.time_ms / res_new.time_ms
        print(f"  📊 Speedup: {speedup:.2f}x")
        
        # Verifica correttezza
        if res_old.context_size != res_new.context_size or res_old.complement_size != res_new.complement_size:
            print("  ⚠️  WARNING: Risultati differenti!")
    
    # Riepilogo grafico
    print("\n" + "=" * 70)
    print("RIEPILOGO GENERALE")
    print("=" * 70)
    
    labels = [f"{c['size']}x{c['size']}\n{c['density']:.0%}" for c in configs]
    times_old = [r.time_ms for r in results_old]
    times_new = [r.time_ms for r in results_new]
    
    # Calcola statistiche
    avg_speedup = mean([old/new for old, new in zip(times_old, times_new)])
    print(f"Speedup medio: {avg_speedup:.2f}x")
    print(f"Tempo risparmiato medio: {mean([old-new for old, new in zip(times_old, times_new)]):.2f} ms")
    
    # Grafico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grafico 1: Confronto tempi
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, times_old, width, label='Vecchia', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, times_new, width, label='Nuova', color='#4ECDC4', alpha=0.8)
    ax1.set_xlabel('Configurazione (Size x Densità)')
    ax1.set_ylabel('Tempo (ms)')
    ax1.set_title('Confronto Tempi di Esecuzione')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Grafico 2: Speedup
    speedups = [old/new for old, new in zip(times_old, times_new)]
    colors = ['#51CF66' if s > 1 else '#FF6B6B' for s in speedups]
    
    ax2.bar(x, speedups, color=colors, alpha=0.8)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Parità (1x)')
    ax2.set_xlabel('Configurazione')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup della Versione Nuova')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_context_complement.png', dpi=150, bbox_inches='tight')
    print("\n📊 Grafico salvato in: benchmark_context_complement.png")
    plt.show()


# ========================= MAIN =========================
if __name__ == "__main__":
    random.seed(42)  # Riproducibilità
    run_comprehensive_benchmark()