#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compito 1 – Generatore di griglie finite con ostacoli
Algoritmi e Strutture Dati (a.a. 2024/25)

Caratteristiche:
- Griglia bidimensionale finita con celle attraversabili (0) o non attraversabili (1).
- Supporto ad almeno 4 tipologie di ostacoli:
  1) Semplici (celle isolate)
  2) Agglomerati (cluster 8-connessi)
  3) Diagonali (catene che si toccano solo per spigolo)
  4) Delimitatori di aree chiuse (cornici rettangolari)
  5) Barre orizzontali/verticali (facoltativo, attivo per default)
- Parametri configurabili (dimensioni, seed, quantità e dimensioni ostacoli).
- Possibilità di combinare più tipologie nella stessa griglia.
- Esportazione in CSV (0/1), TXT ASCII e JSON di riepilogo.

Uso (esempi):
    python grid_generator.py --width 40 --height 25 --seed 123 \
        --simple 40 \
        --agglomerates 6 --agg-min 3 --agg-max 8 \
        --diagonals 4 --diag-min 3 --diag-max 10 \
        --frames 3 --frame-minw 6 --frame-minh 4 --frame-maxw 16 --frame-maxh 10 --frame-thick 1 \
        --bars 5 --bar-min 4 --bar-max 12 --bar-thick 1

    # Salvataggi (cartella di output)
    python grid_generator.py -W 60 -H 35 -S 99 --simple 120 --bars 10 -o out/

Note implementative:
- Il generatore prova a posizionare oggetti evitando sovrapposizioni; ha un limite di tentativi
  per non bloccarsi quando la griglia è piena.
- Non richiede librerie esterne. (Solo Python standard library.)
"""

from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

Cell = Tuple[int, int]
DIAG_STEPS: List[Cell] = [(-1,-1),(-1,1),(1,-1),(1,1)]
NEIGH8: List[Cell] = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

@dataclass
class GridConfig:
    width: int
    height: int
    seed: int | None = None
    simple: int = 0
    agglomerates: int = 0
    agg_min: int = 3
    agg_max: int = 7
    diagonals: int = 0
    diag_min: int = 3
    diag_max: int = 10
    frames: int = 0
    frame_minw: int = 5
    frame_minh: int = 4
    frame_maxw: int = 15
    frame_maxh: int = 10
    frame_thick: int = 1
    bars: int = 0
    bar_min: int = 4
    bar_max: int = 12
    bar_thick: int = 1
    out_dir: str | None = None

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Larghezza e altezza devono essere positivi.")
        if self.agg_min < 1 or self.agg_max < self.agg_min:
            raise ValueError("Valore minimo cluster deve essere almeno 1 e max >= min.")
        if self.diag_min < 2 or self.diag_max < self.diag_min:
            raise ValueError("Valore minimo catena diagonale deve essere almeno 2 e max >= min.")
        if self.frame_minw < 3 or self.frame_minh < 3 or self.frame_maxw < self.frame_minw or self.frame_maxh < self.frame_minh:
            raise ValueError("Dimensioni cornici non validi.")
        if self.frame_thick < 1:
            raise ValueError("Spessore cornici deve essere almeno 1.")
        if self.bar_min < 2 or self.bar_max < self.bar_min:
            raise ValueError("Lunghezza barre non valida.")
        if self.bar_thick < 1:
            raise ValueError("Spessore barre deve essere almeno 1.")
        if self.simple < 0 or self.agglomerates < 0 or self.diagonals < 0 or self.frames < 0 or self.bars < 0:
            raise ValueError("Numero di ostacoli non può essere negativo.")

class Grid:
    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width
        self.cells: List[List[int]] = [[0] * width for _ in range(height)] 

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def get(self, r: int, c: int) -> int:
        return self.cells[r][c]

    def set_blocked(self, r: int, c: int):
        self.cells[r][c] = 1

    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.cells[r][c] == 0

    def occupancy(self) -> int:
        return sum(sum(row) for row in self.cells)

    def density(self) -> float:
        return self.occupancy() / (self.h * self.w)

    def to_csv(self) -> str:
        lines = [",".join(str(v) for v in row) for row in self.cells]
        return "\n".join(lines)

    def to_ascii(self) -> str:
        # '.' libero, '#' ostacolo
        legend = {0: ".", 1: "#"}
        return "\n".join("".join(legend[v] for v in row) for row in self.cells)

    def save_all(self, basepath: str, meta: Dict):
        os.makedirs(os.path.dirname(basepath), exist_ok=True)
        with open(basepath + ".csv", "w", encoding="utf-8") as f:
            f.write(self.to_csv())
        with open(basepath + ".txt", "w", encoding="utf-8") as f:
            f.write(self.to_ascii())
        with open(basepath + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def _random_free_cell(g: Grid, rnd: random.Random) -> Cell | None:
    for _ in range(100):
        r = rnd.randrange(g.h)
        c = rnd.randrange(g.w)
        if g.is_free(r, c):
            return (r, c)

    for rr in range(g.h): 
        for cc in range(g.w):
            if g.is_free(rr, cc):
                return (rr, cc)
    return None


def place_simple_cells(g: Grid, count: int, rnd: random.Random) -> int:
    placed = 0
    attempts = 0
    max_attempts = count * 10 + 100
    while placed < count and attempts < max_attempts:
        attempts += 1
        cell = _random_free_cell(g, rnd)
        if cell is None:
            break
        r, c = cell
        if g.is_free(r, c):
            g.set_blocked(r, c)
            placed += 1
    return placed


def _neighbors8(r: int, c: int) -> List[Cell]:
    return [(r+dr, c+dc) for dr, dc in NEIGH8]

def place_agglomerate(g: Grid, size: int, rnd: random.Random) -> int:
    start = _random_free_cell(g, rnd)
    if start is None:
        return 0
    frontier = [start]
    blocked: List[Cell] = []
    
    while frontier and len(blocked) < size: 
        r, c = frontier.pop(rnd.randrange(len(frontier)))
        if not g.is_free(r, c):
            continue
        g.set_blocked(r, c)
        blocked.append((r, c))
        for nr, nc in _neighbors8(r, c):
            if g.in_bounds(nr, nc) and g.is_free(nr, nc):
                frontier.append((nr, nc))
    return len(blocked)

def place_agglomerates(g: Grid, count: int, size_min: int, size_max: int, rnd: random.Random) -> Dict[str,int]:
    placed_clusters = 0
    placed_cells = 0
    attempts = 0
    while placed_clusters < count and attempts < count * 20:
        attempts += 1
        size = rnd.randint(size_min, size_max)
        got = place_agglomerate(g, size, rnd)
        if got > 0:
            placed_clusters += 1
            placed_cells += got
    return {"clusters": placed_clusters, "cells": placed_cells}



def place_diagonal_chain(g: Grid, length: int, rnd: random.Random) -> int:
    start = _random_free_cell(g, rnd)
    if start is None:
        return 0
    r, c = start
    placed = 0
    step = rnd.choice(DIAG_STEPS)
    turn_at = rnd.randint(2, max(2, length-1)) if length >= 4 else length+1
    for i in range(length):
        if not g.is_free(r, c):
            break
        g.set_blocked(r, c)
        placed += 1
        if i+1 == turn_at:
            step = rnd.choice(DIAG_STEPS)
        nr, nc = r + step[0], c + step[1] 
        if not g.in_bounds(nr, nc) or not g.is_free(nr, nc):
            alternatives = [(r+dr, c+dc) for dr, dc in DIAG_STEPS if g.in_bounds(r+dr, c+dc) and g.is_free(r+dr, c+dc)]
            if not alternatives:
                break
            nr, nc = rnd.choice(alternatives)
        r, c = nr, nc
    return placed


def place_diagonals(g: Grid, count: int, len_min: int, len_max: int, rnd: random.Random) -> Dict[str,int]:
    chains = 0
    cells = 0
    attempts = 0 #numero di tentativi fatti
    while chains < count and attempts < count * 20:
        attempts += 1
        L = rnd.randint(len_min, len_max)
        got = place_diagonal_chain(g, L, rnd)
        if got >= max(2, len_min // 2):
            chains += 1
            cells += got
    return {"chains": chains, "cells": cells}


def _draw_frame(g: Grid, top: int, left: int, height: int, width: int, thick: int) -> int:
    placed = 0
    bottom = top + height - 1
    right = left + width - 1
    for t in range(thick):
        r_top = top + t
        r_bot = bottom - t
        for c in range(left, right+1):
            if g.is_free(r_top, c):
                g.set_blocked(r_top, c); placed += 1
            if g.is_free(r_bot, c):
                g.set_blocked(r_bot, c); placed += 1

        c_left = left + t
        c_right = right - t
        for r in range(top, bottom+1):
            if g.is_free(r, c_left):
                g.set_blocked(r, c_left); placed += 1
            if g.is_free(r, c_right):
                g.set_blocked(r, c_right); placed += 1
    return placed


def place_frames(g: Grid, count: int, minw: int, minh: int, maxw: int, maxh: int, thick: int, rnd: random.Random) -> Dict[str,int]:
    frames = 0
    cells = 0
    attempts = 0
    while frames < count and attempts < count * 50:
        attempts += 1
        w = rnd.randint(minw, min(maxw, g.w-2))
        h = rnd.randint(minh, min(maxh, g.h-2))
        left = rnd.randint(0, g.w - w - 1)
        top = rnd.randint(0, g.h - h - 1)
        if min(w, h) <= 2*thick:
            continue  
        before = g.occupancy()
        _draw_frame(g, top, left, h, w, thick)
        after = g.occupancy()
        delta = after - before
        if delta > 0:
            frames += 1
            cells += delta
    return {"frames": frames, "cells": cells}



def _draw_bar(g: Grid, r0: int, c0: int, length: int, orient: str, thick: int) -> int:
    placed = 0
    if orient == 'H':
        for dr in range(thick):
            r = r0 + dr
            for c in range(c0, min(g.w, c0 + length)):
                if g.in_bounds(r, c) and g.is_free(r, c):
                    g.set_blocked(r, c); placed += 1
    else:
        for dc in range(thick):
            c = c0 + dc
            for r in range(r0, min(g.h, r0 + length)):
                if g.in_bounds(r, c) and g.is_free(r, c):
                    g.set_blocked(r, c); placed += 1
    return placed


def place_bars(g: Grid, count: int, len_min: int, len_max: int, thick: int, rnd: random.Random) -> Dict[str,int]:
    bars = 0
    cells = 0
    attempts = 0
    while bars < count and attempts < count * 50:
        attempts += 1
        orient = 'H' if rnd.random() < 0.5 else 'V'
        L = rnd.randint(len_min, len_max)
        if orient == 'H':
            r0 = rnd.randint(0, max(0, g.h - thick))
            c0 = rnd.randint(0, g.w - 1)
        else:
            r0 = rnd.randint(0, g.h - 1)
            c0 = rnd.randint(0, max(0, g.w - thick))
        before = g.occupancy()
        _draw_bar(g, r0, c0, L, orient, thick)
        after = g.occupancy()
        delta = after - before
        if delta > 0:
            bars += 1
            cells += delta
    return {"bars": bars, "cells": cells}



def generate(config: GridConfig) -> Tuple[Grid, Dict]:
    rnd = random.Random(config.seed)
    g = Grid(height=config.height, width=config.width)

    summary: Dict[str, Dict] = {"config": asdict(config), "placed": {}}

    if config.simple > 0:
        got = place_simple_cells(g, config.simple, rnd)
        summary["placed"]["simple"] = {"cells": got}

    if config.agglomerates > 0:
        res = place_agglomerates(g, config.agglomerates, config.agg_min, config.agg_max, rnd)
        summary["placed"]["agglomerates"] = res

    if config.diagonals > 0:
        res = place_diagonals(g, config.diagonals, config.diag_min, config.diag_max, rnd)
        summary["placed"]["diagonals"] = res

    if config.frames > 0:
        res = place_frames(g, config.frames, config.frame_minw, config.frame_minh, config.frame_maxw, config.frame_maxh, config.frame_thick, rnd)
        summary["placed"]["frames"] = res

    if config.bars > 0:
        res = place_bars(g, config.bars, config.bar_min, config.bar_max, config.bar_thick, rnd)
        summary["placed"]["bars"] = res

    summary["grid"] = {
        "height": g.h,
        "width": g.w,
        "occupied_cells": g.occupancy(),
        "density": round(g.density(), 4),
    }

    return g, summary



def build_argparser() -> argparse.ArgumentParser: 
    p = argparse.ArgumentParser(description="Generatore di griglie con ostacoli (Compito 1)")
    p.add_argument("--width", "-W", type=int, required=True, help="larghezza griglia (colonne)")
    p.add_argument("--height", "-H", type=int, required=True, help="altezza griglia (righe)")
    p.add_argument("--seed", "-S", type=int, default=None, help="seed RNG (opzionale)")

    p.add_argument("--simple", type=int, default=0, help="numero di celle semplici")

    p.add_argument("--agglomerates", type=int, default=0, help="numero di cluster agglomerati")
    p.add_argument("--agg-min", type=int, default=3, help="dimensione minima cluster")
    p.add_argument("--agg-max", type=int, default=7, help="dimensione massima cluster")

    p.add_argument("--diagonals", type=int, default=0, help="numero di catene diagonali")
    p.add_argument("--diag-min", type=int, default=3, help="lunghezza minima catena diagonale")
    p.add_argument("--diag-max", type=int, default=10, help="lunghezza massima catena diagonale")

    p.add_argument("--frames", type=int, default=0, help="numero di cornici (aree chiuse)")
    p.add_argument("--frame-minw", type=int, default=5)
    p.add_argument("--frame-minh", type=int, default=4)
    p.add_argument("--frame-maxw", type=int, default=15)
    p.add_argument("--frame-maxh", type=int, default=10)
    p.add_argument("--frame-thick", type=int, default=1)

    p.add_argument("--bars", type=int, default=0, help="numero di barre orizzontali/verticali")
    p.add_argument("--bar-min", type=int, default=4)
    p.add_argument("--bar-max", type=int, default=12)
    p.add_argument("--bar-thick", type=int, default=1)

    p.add_argument("-o", "--out-dir", type=str, default=None, help="cartella di output per CSV/TXT/JSON")
    return p


def main():
    ap = build_argparser()
    args = ap.parse_args()
    cfg = GridConfig(
        width=args.width,
        height=args.height,
        seed=args.seed,
        simple=args.simple,
        agglomerates=args.agglomerates,
        agg_min=args.agg_min,
        agg_max=args.agg_max,
        diagonals=args.diagonals,
        diag_min=args.diag_min,
        diag_max=args.diag_max,
        frames=args.frames,
        frame_minw=args.frame_minw,
        frame_minh=args.frame_minh,
        frame_maxw=args.frame_maxw,
        frame_maxh=args.frame_maxh,
        frame_thick=args.frame_thick,
        bars=args.bars,
        bar_min=args.bar_min,
        bar_max=args.bar_max,
        bar_thick=args.bar_thick,
        out_dir=args.out_dir,
    )

    g, summary = generate(cfg)

    print(g.to_ascii())
    print("\n-- Riepilogo --")
    print(json.dumps(summary, indent=2))

    if cfg.out_dir:
        base = os.path.join(cfg.out_dir, f"grid_{cfg.width}x{cfg.height}_seed{cfg.seed if cfg.seed is not None else 'NA'}")
        g.save_all(base, summary)
        print(f"\nFile salvati con prefisso: {base}.[csv|txt|json]")

if __name__ == "__main__":
    main()
