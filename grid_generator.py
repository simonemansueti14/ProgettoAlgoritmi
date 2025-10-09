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

Cell = Tuple[int, int]  # (row, col)

# GridConfig è un dataclass, quindi una specie di contenitore di dati che raccoglie tutti i parametri
@dataclass
class GridConfig:
    width: int #dimensioni griglia
    height: int
    seed: int | None = None #seed serve per fissare la casualità, cosi ottieni sempre la stessa griglia se ripeti
    # Semplici
    simple: int = 0 #insieme ad agglomerats rappresenta quanti ostacoli piazzare
    # Agglomerati
    agglomerates: int = 0
    agg_min: int = 3
    agg_max: int = 7
    # Diagonali
    diagonals: int = 0
    diag_min: int = 3
    diag_max: int = 10
    # Cornici (aree chiuse)
    frames: int = 0
    frame_minw: int = 5
    frame_minh: int = 4
    frame_maxw: int = 15
    frame_maxh: int = 10
    frame_thick: int = 1
    # Barre
    bars: int = 0
    bar_min: int = 4
    bar_max: int = 12
    bar_thick: int = 1
    # Output
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

# classe che rappresenta la griglia, matrice di 0 (cellalibera) e 1 (cellaostacolo)
class Grid:
    def __init__(self, height: int, width: int): #init è il costruttore
        self.h = height #self.h la memorizza come variabile interna
        self.w = width
        #self.cells è la matrice della griglia, lista di liste di interi
        #[ [0]*width for _ in range(height) ] crea una matrice di dimensione h x w tutta riempita di 0
        self.cells: List[List[int]] = [[0] * width for _ in range(height)] 

    #controlla se le coordinate (r, c) stanno dentro la griglia
    #restituisce true se la riga è tra 0 e h-1 e la colonna è tra 0 e w-1
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    #restituisce valore (r,c), 0 se libera 1 se ostacolo
    def get(self, r: int, c: int) -> int:
        return self.cells[r][c]

    #trasforma cella (r,c) in ostacolo mettendo 1
    def set_blocked(self, r: int, c: int):
        self.cells[r][c] = 1

    #dice se (r,c) è libera, ma prima controlla in_bounds
    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.cells[r][c] == 0

    #conta quante celle sono ostacoli, sum(row) somma i valori di una riga, poi somma tutte le righe
    def occupancy(self) -> int:
        return sum(sum(row) for row in self.cells)

    #calcola densità degli ostacoli (occupate) / (totali)
    #risultato è numero compreso tra 0.0 e 1.0
    def density(self) -> float:
        return self.occupancy() / (self.h * self.w)


    # -------------------------------- EXPORT --------------------------------
    #converte la griglia in testo formato csv, quindi numeri separati da virgole
    def to_csv(self) -> str:
        lines = [",".join(str(v) for v in row) for row in self.cells]
        return "\n".join(lines)

    #converte la griglia in ASCII, . = libera, # = ostacolo
    def to_ascii(self) -> str:
        # '.' libero, '#' ostacolo
        legend = {0: ".", 1: "#"}
        return "\n".join("".join(legend[v] for v in row) for row in self.cells)

    #salva griglia su 3 file diversi
    def save_all(self, basepath: str, meta: Dict):
        os.makedirs(os.path.dirname(basepath), exist_ok=True)
        with open(basepath + ".csv", "w", encoding="utf-8") as f:
            f.write(self.to_csv())
        with open(basepath + ".txt", "w", encoding="utf-8") as f:
            f.write(self.to_ascii())
        with open(basepath + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# ------------------------------ OBSTACLE PLACER ------------------------------
#cerca una cella libera nella griglia
def _random_free_cell(g: Grid, rnd: random.Random) -> Cell | None:
    # Prova con samplings limitati, poi fallback a scansione
    for _ in range(100): #fa fino a 100 tentativi casuali
        r = rnd.randrange(g.h) #sceglie riga e colonna valide a caso
        c = rnd.randrange(g.w)
        if g.is_free(r, c): #se la cella è libera la restituisce subito
            return (r, c)
    #se i 100 tentativi falliscono passa al fallaback, due for annidati con cui scansiona tutta la griglia
    #prima dell'ultima cella, appena trova una cella libera la ritorna
    #se non c'è una cella libera return none
    for rr in range(g.h): 
        for cc in range(g.w):
            if g.is_free(rr, cc):
                return (rr, cc)
    return None



# 1) ostacoli semplici: celle singole
def place_simple_cells(g: Grid, count: int, rnd: random.Random) -> int:
    placed = 0 #ostacoli effettivamente piazzati
    attempts = 0 #tentativi fatti
    max_attempts = count * 10 + 100 #massimo tentativi consentito per non andare all'infinito
    while placed < count and attempts < max_attempts: #continua finché non ha piazzato tutto e non ha superato il limite
        attempts += 1
        cell = _random_free_cell(g, rnd) #chiede cella libera, se non c'è esce
        if cell is None:
            break
        r, c = cell
        if g.is_free(r, c): #marca quella cella come ostacolo ed esce
            g.set_blocked(r, c)
            placed += 1
    return placed



# 2) agglomerati: cluster 8-connessi via random walk/BFS growth
NEIGH8: List[Cell] = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] #8 direzioni (N,NE,NO,...)
NEIGH4: List[Cell] = [(-1,0),(1,0),(0,-1),(0,1)]    #4 direzioni 

#dato (r,c) restituiscono la lista delle coordinate vicine ( 8 o 4 direzioni)
#non filtrano ancora i limiti, lo si fa con g.in_bounds(...)
def _neighbors8(r: int, c: int) -> List[Cell]:
    return [(r+dr, c+dc) for dr, dc in NEIGH8]

def _neighbors4(r: int, c: int) -> List[Cell]:
    return [(r+dr, c+dc) for dr, dc in NEIGH4]

#crea un agglomerato di ostacoli fino a size celle, è 8-connesso
def place_agglomerate(g: Grid, size: int, rnd: random.Random) -> int:
    start = _random_free_cell(g, rnd) #sceglie un seme libero da cui fa crescere il cluster, se non esiste torna 0
    if start is None:
        return 0
    frontier = [start] #frontiera delle celle candidate dove far crescere il cluster
    blocked: List[Cell] = [] #lista delle celle davvero piazzate come ostacoli
    #finché ho candidati e non ho raggiunto la dimensione richiesta faccio...
    while frontier and len(blocked) < size: 
        r, c = frontier.pop(rnd.randrange(len(frontier))) #prendo a caso una cella dalla frontiera
        if not g.is_free(r, c): #se è libera la blocco e la aggiungo a blocked
            continue
        g.set_blocked(r, c)
        blocked.append((r, c))
        # aggiungi vicini 8-connessi candidati
        for nr, nc in _neighbors8(r, c):
            if g.in_bounds(nr, nc) and g.is_free(nr, nc):
                frontier.append((nr, nc))
    return len(blocked) #può essere meno di size se lo spazio attorno non consentiva di crescere di più
    #il risultato è una crescita random del cluster in tutte le direzioni possibili, finché
    #raggiungo size o mi si esauriscono i candidati (per es sono circondato da ostacoli preesistenti)

#piazza gli agglomerati (numero pari a count con dimensione compresa tra size_min e size_max)
def place_agglomerates(g: Grid, count: int, size_min: int, size_max: int, rnd: random.Random) -> Dict[str,int]:
    placed_clusters = 0 #cluster riusciti
    placed_cells = 0 #celle totali bloccate in tutti i cluster
    attempts = 0
    while placed_clusters < count and attempts < count * 20: #limita i tentativi
        attempts += 1
        size = rnd.randint(size_min, size_max) #sceglie una dimensione casuale
        got = place_agglomerate(g, size, rnd)
        if got > 0: #se ha piazzato almeno una cella conta un cluster riuscito
            placed_clusters += 1
            placed_cells += got
    return {"clusters": placed_clusters, "cells": placed_cells}



# 3) diagonali: catene che si muovono solo in diagonale
DIAG_STEPS: List[Cell] = [(-1,-1),(-1,1),(1,-1),(1,1)] #lista degli spostamenti diagonali
#ogni coppia (dr, dc) dice come cambiare riga e colonna per andare alla prossima cella

#sceglie una cella di partenza libera, se non c'è interrompe e ritorna a 0
def place_diagonal_chain(g: Grid, length: int, rnd: random.Random) -> int:
    start = _random_free_cell(g, rnd)
    if start is None:
        return 0
    r, c = start
    placed = 0 #celle effettivamente piazzate
    step = rnd.choice(DIAG_STEPS) #direzione diagonale iniziale scelta a caso
    #turn_at è il punto in cui, a metà percorso, potrebbe cambiare direzione diagonale
    #se la catena è corta (length < 4), probabilmente non cambia mai
    turn_at = rnd.randint(2, max(2, length-1)) if length >= 4 else length+1
    #Cicla fino alla lunghezza desiderata, se la cella non è più libera interrompe altrimenti la marca come ostacolo
    for i in range(length):
        if not g.is_free(r, c):
            break
        g.set_blocked(r, c)
        placed += 1
        if i+1 == turn_at: #quando arriva qui cambia direzione casualmente
            step = rnd.choice(DIAG_STEPS)
        nr, nc = r + step[0], c + step[1] #calcola la prossima cella spostandosi nella direzione corrente
        #se la prossima cella non è valida cerca un alternativa scegliendo tra le diagonali disponibili, 
        #se non c'è interrompe
        if not g.in_bounds(nr, nc) or not g.is_free(nr, nc):
            # prova a scegliere una diagonale alternativa se bloccata
            alternatives = [(r+dr, c+dc) for dr, dc in DIAG_STEPS if g.in_bounds(r+dr, c+dc) and g.is_free(r+dr, c+dc)]
            if not alternatives:
                break
            nr, nc = rnd.choice(alternatives)
        r, c = nr, nc #aggiorna la posizione
    return placed

#piazza catene diagonali, tiene traccia di quante catene sono riuscite (chains)e di quante totali
#sono state occupate
def place_diagonals(g: Grid, count: int, len_min: int, len_max: int, rnd: random.Random) -> Dict[str,int]:
    chains = 0
    cells = 0
    attempts = 0 #numero di tentativi fatti
    while chains < count and attempts < count * 20:
        attempts += 1
        L = rnd.randint(len_min, len_max) #scelgo lunghezza a caso tra len_min e len_max
        got = place_diagonal_chain(g, L, rnd) #e provo a creare una catena 
        if got >= max(2, len_min // 2):  # considera valida solo se non troppo corta altrimenti scarta
            chains += 1
            cells += got
    #restituisce dizionario con numero di catene valide piazzate e numero totale di celle occupate da queste catene
    return {"chains": chains, "cells": cells}



# 4) Delimitatori di aree chiuse: cornici rettangolari con spessore
#disegna cornice, thick è lo spessore, bottom e rigth sono angolo in basso a dx , top left angolo alto sx
def _draw_frame(g: Grid, top: int, left: int, height: int, width: int, thick: int) -> int:
    placed = 0
    bottom = top + height - 1
    right = left + width - 1
    for t in range(thick): #cicla sullo spessore e aggiunge strati di bordo
        # orizzontali
        r_top = top + t #riga superiore
        r_bot = bottom - t #riga inferiore
        for c in range(left, right+1): #per ogni colonna c da sinistra a destra blocca la cella se è libera
            if g.is_free(r_top, c):
                g.set_blocked(r_top, c); placed += 1
            if g.is_free(r_bot, c):
                g.set_blocked(r_bot, c); placed += 1
        # verticali
        c_left = left + t
        c_right = right - t
        for r in range(top, bottom+1): #stessa cosa per i verticali
            if g.is_free(r, c_left):
                g.set_blocked(r, c_left); placed += 1
            if g.is_free(r, c_right):
                g.set_blocked(r, c_right); placed += 1
    return placed

#piazza più cornici casuali, un numero pari o inferiore a count
def place_frames(g: Grid, count: int, minw: int, minh: int, maxw: int, maxh: int, thick: int, rnd: random.Random) -> Dict[str,int]:
    frames = 0 #quante cornici già piazzate
    cells = 0 #celle totali occupate dalle cornici piazzate
    attempts = 0
    while frames < count and attempts < count * 50:
        attempts += 1
        w = rnd.randint(minw, min(maxw, g.w-2))
        h = rnd.randint(minh, min(maxh, g.h-2))
        left = rnd.randint(0, g.w - w - 1) #sceglie pos di partenza in modo che il 
        top = rnd.randint(0, g.h - h - 1) #rettangolo sia dentro la griglia
        if min(w, h) <= 2*thick: #controlla che ci sia abbastanza spazio, se troppo piccolo scarta
            continue  
        #conta le celle occupate dopo aver disegnato la cornice
        before = g.occupancy()
        _draw_frame(g, top, left, h, w, thick)
        after = g.occupancy()
        delta = after - before
        if delta > 0: #se la cornice ha occupato almeno una cella nuova la considera valida
            frames += 1
            cells += delta
    return {"frames": frames, "cells": cells}



# 5) Barre orizzontali/verticali con spessore
#sono due metodi che disegnano e piazzano le barre orizzontali e verticali, sono come muri dentro la griglia
#orient dice se la barra è verticale o orizzontale
def _draw_bar(g: Grid, r0: int, c0: int, length: int, orient: str, thick: int) -> int:
    placed = 0
    #se è orizzontale cicla sullo spessore dr e ogni volta disegna una riga parallela
    if orient == 'H':
        for dr in range(thick):
            r = r0 + dr
            for c in range(c0, min(g.w, c0 + length)):
                if g.in_bounds(r, c) and g.is_free(r, c): #controlla che la cella stia dentro la grilia la piazza solo se è ancora libera
                    g.set_blocked(r, c); placed += 1
    else:  # 'V', stesso ragionamento dell'orizzontale
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
        before = g.occupancy() #misura quante celle erano occupate prima e dopo 
        _draw_bar(g, r0, c0, L, orient, thick)
        after = g.occupancy()
        delta = after - before
        if delta > 0: #se la barra ha occupato almeno una cella la considera valida
            bars += 1
            cells += delta
    return {"bars": bars, "cells": cells} #ritorna un dizionario con numero di barre piazzate e celle occupate



# ------------------------------ GENERATION ORCHESTRATION ------------------------------
def generate(config: GridConfig) -> Tuple[Grid, Dict]:
    rnd = random.Random(config.seed) #crea generatore casuale fissato sul seed
    g = Grid(height=config.height, width=config.width) #crea griglia vuota tutta di 0

    summary: Dict[str, Dict] = {"config": asdict(config), "placed": {}} #dizionario che conterrà il riepilogo

    #se l'utente vuole ostacoli semplici li piazza
    #salva quante celle singole sono state bloccate
    if config.simple > 0:
        got = place_simple_cells(g, config.simple, rnd)
        summary["placed"]["simple"] = {"cells": got}

    #piazza i cluster, res è dizionario con numero di cluster e numero di celle piazzate
    if config.agglomerates > 0:
        res = place_agglomerates(g, config.agglomerates, config.agg_min, config.agg_max, rnd)
        summary["placed"]["agglomerates"] = res

    #piazza le catene diagonali
    if config.diagonals > 0:
        res = place_diagonals(g, config.diagonals, config.diag_min, config.diag_max, rnd)
        summary["placed"]["diagonals"] = res

    #piazza le cornici
    if config.frames > 0:
        res = place_frames(g, config.frames, config.frame_minw, config.frame_minh, config.frame_maxw, config.frame_maxh, config.frame_thick, rnd)
        summary["placed"]["frames"] = res

    #piazza le barre orizzontali e verticali
    if config.bars > 0:
        res = place_bars(g, config.bars, config.bar_min, config.bar_max, config.bar_thick, rnd)
        summary["placed"]["bars"] = res

    #dopo aver piazzato tutto, aggiunge info generali
    summary["grid"] = {
        "height": g.h,
        "width": g.w,
        "occupied_cells": g.occupancy(),
        "density": round(g.density(), 4),
    }

    return g, summary



# ------------------------------ CLI ------------------------------
#pezzo che si occupa di leggere i parametri da riga di comando
#argparse.ArgumentParser è una libreria python che serve a leggere gli argomenti che passi quando lanci il programma
def build_argparser() -> argparse.ArgumentParser: 
    p = argparse.ArgumentParser(description="Generatore di griglie con ostacoli (Compito 1)")
    p.add_argument("--width", "-W", type=int, required=True, help="larghezza griglia (colonne)")
    p.add_argument("--height", "-H", type=int, required=True, help="altezza griglia (righe)")
    p.add_argument("--seed", "-S", type=int, default=None, help="seed RNG (opzionale)")

    p.add_argument("--simple", type=int, default=0, help="numero di celle semplici") #0 = nessuna cella

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
    p.add_argument("--frame-thick", type=int, default=1) #spessore dei bordi di default a 1

    p.add_argument("--bars", type=int, default=0, help="numero di barre orizzontali/verticali")
    p.add_argument("--bar-min", type=int, default=4)
    p.add_argument("--bar-max", type=int, default=12)
    p.add_argument("--bar-thick", type=int, default=1) #spessore delle barre di default a 1

    p.add_argument("-o", "--out-dir", type=str, default=None, help="cartella di output per CSV/TXT/JSON")
    return p


def main():
    ap = build_argparser() #crea il parser con tutti i comandi possibili
    args = ap.parse_args() #legge i parametro che scrivi nel terminale e li mette in args
    #comando sotto prende i valori letti da terminale (args) e li mette dentro a un oggetto GridConfig
    #questo cfg diventa il "manuale di istruzioni" che dice al generatore cosa deve fare (x es dimensioni griglia ecc)
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

    g, summary = generate(cfg) #chiama la  funzione che costruisce la griglia 

    # stampa veloce a schermo
    print(g.to_ascii())
    print("\n-- Riepilogo --")
    print(json.dumps(summary, indent=2))

    # salvataggi opzionali
    if cfg.out_dir:
        base = os.path.join(cfg.out_dir, f"grid_{cfg.width}x{cfg.height}_seed{cfg.seed if cfg.seed is not None else 'NA'}")
        g.save_all(base, summary)
        print(f"\nFile salvati con prefisso: {base}.[csv|txt|json]")

#avvio del programma
#"se il file è eseguito come programma principale lancia main"
#serve a distinguere quando un file viene eseguito direttamente (con python grid_generator.py) da quando viene importato in un altro script
if __name__ == "__main__":
    main()
