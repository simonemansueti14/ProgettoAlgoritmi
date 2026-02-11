#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compito 4 - Esperimenti su CAMMINOMIN (versione full automatica)
Algoritmi e Strutture Dati (a.a. 2024/25)

- Carica tutte le griglie in experimental_grid/
- Esegue esperimenti CAMMINOMIN e sua variante
- Confronta direzioni O→D e D→O
- Produce riepilogo a terminale + grafico + file JSON
"""

import os, csv, math, time, json, statistics, random
from pathlib import Path
from typing import Tuple, List, Dict, Set
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

from _1_grid_generator import Grid, GridConfig, generate
from _2_grid_analysis import dlib, compute_context_and_complement
from _3_grid_pathfinder import cammino_minimo, compute_frontier

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
def cammino_minimo_variant(
    g: Grid, O: Cell, D: Cell, variant: int = 0, deadline: float = None,
    blocked: Set[Cell] = None, stats: Dict[str, int] = None, best=None
):
    if blocked is None:
        blocked = set()
    if stats is None:
        stats = {"frontier_count": 0, "tipo1_count": 0, "tipo2_count": 0, "recursions": 0}
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
    
    for F, t in frontier:
        if deadline and time.perf_counter() > deadline:
            return best[0], best[1], stats, False
    
        stats[f"tipo{t}_count"] += 1
        lF = dlib(O, F)
    
        if variant == 0:
            # Riga 16: condizione BASE (solo lF)
            pruning_threshold = lF
        else:  # variant == 1
            # Riga 17: condizione FORTE (lF + euristica)
            pruning_threshold = lF + dlib(F, D)
    
        # Usa la soglia scelta dalla variante
        if pruning_threshold >= best[0]:
            continue  #Usa CONTINUE, non BREAK (devi provare altre celle!)
        
        # Ricorsione
        stats["recursions"] += 1
        lFD, seqFD, stats, sub_completed = cammino_minimo_variant(
            g, F, D, variant, deadline, blocked.union(closure), stats, best
        )

        # Aggiorna best anche se timeout
        if lFD != math.inf and lF + lFD < best[0]:
            best = (lF + lFD, [(O, 0), (F, t)] + seqFD[1:])

        if not sub_completed:
            return best[0], best[1], stats, False

        if lFD == math.inf:
            continue

        # Aggiorna miglior cammino
        lTot = lF + lFD
        if lTot < lunghezzaMin:
            lunghezzaMin = lTot
            seqMin = [(O, 0), (F, t)] + seqFD[1:]
            best = (lunghezzaMin, seqMin)

    return lunghezzaMin, seqMin, stats, completed


#---------------------METODO PER GENERAZIONE GRIGLIE SPERIMENTALI DA ES 1----------------------

def auto_generate_all_grids(sizes: List[int], timestamp: str, repliche: int, densita: float):
    project_root = Path(__file__).parent
    input_dir = project_root / "input_es4"
    input_dir.mkdir(exist_ok=True)

    base_dir = input_dir / f"experimental_grids_es4_{timestamp}"
    base_dir.mkdir(exist_ok=True)

    obstacle_types = ["simple", "agglomerates", "diagonals", "bars", "frames"]

    print(f"Ostacoli che verranno generati per ogni griglia ({densita*100:.0f}%):\n")

    for size in sizes:
        folder = base_dir / f"{size}x{size}"
        folder.mkdir(exist_ok=True)

        budget = int(size * size * densita)
        print(f"Griglia {size}x{size} → budget celle = {budget}")

        for obstacle in obstacle_types:
            for i in range(1, repliche + 1):
                seed = random.randint(0, 99999)
                name = f"{obstacle}_{i:02d}"

                cfg_kwargs = dict(
                    name=name,
                    width=size,
                    height=size,
                    seed=seed,
                    simple=0,
                    agglomerates=0,
                    diagonals=0,
                    frames=0,
                    bars=0,
                    out_dir=str(folder)
                )

                # ===== STIMA CELLE & LIMITI =====

                if obstacle == "simple":
                    cfg_kwargs["simple"] = budget

                elif obstacle == "agglomerates":
                    agg_min, agg_max = 3, 7
                    max_clusters = budget // agg_max
                    if max_clusters == 0:
                        continue
                    cfg_kwargs.update({
                        "agglomerates": max_clusters,
                        "agg_min": agg_min,
                        "agg_max": agg_max
                    })

                elif obstacle == "diagonals":
                    diag_min, diag_max = 3, 10
                    max_chains = budget // diag_max
                    if max_chains == 0:
                        continue
                    cfg_kwargs.update({
                        "diagonals": max_chains,
                        "diag_min": diag_min,
                        "diag_max": diag_max
                    })

                elif obstacle == "bars":
                    bar_min, bar_max, thick = 4, 12, 1
                    celle_bar = bar_max * thick
                    max_bars = budget // celle_bar
                    if max_bars == 0:
                        continue
                    cfg_kwargs.update({
                        "bars": max_bars,
                        "bar_min": bar_min,
                        "bar_max": bar_max,
                        "bar_thick": thick
                    })

                elif obstacle == "frames":
                    frame_minw = max(3, size // 8)
                    frame_minh = max(3, size // 8)
                    frame_maxw = size
                    frame_maxh = size
                    thick = 1

                    perimetro_max = 2 * (frame_maxw + frame_maxh) - 4
                    max_frames = budget // perimetro_max
                    if max_frames == 0:
                        continue

                    cfg_kwargs.update({
                        "frames": max_frames,
                        "frame_minw": frame_minw,
                        "frame_minh": frame_minh,
                        "frame_maxw": frame_maxw,
                        "frame_maxh": frame_maxh,
                        "frame_thick": thick
                    })

                print(cfg_kwargs)
                cfg = GridConfig(**cfg_kwargs)
                g, summary = generate(cfg)

                base = os.path.join(
                    cfg.out_dir,
                    cfg.name if cfg.name else f"grid_{cfg.width}x{cfg.height}_seed{cfg.seed}"
                )
                g.save_all(base, summary)

        print(f"Cartella griglie {size}x{size} generata in → {folder}")

    print("\nGenerazione completata con successo!")




#Sceglie origine e destinazione più lontane possibile tra loro (buon compromesso statistico onde non riempire l'experimental_params.json a mano per centinaia di griglie)
def choose_origin_and_dest(g: Grid) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Trova origine (in alto a sinistra) e destinazione (in basso a destra) libere.
       - Origine: scorre a destra, poi va a capo in basso.
       - Destinazione: scorre a sinistra, poi va a capo verso l’alto.
    """
    h, w = g.h, g.w
    origin = [0, 0]
    dest = [h - 1, w - 1]

    max_iters = h * w * 2
    iters = 0

    while (not g.is_free(origin[0], origin[1])) or (not g.is_free(dest[0], dest[1])):
        #sposta l’origine (scorri a destra poi vai a capo in basso)
        if not g.is_free(origin[0], origin[1]):
            origin[1] += 1  # vai a destra
            if origin[1] >= w:  # se fine riga
                origin[1] = 0
                origin[0] = (origin[0] + 1) % h  # vai giù di una riga
        #sposta la destinazione (scorri a sinistra poi vai a capo verso l’alto)
        if not g.is_free(dest[0], dest[1]):
            dest[1] -= 1  # vai a sinistra
            if dest[1] < 0:  # se inizio riga
                dest[1] = w - 1
                dest[0] = (dest[0] - 1) % h  # vai su di una riga

        iters += 1
        if iters > max_iters:
            raise RuntimeError("Impossibile trovare celle libere per origine e destinazione")

    return tuple(origin), tuple(dest)


# ---------------------------------- ESPERIMENTI AUTOMATICI ----------------------------------
def experiment(g: Grid, O: Cell, D: Cell, trials:int=3, variant:int=0, deadline:float=math.inf) -> Dict:
    results = {}
    for direction, label in [((O, D), "OtoD"), ((D, O), "DtoO")]:
        O_, D_ = direction
        lengths, times = [], []
        frontier_counts, tipo1_counts, tipo2_counts, recursions = [], [], [], []

        print(label)
        for i in range(1, trials + 1):
            start = time.perf_counter()
            deadline_assoluta = start+deadline
            length, _, stats, completed = cammino_minimo_variant(g, O_, D_, variant, deadline_assoluta)
            elapsed = time.perf_counter() - start

            print(f"trial #{i} - tempo: {elapsed:.4f}s | frontiere={stats['frontier_count']} | tipo1={stats['tipo1_count']} | tipo2={stats['tipo2_count']} | Ricorsioni effettuate: {stats['recursions']} | Completato: {completed} | Lunghezza Min: {length}")

            lengths.append(length)
            times.append(elapsed)
            frontier_counts.append(stats["frontier_count"])
            tipo1_counts.append(stats["tipo1_count"])
            tipo2_counts.append(stats["tipo2_count"])
            recursions.append(stats["recursions"])

        results[label] = {
            "avg_length": statistics.mean(lengths),
            "avg_time": statistics.mean(times),
            "avg_frontier": statistics.mean(frontier_counts),
            "avg_tipo1": statistics.mean(tipo1_counts),
            "avg_tipo2": statistics.mean(tipo2_counts),
            "recursions": statistics.mean(recursions),
            "valid": completed,
            "variant": variant
        }
        #print(" completata")

    return results


def run_experiments(grid_dir: Path, variant:int, trials:int=3, deadline:float=math.inf) -> Dict:
    summary = {}
    for grid_csv in grid_dir.glob("*.csv"):
        grid_name = grid_csv.stem
        print(f"\nCarico {grid_name} ...")
        g = load_grid_from_csv(grid_csv)

        # 🔹 Scegli automaticamente origine e destinazione
        origin, dest = choose_origin_and_dest(g)
        print(f"scelte automatiche O e D:  O={origin}, D={dest}")

        res = experiment(g, origin, dest, trials=trials, variant=variant, deadline=deadline)
        summary[grid_name] = res
    return summary

# ---------------------------------- VISUALIZZAZIONE RISULTATI ----------------------------------
def summarize_results(summary: Dict):
    print("\n=== RIEPILOGO ESPERIMENTI ===")
    for gname, res in summary.items():
        print(f"\n{gname}")
        for direction, vals in res.items():
            print(f"  {direction}: distanza min={vals['avg_length']:.2f}, tempo medio={vals['avg_time']:.3f}s, completato={vals['valid']}, variant={vals['variant']}, Ricorsioni effettuate={vals['recursions']}")

#---------------------------------------------PLOT PRESTAZIONI TEMPORALI---------------------------------------------
def plot_results(summary: Dict, variant:int, dim:int, save_dir: Path | None=None):
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

    if save_dir:
        fname = save_dir / f"tempi_variant{variant}_{dim}x{dim}.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Grafico tempi salvato in: {fname.name}")

    plt.show()

#--------------------------------------------- PLOT PRESTAZIONI SPAZIALI ---------------------------------------------
def plot_stats(summary: Dict, variant:int, dim:int, save_dir:Path | None=None):
    labels = list(summary.keys())
    frontiere_OtoD, frontiere_DtoO, tipo1, tipo2, recursions_OtoD, recursions_DtoO = [], [], [], [], [], []
    for gname, res in summary.items():
        #append valori al grafico
        frontiere_OtoD.append(res["OtoD"]["avg_frontier"])
        frontiere_DtoO.append(res["DtoO"]["avg_frontier"])
        tipo1.append((res["OtoD"]["avg_tipo1"] + res["DtoO"]["avg_tipo1"]) / 2)
        tipo2.append((res["OtoD"]["avg_tipo2"] + res["DtoO"]["avg_tipo2"]) / 2)
        recursions_OtoD.append(res["OtoD"]["recursions"])
        recursions_DtoO.append(res["DtoO"]["recursions"])

    x = np.arange(len(labels))
    width = 0.13  # larghezza di ogni barra

    plt.figure(figsize=(10, 5))
    plt.bar(x - 2.5*width, frontiere_OtoD, width=width, label="Frontiere individuate O→D")
    plt.bar(x - 1.5*width, frontiere_DtoO, width=width, label="Frontiere individuate D→O")
    plt.bar(x - 0.5*width, tipo1, width=width, label="Scelte Tipo 1 (media O→D D→O)")
    plt.bar(x + 0.5*width, tipo2, width=width, label="Scelte Tipo 2 (media O→D D→O)")
    plt.bar(x + 1.5*width, recursions_OtoD, width=width, label="ricorsioni algoritmo effettuate (O→D)")
    plt.bar(x + 2.5*width, recursions_DtoO, width=width, label="ricorsioni algoritmo effettuate (D→O)")
    

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Valori")
    plt.title(f"Statistiche interne - variante: {variant} - griglie {dim}x{dim}")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        fname = save_dir / f"stats_variant{variant}_{dim}x{dim}.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Grafico statistiche salvato in: {fname.name}")
    else:
        plt.show()


    # ---------------- converte math.inf in "inf" per errori JSON ------------------
def make_json_safe(obj):
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    else:
        return obj

# ---------------------------------- MAIN ----------------------------------
def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #GENERAZIONE GRIGLIE
    #---- QUI METTERE LA LISTA DI DIMENSIONI NXN CHE SI VOGLIONO GENERARE

    setDeadline = None
    sizes = []
    print("Inserisci la dim n delle griglie n×n (INVIO per terminare):")

    while True:
        val = input("n = ").strip()
        if val == "":
            break
        if not val.isdigit() or int(val) <= 0:
            print("Inserisci un intero positivo!")
            continue
        sizes.append(int(val))

    if not sizes:
        print("Nessuna dimensione inserita")
        return
    
    # gestione repliche
    print("Inserire il numero di repliche:")

    while True:
        repliche_str = input("repliche = ").strip()

        if repliche_str == "":
            repliche = None      
            break

        if not repliche_str.isdigit():
            print("Inserisci un intero positivo!")
            continue

        repliche = int(repliche_str)

        if repliche <= 0:
            print("Inserisci un intero positivo!")
            continue
        break
    
     # gestione densità
    print("Inserire densità ostacoli (compresa tra 0 e 1):")

    while True:
        densita_str = input("densita = ").strip()
        if densita_str == "":
            break
        try:
            densita = float(densita_str)
        except ValueError:
            print("Inserisci un numero valido (es. 0.2, 0.5, 1.0)")
            continue

        if not (0 < densita <= 1):
            print("La densità deve essere compresa tra 0 e 1!")
            continue
        break

    # deadline
    while True:
        dl = input("Inserisci deadline (secondi): ").strip()
        try:
            setDeadline = float(dl)
            if setDeadline > 0:
                break
        except ValueError:
            pass
        print("Inserisci un numero positivo!")

    #== GENERAZIONE EFFETTIVA GRIGLIE ==
    auto_generate_all_grids(sizes, timestamp, repliche, densita)

    #cartelle griglie
    project_root = Path(__file__).parent
    input_dir = project_root / "input_es4"
    grid_dir = input_dir / f"experimental_grids_es4_{timestamp}"

    #param_file = base_dir / "experimental_params.json"

    #cartelle plot
    plots_dir = Path(__file__).parent / "results_es4" / "plots" / f"{timestamp}"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not grid_dir.exists():
        print(f" Cartella {grid_dir}/ non trovata.")
        return

    combined_summary = {
        "variant_0": {},
        "variant_1": {}
    }

    # === SCORRI TUTTE LE SOTTOCARTELLE (dimensioni griglia) ===
    for size_folder in grid_dir.iterdir():
        if not size_folder.is_dir():
            continue

        print(f"\n=== ESECUZIONE ESPERIMENTI SU {size_folder}/ ===")

        dim = int(size_folder.name.split('x')[0])

        # === VARIANTE 0 ===
        print("\n--- VARIANTE = 0 ---")
        summary_var0 = run_experiments(size_folder, variant=0, trials=5, deadline=setDeadline)
        summarize_results(summary_var0)
        plot_results(summary_var0, 0, dim, save_dir=plots_dir)
        plot_stats(summary_var0, 0, dim, save_dir=plots_dir)
        combined_summary["variant_0"][size_folder.name] = make_json_safe(summary_var0)

        # === VARIANTE 1 ===
        print("\n--- VARIANTE = 1 ---")
        summary_var1 = run_experiments(size_folder, variant=1, trials=5, deadline=setDeadline)
        summarize_results(summary_var1)
        plot_results(summary_var1, 1, dim, save_dir=plots_dir)
        plot_stats(summary_var1, 1, dim, save_dir=plots_dir)
        combined_summary["variant_1"][size_folder.name] = make_json_safe(summary_var1)

    # === SCRIVE I RISULTATI FINALI ===
    json_out_dir = Path(__file__).parent / "results_es4" / "json_outputs"
    json_out_dir.mkdir(exist_ok=True)

    output_file = json_out_dir / f"experiments_output_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)

    print(f"\nRisultati salvati in {output_file.name}")

if __name__ == "__main__":
    main()
