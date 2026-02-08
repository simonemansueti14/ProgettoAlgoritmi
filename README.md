# Progetto Algoritmi e Strutture Dati (a.a. 2024/25)

Progetto sviluppato dagli studenti dell'Università degli Studi di Brescia:
- Maffezzoli Alessio a.maffezzoli001@studenti.unibs.it
- Mansueti Simone s.mansueti@studenti.unibs.it
- Spazzini Michael michaelspazzini@gmail.com



**Italiano**

**Obiettivo**

Progetto su griglie con ostacoli: generazione, analisi di raggiungibilità con cammini “liberi”, ricerca di cammini minimi con l’algoritmo CAMMINOMIN e campagna di esperimenti con confronto tra varianti.

**Contenuto**
1. `_1_grid_generator.py`: generatore di griglie binarie con diversi tipi di ostacoli (celle isolate, agglomerati, diagonali, cornici, barre). Salva CSV/TXT e stampa riepilogo.
2. `_2_grid_analysis.py`: calcolo contesto e complemento di una cella O, e distanza libera `dlib(O,D)` secondo la definizione di traccia.
3. `_3_grid_pathfinder.py`: implementazione ricorsiva di CAMMINOMIN con statistiche interne, timeout opzionale e ricostruzione del cammino completo dalle landmark.
4. `_4_grid_experiments.py`: generazione automatica griglie per esperimenti, esecuzione di CAMMINOMIN e variante, confronto O→D e D→O, grafici e JSON di riepilogo.
5. `data_extraction.py`: aggregazione dei risultati JSON degli esperimenti e produzione di grafici medi per dimensione.

**Esecuzione rapida**
Requisiti:
1. Python 3.
2. `matplotlib` e `numpy` per esperimenti e grafici.

Installazione dipendenze:
```bash
pip install -r requirements.txt
```

Generazione griglia:
```bash
python _1_grid_generator.py --width 40 --height 25 --seed 123 --simple 40 --agglomerates 6 --agg-min 3 --agg-max 8 --diagonals 4 --diag-min 3 --diag-max 10 --frames 3 --frame-minw 6 --frame-minh 4 --frame-maxw 16 --frame-maxh 10 --frame-thick 1 --bars 5 --bar-min 4 --bar-max 12 --bar-thick 1 -o out/
```
Analisi contesto/complemento:
```bash
python _2_grid_analysis.py
```
Cammino minimo (con timeout opzionale):
```bash
python _3_grid_pathfinder.py
```
Esperimenti completi (generazione, esecuzione, grafici, JSON):
```bash
python _4_grid_experiments.py
```
Aggregazione risultati esperimenti:
```bash
python data_extraction.py
```

**Output principali**
1. Griglie: CSV e TXT nella cartella scelta in output.
2. Esperimenti: `results_es4/plots/` per i grafici e `results_es4/json_outputs/` per i riepiloghi.
3. Cammini: JSON in `output_es_3/` (se abilitato).

---

**English**

**Goal**
Grid-based project: obstacle generation, reachability analysis with “free” paths, shortest path search via CAMMINOMIN, and an experimental campaign comparing variants.

**Contents**
1. `_1_grid_generator.py`: binary grid generator with multiple obstacle types (single cells, agglomerates, diagonals, frames, bars). Saves CSV/TXT and prints a summary.
2. `_2_grid_analysis.py`: computes context and complement of an origin cell O and the free distance `dlib(O,D)`.
3. `_3_grid_pathfinder.py`: recursive CAMMINOMIN implementation with internal stats, optional timeout, and full path reconstruction from landmarks.
4. `_4_grid_experiments.py`: automatic grid generation, runs CAMMINOMIN and a variant, compares O→D vs D→O, produces plots and JSON summaries.
5. `data_extraction.py`: aggregates experiment JSON results and produces average-by-size plots.

**Quick Run**
Requirements:
1. Python 3.
2. `matplotlib` and `numpy` for experiments and plots.

Install dependencies:
```bash
pip install -r requirements.txt
```

Grid generation:
```bash
python _1_grid_generator.py --width 40 --height 25 --seed 123 --simple 40 --agglomerates 6 --agg-min 3 --agg-max 8 --diagonals 4 --diag-min 3 --diag-max 10 --frames 3 --frame-minw 6 --frame-minh 4 --frame-maxw 16 --frame-maxh 10 --frame-thick 1 --bars 5 --bar-min 4 --bar-max 12 --bar-thick 1 -o out/
```
Context/complement analysis:
```bash
python _2_grid_analysis.py
```
Shortest path (optional timeout):
```bash
python _3_grid_pathfinder.py
```
Full experiments (generation, execution, plots, JSON):
```bash
python _4_grid_experiments.py
```
Experiment aggregation:
```bash
python data_extraction.py
```

**Main Outputs**
1. Grids: CSV and TXT in the chosen output directory.
1. Experiments: `results_es4/plots/` for plots and `results_es4/json_outputs/` for summaries.
1. Paths: JSON in `output_es_3/` (if enabled).


## License
This project is licensed under the GPLv3 License.
