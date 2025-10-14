# Funzioni usate negli script

## Script 1

* in_bounds: funzione per valutare se coordinate (r,c) sono dentro la griglia
* get: funzione per valutare se cella libera (1) o vuota (0)
* set_blocker: funzione per trasformare cella in ostacolo
* is_free: funzione per valutare se cella libera
    - dipende da in_bounds
* occupancy: funzione per contare quante celle hanno ostacoli
* density: funzione per calcolare densità ostacoli/totali

## Altre funzioni di python e non

La funzione **_random_free_cell** viene usata per cercare una cella libera, prima attraverso un ciclo iterativo e successivamente passando tutti la griglia cella per cella. Se non trova nulla, ritorna None
```python
def _random_free_cell(g: Grid, rnd: random.Random) -> Cell | None
```

Funzione per piazzare ostacoli, nonché modificare valore a 1 della cella:
```python
def place_simple_cells(g: Grid, count: int, rnd: random.Random) -> int
```
La funzione **_neighbors8** restituisce le 8 celle più vicine alla data cella passata come parametro alla funzione
```python
def _neighbors8(r: int, c: int) -> List[Cell]:
    return [(r+dr, c+dc) for dr, dc in NEIGH8]
```

La funzione **place_agglomerate** viene utilizzata per la creazione di agglomerati di ostacoli, nonché

