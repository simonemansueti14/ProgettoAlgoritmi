# Progetto algoritmi e struttura dati
Il progetto è stato sviluppato da Mansueti Simone, Spazzini Michael e Maffezzoli Alessio.

## Compito 1
Realizzare un generatore che sia in grado di produrre griglie finite contenenti:

* celle non attraversabili disposte secondo almeno 4 tipologie di ostacolo:
    + **Semplice**
    + **Agglomerato**: ostacolo che non può essere scavalcato
    + **Diagonale**: ostacoli che si toccano solo sullo spigolo e dunque scavalcabili solo diagonalmente
    + **Delimitatori di aree chiuse**: ostacoli che realizzano un area chiusa
    + **A barre**: ostacoli/muri verticali o orizzontali
* Un griglia può contenere una o più tipologie di ostacoli contemporaneamente
* L’utente deve poter configurare la dimensione della griglia desiderata e altri parametri, ritenuti significativi (anche ai fini della sperimentazione) che guidino il generatore (semi-) casuale.

Command line di lancio per generazione griglia 40x25 con solo 2 ostacoli semplici

```
grid_generator.py --width 40 --height 25 --seed 0 --simple 2 --agglomerates 0 --agg-min 0 --agg-max 0 --diagonals 0 --diag-min 0 --diag-max 4 --frames 0 --frame-minw 0 --frame-minh 0 --frame-maxw 0 --frame-maxh 0 --frame-thick 0 --bars 0 --bar-min 0 --bar-max 0 --bar-thick 0
```

## Compito 2
- Scrivere lo pseudocodice di un algoritmo che, data una griglia finita e una cella O della stessa, calcoli il contesto ed il complemento di O.
Costruire un’applicazione software che implementi tale algoritmo, acquisendo in ingresso una griglia prodotta dal generatore di cui al Compito 1
- La medesima applicazione deve essere in grado di calcolare la distanza libera fra due celle O e D.