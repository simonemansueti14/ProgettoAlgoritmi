#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox
import math
import time
import os
import json

from _1_grid_generator import GridConfig, generate
from _2_grid_analysis import compute_context_and_complement
from _3_grid_pathfinder import cammino_minimo, build_path_from_landmarks


CANVAS_SIZE = 700


class HaveFunApp:

    def __init__(self, root):
        self.root = root
        self.root.title("HaveFun - CAMMINOMIN Interactive")

        self.grid = None
        self.O = None
        self.D = None
        self.path = []
        self.stats = {}
        self.context = set()
        self.complement = set()

        self.grid_window = None
        self.canvas = None

        self.show_context = tk.BooleanVar(value=False)
        self.show_complement = tk.BooleanVar(value=False)

        self.build_interface()

    # --------------------------------------------------
    # INTERFACCIA
    # --------------------------------------------------

    def build_interface(self):

        frame = ttk.Frame(self.root, padding=10)
        frame.pack()

        ttk.Label(frame, text="Righe").grid(row=0, column=0)
        self.rows_entry = ttk.Entry(frame, width=6)
        self.rows_entry.grid(row=0, column=1)

        ttk.Label(frame, text="Colonne").grid(row=0, column=2)
        self.cols_entry = ttk.Entry(frame, width=6)
        self.cols_entry.grid(row=0, column=3)

        ttk.Label(frame, text="Origine (r,c)").grid(row=1, column=0)
        self.o_r = ttk.Entry(frame, width=5)
        self.o_c = ttk.Entry(frame, width=5)
        self.o_r.grid(row=1, column=1)
        self.o_c.grid(row=1, column=2)

        ttk.Label(frame, text="Destinazione (r,c)").grid(row=2, column=0)
        self.d_r = ttk.Entry(frame, width=5)
        self.d_c = ttk.Entry(frame, width=5)
        self.d_r.grid(row=2, column=1)
        self.d_c.grid(row=2, column=2)

        row_offset = 3
        labels = ["Semplici", "Agglomerati", "Diagonali", "Cornici", "Barre"]
        self.obstacle_entries = []

        for i, label in enumerate(labels):
            ttk.Label(frame, text=label).grid(row=row_offset+i, column=0)
            e = ttk.Entry(frame, width=6)
            e.insert(0, "0")
            e.grid(row=row_offset+i, column=1)
            self.obstacle_entries.append(e)

        ttk.Button(frame, text="Genera Griglia",
                   command=self.generate_grid)\
            .grid(row=row_offset+5, column=0, columnspan=2, pady=10)

        ttk.Button(frame, text="CALCOLA!!",
                   command=self.calculate_path)\
            .grid(row=row_offset+5, column=2, columnspan=2)

        # Switch CONTEXTO / COMPLEMENTO
        switch_frame = ttk.Frame(self.root)
        switch_frame.pack(pady=5)

        self.context_switch = ttk.Checkbutton(
            switch_frame,
            text="Mostra Contesto",
            variable=self.show_context,
            command=self.draw_grid
        )
        self.context_switch.grid(row=0, column=0, padx=10)

        self.complement_switch = ttk.Checkbutton(
            switch_frame,
            text="Mostra Complemento",
            variable=self.show_complement,
            command=self.draw_grid
        )
        self.complement_switch.grid(row=0, column=1, padx=10)

    # --------------------------------------------------
    # GENERAZIONE GRIGLIA
    # --------------------------------------------------

    def generate_grid(self):

        try:
            rows = int(self.rows_entry.get())
            cols = int(self.cols_entry.get())
            o_r = int(self.o_r.get())
            o_c = int(self.o_c.get())
            d_r = int(self.d_r.get())
            d_c = int(self.d_c.get())
        except ValueError:
            messagebox.showerror("Errore", "Inserisci numeri validi.")
            return

        if not (rows > 0 and cols > 0):
            messagebox.showerror("Errore", "Righe e Colonne devono essere > 0.")
            return

        if not (0 <= o_r < rows and 0 <= o_c < cols and
                0 <= d_r < rows and 0 <= d_c < cols):
            messagebox.showerror("Errore", "Origine/Destinazione fuori dai limiti.")
            return

        if (o_r, o_c) == (d_r, d_c):
            messagebox.showerror("Errore", "Origine e Destinazione coincidono.")
            return

        if not all(int(e.get()) >= 0 for e in self.obstacle_entries):
            messagebox.showerror("Errore", "Valori ostacoli non negativi.")
            return

        # RESET COMPLETO STATO
        self.path = []
        self.stats = {}
        self.show_context.set(False)
        self.show_complement.set(False)

        self.O = (o_r, o_c)
        self.D = (d_r, d_c)

        cfg = GridConfig(
            name="havefun",
            width=cols,
            height=rows,
            simple=int(self.obstacle_entries[0].get()),
            agglomerates=int(self.obstacle_entries[1].get()),
            diagonals=int(self.obstacle_entries[2].get()),
            frames=int(self.obstacle_entries[3].get()),
            bars=int(self.obstacle_entries[4].get())
        )

        self.grid, _ = generate(cfg)

        self.grid.cells[o_r][o_c] = 0
        self.grid.cells[d_r][d_c] = 0

        self.context, self.complement = compute_context_and_complement(self.grid, self.O)

        self.open_grid_window()

    # --------------------------------------------------
    # FINESTRA GRIGLIA
    # --------------------------------------------------

    def open_grid_window(self):

        if self.grid_window and tk.Toplevel.winfo_exists(self.grid_window):
            self.grid_window.destroy()

        self.grid_window = tk.Toplevel(self.root)
        self.grid_window.title("Visualizzazione Griglia")
        self.grid_window.geometry(f"{CANVAS_SIZE}x{CANVAS_SIZE}")

        self.grid_window.protocol("WM_DELETE_WINDOW", self.close_grid_window)

        self.canvas = tk.Canvas(self.grid_window,
                                width=CANVAS_SIZE,
                                height=CANVAS_SIZE,
                                bg="white")
        self.canvas.pack()

        self.draw_grid()

    def close_grid_window(self):
        self.path = []
        self.canvas = None
        self.grid_window.destroy()

    # --------------------------------------------------
    # DISEGNO
    # --------------------------------------------------

    def draw_grid(self):

        if not self.grid or not self.canvas:
            return

        self.canvas.delete("all")

        cell_size = min(CANVAS_SIZE / self.grid.h,
                        CANVAS_SIZE / self.grid.w)

        for r in range(self.grid.h):
            for c in range(self.grid.w):

                x1 = c * cell_size
                y1 = r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                color = "white"

                if self.grid.cells[r][c] == 1:
                    color = "black"
                elif self.show_context.get() and (r, c) in self.context:
                    color = "#b3ffcc"
                elif self.show_complement.get() and (r, c) in self.complement:
                    color = "#cce0ff"

                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color, outline="gray"
                )

        font_size = max(int(cell_size / 2), 8)

        for label, (r, c), color in [("O", self.O, "blue"), ("D", self.D, "red")]:
            x = c * cell_size + cell_size / 2
            y = r * cell_size + cell_size / 2
            self.canvas.create_text(
                x, y,
                text=label,
                font=("Arial", font_size, "bold"),
                fill=color
            )

        # Percorso rosso sottile scalabile
        if self.path:
            for i in range(len(self.path)-1):
                r1, c1 = self.path[i]
                r2, c2 = self.path[i+1]

                x1 = c1 * cell_size + cell_size/2
                y1 = r1 * cell_size + cell_size/2
                x2 = c2 * cell_size + cell_size/2
                y2 = r2 * cell_size + cell_size/2

                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="red",
                    width=max(1, cell_size/15)
                )

    # --------------------------------------------------
    # CALCOLO
    # --------------------------------------------------

    def calculate_path(self):

        if not self.grid:
            return

        start_time = time.perf_counter()
        deadline = start_time + 60

        length, seq, stats, completed = cammino_minimo(
            self.grid,
            self.O,
            self.D,
            deadline=deadline
        )

        exec_time = time.perf_counter() - start_time

        if length == math.inf:
            messagebox.showinfo("Risultato", "Cammino non esistente.")
            return

        self.path = build_path_from_landmarks(self.grid, seq)
        self.stats = stats
        self.stats["execution_time"] = exec_time
        self.stats["landmarks"] = seq
        self.stats["completed"] = completed

        self.draw_grid()
        self.show_stats_window(length)

    # --------------------------------------------------
    # STATISTICHE
    # --------------------------------------------------

    def show_stats_window(self, length):

        win = tk.Toplevel(self.root)
        win.title("Riepilogo CAMMINOMIN")
        win.geometry("600x500")

        text = tk.Text(win, font=("Courier", 10))
        text.pack(expand=True, fill="both")

        content = f"""
================ RISULTATI =================

==========================
CAMMINO COMPLETATO: {"Sì" if self.stats.get("completed", True) else "No"}
==========================

Lunghezza cammino: {length:.6f}
Tempo esecuzione:  {self.stats.get("execution_time",0):.6f} s

Frontiere visitate: {self.stats.get("frontier_count",0)}
Scelte Tipo1:       {self.stats.get("tipo1_count",0)}
Scelte Tipo2:       {self.stats.get("tipo2_count",0)}
Ricorsioni:         {self.stats.get("valorefalsoriga16",0)}

Landmarks:
{self.stats.get("landmarks")}

Cammino completo:
{self.path}
"""

        text.insert("1.0", content)
        text.config(state="disabled")

        ttk.Button(win,
                   text="Salva risultato",
                   command=lambda: self.save_result(length))\
            .pack(pady=10)

    # --------------------------------------------------
    # SALVATAGGIO
    # --------------------------------------------------

    def save_result(self, length):

        os.makedirs("havefun_results", exist_ok=True)
        timestamp = int(time.time())
        filename = f"havefun_results/havefun_results_{timestamp}.json"

        data = {
            "grid_size": [self.grid.h, self.grid.w],
            "origin": self.O,
            "destination": self.D,
            "length": length,
            "path": self.path,
            "stats": self.stats
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        messagebox.showinfo("Salvato",
                            f"File salvato in:\n{filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HaveFunApp(root)
    root.mainloop()

#piccole modifiche da apportare:
#- se clicco calcola senza aver generato la griglia dà errore -> disattiva il pulsante "CALCOLA!!" + mostra contesto e mostra complemento finché non è stata generata una griglia
#- cliccando di nuovo genera, deve chiudermi la finestra del riepilogo se è aperta, altrimenti si sovrappone e fa confusione
#- campi di input se non metto niente (vuoto) o metto valori non numerici, deve dirmi di inserire numeri validi, invece ora dà errore
#- rescaling dimensioni ostacoli con dimensioni bidimensionali (barre, frames, diagonali) -> dimensione massima almeno dim_griglia_max -1 (attualmente se metto griglia piccola con un numero di ostacoli appena altino, mi si chiude l'origine in uno spazio chiuso)
#- controllo n° tot di celle ostacolo <= nxn -2 -> altrimenti warning "troppi ostacoli inseriti".

#eventuali migliorie:
#- pulsante per accendere/spegnere landmark
#- pulsante per accendere/spegnere celle di frontiera dell'origine
#- pulsalte Salva risultati salva anche jpg griglia con O, D, landmark e cammino disegnati -> dentro havefun_results creazione di ulteriore cartella: /havefun_results_TIMESTAMP/ QUI mettere sia json che jpg
#-BONUS: generazione ostacoli con click del mouse direttamente sulla griglia.

