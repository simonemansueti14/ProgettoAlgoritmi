import json
from pathlib import Path
import matplotlib.pyplot as plt

FIELDS = ["avg_time", "avg_frontier", "avg_tipo1", "avg_tipo2", "recursions"]

def extract_global_averages(json_dir: Path):
    """
    Ritorna un dizionario del tipo:
    results[variant][field][dimension] = valore medio
    """
    results = {
        0: {f: {} for f in FIELDS},
        1: {f: {} for f in FIELDS}
    }

    for json_file in sorted(json_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for variant_key, variant_data in data.items():
            variant = 0 if variant_key == "variant_0" else 1

            for dim_key, grids in variant_data.items():
                n = int(dim_key.split("x")[0])

                accum = {f: [] for f in FIELDS}

                for _, runs in grids.items():
                    for direction in ["OtoD", "DtoO"]:
                        for f in FIELDS:
                            accum[f].append(runs[direction][f])

                for f in FIELDS:
                    results[variant][f][n] = sum(accum[f]) / len(accum[f])

    return results

def plot_average_metric(results, field, ylabel, filename):
    base_dir = Path(__file__).parent / "results_es4" / "plots" / "average"
    base_dir.mkdir(parents=True, exist_ok=True)

    dims = sorted(results[0][field].keys())
    v0 = [results[0][field][n] for n in dims]
    v1 = [results[1][field][n] for n in dims]

    plt.figure()
    plt.plot(dims, v0, marker="o", label="Variante 0")
    plt.plot(dims, v1, marker="s", label="Variante 1")
    plt.xlabel("Dimensione griglia (n)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    out = base_dir / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Salvato: {out}")

if __name__ == "__main__":
    json_dir = Path(__file__).parent / "results_es4" / "json_outputs"
    results = extract_global_averages(json_dir)

    plot_average_metric(
        results,
        field="avg_time",
        ylabel="Tempo medio globale (s)",
        filename="avg_time_vs_dimension.png"
    )

    plot_average_metric(
        results,
        field="avg_frontier",
        ylabel="Numero medio celle di frontiera",
        filename="avg_frontier_vs_dimension.png"
    )

    plot_average_metric(
        results,
        field="avg_tipo1",
        ylabel="Numero medio scelte tipo 1",
        filename="avg_tipo1_vs_dimension.png"
    )

    plot_average_metric(
        results,
        field="avg_tipo2",
        ylabel="Numero medio scelte tipo 2",
        filename="avg_tipo2_vs_dimension.png"
    )

    plot_average_metric(
        results,
        field="recursions",
        ylabel="Numero medio ricorsioni",
        filename="avg_recursions_vs_dimension.png"
    )
