"""Tablas, rankings y graficos para comparar experimentos."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_ALGORITHM, DEFAULT_RANK_METRIC, MINIMIZE_METRICS
from evaluation import METRIC_KEYS

DISPLAY_NAMES = {
    "accuracy": "Acc",
    "balanced_accuracy": "BalAcc",
    "precision_macro": "Prec_m",
    "recall_macro": "Rec_m",
    "f1_macro": "F1_m",
    "mae_ordinal": "MAE",
    "qwk": "QWK",
    "accuracy_pm1": "Acc±1",
    "errores_graves": "Err≥2",
}


def resolve_rank_mode(metric: str, mode: str) -> str:
    """Resuelve max/min. auto minimiza MAE y errores graves."""

    if mode == "auto":
        return "min" if metric in MINIMIZE_METRICS else "max"
    if mode not in {"max", "min"}:
        raise ValueError("rank-mode debe ser max, min o auto.")
    return mode


def experiment_to_row(
    results: dict, algorithm: str = DEFAULT_ALGORITHM
) -> dict:
    """Convierte el resultado de un experimento en una fila comparable."""

    row = {
        "algorithm": algorithm,
        "target": results["target_name"],
        "hidden_dim": results["config"]["hidden_dim"],
        "dropout": results["config"]["dropout"],
        "learning_rate": results["config"]["learning_rate"],
        "weight_decay": results["config"]["weight_decay"],
        "epochs": results["config"]["epochs"],
        "outer_folds": results["config"]["outer_folds"],
        "inner_folds": results["config"]["inner_folds"],
    }
    for metric_name in METRIC_KEYS:
        row[metric_name] = results["summary"][f"mean_{metric_name}"]
        row[f"{metric_name}_std"] = results["summary"][f"std_{metric_name}"]
    return row


def rank_rows(rows: list[dict], metric: str, mode: str) -> list[dict]:
    """Ordena filas por una metrica. mode es max o min."""

    if metric not in METRIC_KEYS:
        raise ValueError(
            f"Metrica de ranking invalida: {metric}. Use una de {METRIC_KEYS}."
        )
    descending = mode == "max"
    return sorted(rows, key=lambda row: row[metric], reverse=descending)


def _format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} ± {std_value:.4f}"


def markdown_results_table(rows: list[dict]) -> str:
    """Tabla completa algoritmo × objetivo × metricas."""

    headers = ["Algoritmo", "Objetivo"] + [DISPLAY_NAMES[key] for key in METRIC_KEYS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [str(row["algorithm"]), str(row["target"])]
        for key in METRIC_KEYS:
            cells.append(_format_mean_std(row[key], row[f"{key}_std"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def markdown_ranking_table(
    ranked_rows: list[dict], metric: str, mode: str
) -> str:
    """Ranking con el mejor resultado destacado."""

    direction = "maximizar" if mode == "max" else "minimizar"
    title = (
        f"# Ranking por {DISPLAY_NAMES.get(metric, metric)} "
        f"({direction})\n\n"
    )
    headers = [
        "Puesto",
        "Algoritmo",
        "Objetivo",
        DISPLAY_NAMES.get(metric, metric),
        "Acc",
        "Prec_m",
        "Rec_m",
        "F1_m",
        "MAE",
        "Err≥2",
    ]
    lines = [
        title,
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for index, row in enumerate(ranked_rows, start=1):
        algorithm = row["algorithm"]
        target = row["target"]
        metric_cell = _format_mean_std(row[metric], row[f"{metric}_std"])
        if index == 1:
            algorithm = f"**{algorithm}**"
            target = f"**{target}**"
            metric_cell = f"**{metric_cell}**"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    algorithm,
                    target,
                    metric_cell,
                    f"{row['accuracy']:.4f}",
                    f"{row['precision_macro']:.4f}",
                    f"{row['recall_macro']:.4f}",
                    f"{row['f1_macro']:.4f}",
                    f"{row['mae_ordinal']:.4f}",
                    f"{row['errores_graves']:.4f}",
                ]
            )
            + " |"
        )
    if ranked_rows:
        best = ranked_rows[0]
        lines.append("")
        lines.append(
            f"Mejor resultado: {best['algorithm']} en {best['target']} "
            f"({DISPLAY_NAMES.get(metric, metric)} = {best[metric]:.4f})."
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def plot_metric_bars(
    rows: list[dict], metric: str, output_path: Path, ylabel: str
) -> None:
    """Barras agrupadas por objetivo y algoritmo."""

    targets = list(dict.fromkeys(row["target"] for row in rows))
    algorithms = list(dict.fromkeys(row["algorithm"] for row in rows))
    x = np.arange(len(targets))
    width = 0.8 / max(len(algorithms), 1)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    for index, algorithm in enumerate(algorithms):
        values = []
        errors = []
        for target in targets:
            match = next(
                (
                    row
                    for row in rows
                    if row["algorithm"] == algorithm and row["target"] == target
                ),
                None,
            )
            values.append(0.0 if match is None else match[metric])
            errors.append(0.0 if match is None else match[f"{metric}_std"])
        axis.bar(
            x + index * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=algorithm,
        )

    axis.set_xticks(x + width * (len(algorithms) - 1) / 2)
    axis.set_xticklabels(targets, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(ylabel)
    axis.legend()
    axis.grid(axis="y", linestyle=":", alpha=0.4)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_ranking_bars(
    ranked_rows: list[dict], metric: str, mode: str, output_path: Path
) -> None:
    """Barras horizontales del ranking activo."""

    labels = [f"{row['algorithm']} / {row['target']}" for row in ranked_rows]
    values = [row[metric] for row in ranked_rows]
    figure, axis = plt.subplots(figsize=(8, max(3.0, 0.45 * len(ranked_rows) + 1.2)))
    colors = ["#B51700" if index == 0 else "#0076BA" for index in range(len(values))]
    axis.barh(range(len(values)), values, color=colors)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    direction = "mayor es mejor" if mode == "max" else "menor es mejor"
    axis.set_xlabel(f"{DISPLAY_NAMES.get(metric, metric)} ({direction})")
    axis.set_title(f"Ranking por {DISPLAY_NAMES.get(metric, metric)}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray, title: str, output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xlabel("Predicho")
    axis.set_ylabel("Real")
    axis.set_title(title)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                str(int(matrix[row_index, col_index])),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_experiment_reports(
    rows: list[dict],
    output_dir: str | Path,
    rank_metric: str = DEFAULT_RANK_METRIC,
    rank_mode: str = "auto",
    confusion_by_target: dict[str, np.ndarray] | None = None,
) -> dict[str, Path]:
    """Escribe CSV, Markdown y PNG. Devuelve las rutas generadas."""

    if not rows:
        raise ValueError("No hay filas de resultados para reportar.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mode = resolve_rank_mode(rank_metric, rank_mode)

    csv_fields = [
        "algorithm",
        "target",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "epochs",
        "outer_folds",
        "inner_folds",
        *METRIC_KEYS,
        *[f"{key}_std" for key in METRIC_KEYS],
    ]

    ranking_f1 = rank_rows(rows, "f1_macro", "max")
    ranking_mae = rank_rows(rows, "mae_ordinal", "min")
    ranking_custom = rank_rows(rows, rank_metric, mode)

    paths = {
        "resultados_csv": output_path / "resultados.csv",
        "resultados_md": output_path / "resultados.md",
        "ranking_f1_csv": output_path / "ranking_f1.csv",
        "ranking_f1_md": output_path / "ranking_f1.md",
        "ranking_mae_csv": output_path / "ranking_mae.csv",
        "ranking_mae_md": output_path / "ranking_mae.md",
        "ranking_custom_csv": output_path / f"ranking_{rank_metric}.csv",
        "ranking_custom_md": output_path / f"ranking_{rank_metric}.md",
        "plot_f1": output_path / "barras_f1.png",
        "plot_mae": output_path / "barras_mae.png",
        "plot_ranking": output_path / f"ranking_{rank_metric}.png",
    }

    write_csv(paths["resultados_csv"], rows, csv_fields)
    write_text(paths["resultados_md"], "# Resultados\n\n" + markdown_results_table(rows))
    write_csv(paths["ranking_f1_csv"], ranking_f1, csv_fields)
    write_text(paths["ranking_f1_md"], markdown_ranking_table(ranking_f1, "f1_macro", "max"))
    write_csv(paths["ranking_mae_csv"], ranking_mae, csv_fields)
    write_text(
        paths["ranking_mae_md"],
        markdown_ranking_table(ranking_mae, "mae_ordinal", "min"),
    )
    write_csv(paths["ranking_custom_csv"], ranking_custom, csv_fields)
    write_text(
        paths["ranking_custom_md"],
        markdown_ranking_table(ranking_custom, rank_metric, mode),
    )
    plot_metric_bars(rows, "f1_macro", paths["plot_f1"], "F1 macro")
    plot_metric_bars(rows, "mae_ordinal", paths["plot_mae"], "MAE ordinal")
    plot_ranking_bars(ranking_custom, rank_metric, mode, paths["plot_ranking"])

    if confusion_by_target:
        for target_name, matrix in confusion_by_target.items():
            confusion_path = output_path / f"confusion_{target_name}.png"
            plot_confusion_matrix(
                np.asarray(matrix),
                f"Matriz de confusion ({target_name})",
                confusion_path,
            )
            paths[f"confusion_{target_name}"] = confusion_path

    return paths


def format_ranking_console(ranked_rows: list[dict], metric: str, mode: str) -> str:
    """Version compacta del ranking para imprimir en consola."""

    direction = "max" if mode == "max" else "min"
    lines = [
        f"Ranking {DISPLAY_NAMES.get(metric, metric)} ({direction}):",
    ]
    for index, row in enumerate(ranked_rows, start=1):
        marker = " <- mejor" if index == 1 else ""
        lines.append(
            f"  {index}. {row['algorithm']} / {row['target']}: "
            f"{row[metric]:.4f} ± {row[f'{metric}_std']:.4f}{marker}"
        )
    return "\n".join(lines)
