import argparse
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    DEFAULT_ALGORITHM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_GDS_INNER_FOLDS,
    DEFAULT_GDS_OUTER_FOLDS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INNER_FOLDS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RANK_METRIC,
    DEFAULT_RANK_MODE,
    DEFAULT_TARGET,
    DEFAULT_WEIGHT_DECAY,
    HYPERPARAMETER_GRID,
    TARGET_COLUMNS,
)
from data_loader import CognitiveDataset, load_dataframe  # noqa: E402
from evaluation import (  # noqa: E402
    METRIC_KEYS,
    compute_all_metrics,
    compute_confusion_matrix,
    format_classification_report,
)
from models import ShallowMultiClassNet, MLPCoral  # noqa: E402
from losses import coral_loss, effective_number_weights  # noqa: E402
from ordinal import logits_to_ordinal_predictions  # noqa: E402
from preprocessing import prepare_experiment_data, split_for_validation  # noqa: E402
from reporting import (  # noqa: E402
    experiment_to_row,
    format_ranking_console,
    rank_rows,
    resolve_rank_mode,
    save_experiment_reports,
)


def build_project_objects(
    data_path: str | Path,
    target_name: str = DEFAULT_TARGET,
) -> dict:
    """Construye los objetos base de datos del experimento."""

    dataframe = load_dataframe(data_path)
    X, y, classes, class_to_idx = prepare_experiment_data(dataframe, target_name)

    return {
        "dataframe": dataframe,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "X": X,
        "y": y,
        "X_shape": X.shape,
        "y_shape": y.shape,
    }


def set_seed(seed: int) -> None:
    """Fija semillas para que la ejecucion sea reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resuelve el dispositivo solicitado por linea de comandos."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se solicito CUDA, pero no hay GPU disponible.")
    return device


def build_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Construye un DataLoader simple para entrenamiento o evaluacion."""
    dataset = CognitiveDataset(X, y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
) -> float:
    """Entrena una epoca con la funcion de perdida provista."""
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_coral: bool = False,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evalua el modelo y devuelve metricas, etiquetas y predicciones."""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            
            if is_coral:
                predictions = logits_to_ordinal_predictions(logits).cpu().numpy()
            else:
                predictions = logits.argmax(dim=1).cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(targets.numpy())

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_targets)
    return compute_all_metrics(y_true, y_pred), y_true, y_pred


def run_training_cycle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    num_classes: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
    is_coral: bool = False,
    use_weights: bool = False,
    beta: float = 0.99,
) -> dict:
    """Entrena y evalua una configuracion puntual del modelo."""
    set_seed(seed)

    train_loader = build_data_loader(
        X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed
    )
    eval_loader = build_data_loader(
        X_eval, y_eval, batch_size=batch_size, shuffle=False, seed=seed
    )

    if is_coral:
        model = MLPCoral(
            num_features=X_train.shape[1],
            num_classes=num_classes,
            dropout=dropout,
        ).to(device)
    else:
        model = ShallowMultiClassNet(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            dropout=dropout,
            output_dim=num_classes,
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    class_weights_tensor = None
    if use_weights:
        class_weights_tensor = effective_number_weights(y_train, num_classes, beta).to(device)

    def criterion(logits, targets):
        if is_coral:
            return coral_loss(logits, targets, num_classes, class_weights_tensor)
        else:
            ce_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
            return ce_loss(logits, targets)

    history = []
    for _ in range(epochs):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        history.append(epoch_loss)

    metrics, y_true, y_pred = evaluate_model(model, eval_loader, device=device, is_coral=is_coral)

    return {
        "model": model,
        "train_losses": history,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "final_train_loss": history[-1],
    }


def train_one_experiment(
    data_path: str | Path,
    target_name: str = DEFAULT_TARGET,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    seed: int = DEFAULT_RANDOM_SEED,
    device_name: str = "cpu",
    is_coral: bool = False,
    use_weights: bool = False,
    beta: float = 0.99,
) -> dict:
    """Ejecuta el flujo de validacion anidada con busqueda de hiperparametros."""
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor o igual que 1.")
    if epochs < 1:
        raise ValueError("epochs debe ser mayor o igual que 1.")
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds e inner_folds deben ser al menos 2.")

    artifacts = build_project_objects(data_path=data_path, target_name=target_name)
    X = artifacts["X"]
    y = artifacts["y"]
    num_classes = len(artifacts["classes"])
    device = resolve_device(device_name)

    outer_splits = split_for_validation(y, n_splits=outer_folds, random_state=seed)
    outer_results = []

    for outer_fold_index, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
        X_outer_train = X[outer_train_idx]
        y_outer_train = y[outer_train_idx]
        X_outer_test = X[outer_test_idx]
        y_outer_test = y[outer_test_idx]

        if can_make_stratified_splits(y_outer_train, inner_folds):
            best_config = None
            best_mae = float("inf")
            best_qwk = -float("inf")

            for config in HYPERPARAMETER_GRID:
                # Agregar beta a la configuración si usamos pesos para documentarlo en reportes
                if use_weights:
                    config = dict(config)
                    config["beta"] = beta

                inner_mae_scores = []
                inner_qwk_scores = []
                inner_splits = split_for_validation(
                    y_outer_train,
                    n_splits=inner_folds,
                    random_state=seed + outer_fold_index,
                )

                for inner_fold_index, (inner_train_idx, inner_val_idx) in enumerate(inner_splits, start=1):
                    inner_result = run_training_cycle(
                        X_train=X_outer_train[inner_train_idx],
                        y_train=y_outer_train[inner_train_idx],
                        X_eval=X_outer_train[inner_val_idx],
                        y_eval=y_outer_train[inner_val_idx],
                        num_classes=num_classes,
                        hidden_dim=config["hidden_dim"],
                        dropout=config["dropout"],
                        learning_rate=config["learning_rate"],
                        weight_decay=config["weight_decay"],
                        batch_size=batch_size,
                        epochs=epochs,
                        seed=seed + outer_fold_index * 100 + inner_fold_index,
                        device=device,
                        is_coral=is_coral,
                        use_weights=use_weights,
                        beta=beta,
                    )
                    inner_mae_scores.append(inner_result["metrics"]["mae_ordinal"])
                    inner_qwk_scores.append(inner_result["metrics"]["qwk"])

                avg_mae = float(np.mean(inner_mae_scores))
                avg_qwk = float(np.mean(inner_qwk_scores))

                if avg_mae < best_mae or (avg_mae == best_mae and avg_qwk > best_qwk):
                    best_mae = avg_mae
                    best_qwk = avg_qwk
                    best_config = config
                    best_inner_mae_scores = inner_mae_scores
                    best_inner_qwk_scores = inner_qwk_scores
            
            inner_mae_scores = best_inner_mae_scores
            inner_qwk_scores = best_inner_qwk_scores
        else:
            print(
                f"Aviso: el fold externo {outer_fold_index} de {target_name} "
                f"no admite {inner_folds} folds internos estratificados "
                "(clase rara). Se omite la validacion interna en este fold."
            )
            best_config = {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            }
            if use_weights:
                best_config["beta"] = beta

        final_result = run_training_cycle(
            X_train=X_outer_train,
            y_train=y_outer_train,
            X_eval=X_outer_test,
            y_eval=y_outer_test,
            num_classes=num_classes,
            hidden_dim=best_config["hidden_dim"],
            dropout=best_config["dropout"],
            learning_rate=best_config["learning_rate"],
            weight_decay=best_config["weight_decay"],
            batch_size=batch_size,
            epochs=epochs,
            seed=seed + outer_fold_index * 1000,
            device=device,
            is_coral=is_coral,
            use_weights=use_weights,
            beta=beta,
        )

        outer_results.append(
            {
                "outer_fold": outer_fold_index,
                "inner_mae_mean": float(np.mean(inner_mae_scores)) if inner_mae_scores else float("nan"),
                "inner_mae_std": float(np.std(inner_mae_scores)) if inner_mae_scores else float("nan"),
                "inner_qwk_mean": float(np.mean(inner_qwk_scores)) if inner_qwk_scores else float("nan"),
                "inner_qwk_std": float(np.std(inner_qwk_scores)) if inner_qwk_scores else float("nan"),
                "outer_metrics": final_result["metrics"],
                "y_true": final_result["y_true"],
                "y_pred": final_result["y_pred"],
                "final_train_loss": final_result["final_train_loss"],
                "best_config": best_config,
            }
        )

    summary = {}
    for metric_name in METRIC_KEYS:
        values = np.asarray(
            [fold_result["outer_metrics"][metric_name] for fold_result in outer_results],
            dtype=np.float64,
        )
        summary[f"mean_{metric_name}"] = float(values.mean())
        summary[f"std_{metric_name}"] = float(values.std())

    last_fold = outer_results[-1]
    
    # Determinar el nombre del algoritmo para la tabla de reporte
    algorithm_name = "Softmax (HP)"
    if is_coral:
        algorithm_name = "CORAL + pesos" if use_weights else "CORAL"

    return {
        "target_name": target_name,
        "classes": artifacts["classes"],
        "X_shape": artifacts["X_shape"],
        "y_shape": artifacts["y_shape"],
        "device": str(device),
        "config": {
            **last_fold["best_config"],  # Reportar la mejor configuración encontrada
            "batch_size": batch_size,
            "epochs": epochs,
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "seed": seed,
        },
        "outer_folds": outer_results,
        "summary": summary,
        "last_fold_report": format_classification_report(
            last_fold["y_true"],
            last_fold["y_pred"],
            class_names=artifacts["classes"],
        ),
        "last_fold_confusion": compute_confusion_matrix(
            last_fold["y_true"], last_fold["y_pred"]
        ),
        "algorithm": algorithm_name,
    }


def can_make_stratified_splits(y: np.ndarray, n_splits: int) -> bool:
    """True si StratifiedKFold puede respetar n_splits."""
    labels = np.asarray(y).ravel()
    unique_labels, counts = np.unique(labels, return_counts=True)
    return len(unique_labels) >= 2 and int(counts.min()) >= n_splits


def folds_for_target(
    target_name: str,
    outer_folds: int,
    inner_folds: int,
    gds_outer_folds: int,
    gds_inner_folds: int,
    all_targets: bool,
) -> tuple[int, int]:
    """Con --all-targets, GDS usa folds reducidos para la estratificacion."""
    if all_targets and target_name == "GDS":
        return gds_outer_folds, gds_inner_folds
    return outer_folds, inner_folds


def print_experiment_results(results: dict) -> None:
    """Imprime el resumen de un experimento puntual."""
    print("Experimento ejecutado correctamente.")
    print(f"Algoritmo: {results.get('algorithm', DEFAULT_ALGORITHM)}")
    print(f"Experimento activo: {results['target_name']}")
    print(f"Dispositivo usado: {results['device']}")
    print(f"Forma de X: {results['X_shape']}")
    print(f"Forma de y: {results['y_shape']}")
    print(f"Clases encontradas: {results['classes']}")
    print(
        "Folds: "
        f"outer={results['config']['outer_folds']}, "
        f"inner={results['config']['inner_folds']}"
    )
    print("Resumen externo (media +/- std):")
    for metric_name in METRIC_KEYS:
        mean_value = results["summary"][f"mean_{metric_name}"]
        std_value = results["summary"][f"std_{metric_name}"]
        print(f"  {metric_name}: {mean_value:.4f} +/- {std_value:.4f}")

    for fold_result in results["outer_folds"]:
        outer = fold_result["outer_metrics"]
        print(
            f"Fold externo {fold_result['outer_fold']}: "
            f"MAE interno = {fold_result['inner_mae_mean']:.4f} "
            f"+/- {fold_result['inner_mae_std']:.4f}, "
            f"acc = {outer['accuracy']:.4f}, "
            f"F1_macro = {outer['f1_macro']:.4f}, "
            f"MAE = {outer['mae_ordinal']:.4f}, "
            f"err>=2 = {outer['errores_graves']:.4f}, "
            f"loss final = {fold_result['final_train_loss']:.4f}"
        )

    print("Reporte del ultimo fold externo:")
    print(results["last_fold_report"])
    print("Matriz de confusion del ultimo fold externo:")
    print(results["last_fold_confusion"])


def build_argument_parser() -> argparse.ArgumentParser:
    """Define los argumentos de linea de comandos para el experimento base."""
    parser = argparse.ArgumentParser(
        description=(
            "Laboratorio base: red poco profunda multiclase, validacion "
            "anidada y metricas ordinales. Use --all-targets para los seis "
            "experimentos, tablas y rankings."
        )
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Ruta al archivo de datos. Debe estar dentro de dataset/.",
    )
    parser.add_argument(
        "--target-name",
        default=DEFAULT_TARGET,
        help=(
            f"Experimento activo. Por defecto usa {DEFAULT_TARGET}. "
            "Se ignora si se usa --all-targets."
        ),
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help=(
            "Ejecuta GDS y GDS_R1 ... GDS_R5. GDS usa --gds-outer-folds "
            "y --gds-inner-folds."
        ),
    )
    parser.add_argument(
        "--coral",
        action="store_true",
        help="Activa la arquitectura CORAL en lugar de Softmax.",
    )
    parser.add_argument(
        "--use-weights",
        action="store_true",
        help="Activa los pesos por número efectivo de muestras para lidiar con el desbalance.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.99,
        help="Parámetro beta para el cálculo del número efectivo de muestras.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help="Numero de neuronas de la capa oculta.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Probabilidad de dropout.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate del optimizador Adam.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay del optimizador Adam.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Tamano de batch para DataLoader.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Numero de epocas de entrenamiento por fold.",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=DEFAULT_OUTER_FOLDS,
        help="Cantidad de folds externos para la evaluacion final.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=DEFAULT_INNER_FOLDS,
        help="Cantidad de folds internos para la validacion.",
    )
    parser.add_argument(
        "--gds-outer-folds",
        type=int,
        default=DEFAULT_GDS_OUTER_FOLDS,
        help="Folds externos para GDS (por defecto 2; 5 folds fallan).",
    )
    parser.add_argument(
        "--gds-inner-folds",
        type=int,
        default=DEFAULT_GDS_INNER_FOLDS,
        help="Folds internos para GDS (por defecto 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Dispositivo para ejecutar el experimento.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta para CSV, Markdown y graficos PNG.",
    )
    parser.add_argument(
        "--rank-metric",
        default=DEFAULT_RANK_METRIC,
        choices=METRIC_KEYS,
        help="Metrica del ranking generico (METRIC_KEYS).",
    )
    parser.add_argument(
        "--rank-mode",
        default=DEFAULT_RANK_MODE,
        choices=["auto", "max", "min"],
        help="auto minimiza MAE y errores graves; maximiza el resto.",
    )
    return parser


def main() -> None:
    """Ejecuta uno o todos los experimentos y escribe reportes."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.all_targets:
        target_names = list(TARGET_COLUMNS)
        print(
            "Modo --all-targets: se ignoran --target-name. "
            f"GDS usa {args.gds_outer_folds} folds externos y "
            f"{args.gds_inner_folds} internos."
        )
    else:
        target_names = [args.target_name]

    rows = []
    confusion_by_target = {}
    
    for target_name in target_names:
        outer_folds, inner_folds = folds_for_target(
            target_name,
            args.outer_folds,
            args.inner_folds,
            args.gds_outer_folds,
            args.gds_inner_folds,
            args.all_targets,
        )
        
        alg_label = "CORAL" if args.coral else "Softmax (HP)"
        if args.coral and args.use_weights:
            alg_label = "CORAL + pesos"
            
        print(f"\n=== {alg_label} / {target_name} ===")
        results = train_one_experiment(
            data_path=args.data_path,
            target_name=target_name,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            seed=args.seed,
            device_name=args.device,
            is_coral=args.coral,
            use_weights=args.use_weights,
            beta=args.beta,
        )
        print_experiment_results(results)
        rows.append(experiment_to_row(results, algorithm=results["algorithm"]))
        confusion_by_target[target_name] = results["last_fold_confusion"]

    rank_mode = resolve_rank_mode(args.rank_metric, args.rank_mode)
    print()
    print(format_ranking_console(rank_rows(rows, "f1_macro", "max"), "f1_macro", "max"))
    print(format_ranking_console(rank_rows(rows, "mae_ordinal", "min"), "mae_ordinal", "min"))
    print(
        format_ranking_console(
            rank_rows(rows, args.rank_metric, rank_mode),
            args.rank_metric,
            rank_mode,
        )
    )

    paths = save_experiment_reports(
        rows,
        args.output_dir,
        rank_metric=args.rank_metric,
        rank_mode=args.rank_mode,
        confusion_by_target=confusion_by_target,
    )
    print(f"\nReportes escritos en {args.output_dir}/")
    for name, path in paths.items():
        print(f"  {name}: {path}")

if __name__ == "__main__":
    main()