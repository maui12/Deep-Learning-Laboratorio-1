"""Metricas para evaluar salidas multiclase y ordinales."""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mae_ordinal",
    "qwk",
    "accuracy_pm1",
    "errores_graves",
]


def _as_labels(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred deben tener la misma forma.")

    if len(y_true) == 0:
        raise ValueError("No se puede calcular una metrica sobre un arreglo vacio.")

    return y_true, y_pred


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraccion de muestras con clase exacta correcta."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy promedio por clase. Util con desbalance; no usa el orden."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    return float(balanced_accuracy_score(y_true, y_pred))


def precision_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision promedio no ponderado entre clases."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    return float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )


def recall_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall promedio no ponderado entre clases."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 promedio no ponderado entre clases."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def mae_ordinal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Distancia media |y - y_hat| sobre las clases enteras. Mas baja es mejor."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Acuerdo ordinal con penalizacion cuadratica. Mas alta es mejor (maximo 1)."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(labels) < 2:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    return float(
        cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)
    )


def accuracy_pm1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraccion de predicciones con |y - y_hat| <= 1."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def errores_graves(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraccion de predicciones con |y - y_hat| >= 2. Mas baja es mejor."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred) >= 2))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula el conjunto de metricas entregado en el laboratorio."""

    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "precision_macro": precision_macro(y_true, y_pred),
        "recall_macro": recall_macro(y_true, y_pred),
        "f1_macro": f1_macro(y_true, y_pred),
        "mae_ordinal": mae_ordinal(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "accuracy_pm1": accuracy_pm1(y_true, y_pred),
        "errores_graves": errores_graves(y_true, y_pred),
    }


def format_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list | None = None
) -> str:
    """Reporte por clase de precision, recall y F1."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    target_names = None
    if class_names is not None and len(class_names) >= int(labels.max()) + 1:
        target_names = [str(class_names[int(label)]) for label in labels]
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Matriz de confusion para inspeccionar errores adyacentes vs saltos."""

    y_true, y_pred = _as_labels(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return confusion_matrix(y_true, y_pred, labels=labels)
