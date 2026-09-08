"""Funciones para preparar el target multiclase del experimento activo."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import TARGET_COLUMNS
from data_loader import build_input_matrix


def encode_target_as_indices(
    dataframe: pd.DataFrame, target_name: str
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Toma una sola columna objetivo y la convierte a indices 0 .. K-1.

    Ejemplo:
    Si las clases originales son 1, 2 y 3, entonces la clase 2 queda como 1.
    """

    if target_name not in TARGET_COLUMNS:
        raise ValueError(
            f"Target invalido: {target_name}. Debe ser uno de {TARGET_COLUMNS}."
        )

    if target_name not in dataframe.columns:
        raise ValueError(f"La columna objetivo {target_name} no existe en el dataset.")

    y_raw = dataframe[target_name].astype(int)
    classes = sorted(y_raw.unique().tolist())
    class_to_idx = {class_value: idx for idx, class_value in enumerate(classes)}
    y_idx = y_raw.map(class_to_idx).to_numpy(dtype=np.int64)
    return y_idx, classes, class_to_idx


def prepare_experiment_data(
    dataframe: pd.DataFrame, target_name: str
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int]]:
    """Prepara X e y para un experimento puntual."""

    X = build_input_matrix(dataframe)
    y, classes, class_to_idx = encode_target_as_indices(dataframe, target_name)
    return X, y, classes, class_to_idx


def split_for_validation(
    y: np.ndarray, n_splits: int, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """API publica para crear folds estratificados del laboratorio."""

    return build_nested_splits(y, n_splits=n_splits, random_state=random_state)


def build_nested_splits(
    y: np.ndarray, n_splits: int, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Genera folds estratificados para la validacion del laboratorio.

    y debe ser un vector de clases enteras, ya recodificado a 0 .. K-1.
    """

    labels = np.asarray(y).ravel()
    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(unique_labels) < 2:
        raise ValueError(
            "Se requieren al menos dos clases distintas para realizar validacion."
        )

    if counts.min() < n_splits:
        raise ValueError(
            "No se puede crear una validacion estratificada con "
            f"{n_splits} folds porque la clase menos frecuente solo tiene "
            f"{counts.min()} muestras. Prueba con menos folds o con un "
            "objetivo distinto de GDS."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    dummy_inputs = np.zeros(len(labels), dtype=np.float32)
    return list(splitter.split(dummy_inputs, labels))
