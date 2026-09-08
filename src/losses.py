"""Perdidas y ponderaciones que los alumnos deben implementar."""

import numpy as np
import torch


def labels_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    TODO(alumno):
    Convierte clases enteras a umbrales binarios acumulativos.

    Ejemplo:
    Si num_classes = 5 y la etiqueta es 2, el vector debe ser [1, 1, 0, 0].

    Formas:
    - labels: (batch_size,)
    - salida: (batch_size, num_classes - 1)
    """

    raise NotImplementedError("TODO: implementar labels_to_levels().")


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    TODO(alumno):
    BCE con logits sobre los K-1 umbrales ordinales.

    Formas:
    - logits: (batch_size, num_classes - 1)
    - labels: (batch_size,)
    - class_weights: (num_classes,) o None
    """

    raise NotImplementedError("TODO: implementar coral_loss().")


def effective_number_weights(
    labels: np.ndarray,
    num_classes: int,
    beta: float = 0.99,
) -> torch.Tensor:
    """
    TODO(alumno):
    Pesos por numero efectivo de muestras:

        w_c = (1 - beta) / (1 - beta ** n_c)

    Normalizar los pesos para que su media sea 1.

    Formas:
    - labels: (N,)
    - salida: (num_classes,)
    """

    raise NotImplementedError("TODO: implementar effective_number_weights().")
