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

    # Generar los niveles [1, 2, ..., K-1] en el mismo dispositivo que las etiquetas
    levels = torch.arange(1, num_classes, device=labels.device)
    
    # Expandir labels a (batch_size, 1) para comparar con levels (K-1)
    # y convertir el resultado booleano a float32
    return (labels.unsqueeze(dim=1) >= levels).float()





def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    levels = labels_to_levels(labels, num_classes)
    
    # BCE individual sobre los K-1 umbrales sin reducción inmediata
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, levels, reduction="none"
    )
    
    # Sumar los errores de los K-1 umbrales para cada muestra
    loss = bce.sum(dim=1)
    
    # Ponderar por el peso de la clase respectiva si existe
    if class_weights is not None:
        loss = loss * class_weights[labels]
        
    return loss.mean()


def effective_number_weights(
    labels: np.ndarray,
    num_classes: int,
    beta: float = 0.99,
) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1.0) # Prevenir división por cero
    
    # Fórmula w_c = (1 - beta) / (1 - beta ** n_c)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    
    # Normalizar para que la media sea 1
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)
