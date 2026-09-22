"""Modelos del laboratorio."""

import torch
import torch.nn as nn


class ShallowMultiClassNet(nn.Module):
    """
    Red neuronal poco profunda para clasificacion multiclase.

    Tiene:
    - una capa de entrada,
    - una capa oculta,
    - dropout,
    - y una capa de salida con tantas neuronas como clases tenga el experimento.

    El forward devuelve logits. No se agrega Softmax aqui porque
    CrossEntropyLoss lo aplica internamente.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 32,
        dropout: float = 0.15,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits


class CoralLayer(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Un peso lineal compartido hacia un único puntaje latente
        self.fc = nn.Linear(input_size, 1, bias=False)
        
        # Sesgos ordenados
        self.bias_first = nn.Parameter(torch.zeros(1))
        if num_classes > 2:
            self.bias_diffs = nn.Parameter(torch.zeros(num_classes - 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = self.fc(inputs) # (batch_size, 1)
        
        if self.num_classes == 2:
            return score + self.bias_first
            
        # Diferencias positivas usando softplus
        diffs = torch.nn.functional.softplus(self.bias_diffs)
        # Forzar orden decreciente: b_k = b_{k-1} - diff_k
        biases = torch.cat([self.bias_first, self.bias_first - torch.cumsum(diffs, dim=0)])
        
        # (batch_size, K-1) por broadcasting
        return score + biases


class MLPCoral(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout

        # Arquitectura sugerida: 15 -> Linear(32) -> ReLU -> BatchNorm1d -> Dropout -> Linear(16) -> ReLU
        self.features = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.coral = CoralLayer(16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.features(inputs)
        return self.coral(hidden)