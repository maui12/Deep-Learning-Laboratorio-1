"""Constantes simples para mantener el proyecto ordenado."""

FEATURE_COLUMNS = [
    "Día",
    "Mes",
    "Año",
    "Estación",
    "País",
    "Ciudad",
    "CalleLugar",
    "NumeroPiso",
    "Miguel2",
    "González2",
    "Avenida2",
    "Imperial2",
    "A682",
    "Caldera2",
    "Copiapo2",
]

TARGET_COLUMNS = [
    "GDS",
    "GDS_R1",
    "GDS_R2",
    "GDS_R3",
    "GDS_R4",
    "GDS_R5",
]

ID_COLUMN = "ID"

DEFAULT_TARGET = "GDS_R2"
DEFAULT_HIDDEN_DIM = 32
DEFAULT_DROPOUT = 0.15
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 3
DEFAULT_GDS_OUTER_FOLDS = 2
DEFAULT_GDS_INNER_FOLDS = 2
DEFAULT_RANDOM_SEED = 42
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_RANK_METRIC = "f1_macro"
DEFAULT_RANK_MODE = "auto"
DEFAULT_ALGORITHM = "Softmax"
MINIMIZE_METRICS = ("mae_ordinal", "errores_graves")

# TODO(alumno): recorrer este grid en el loop interno de validacion.
# Seleccionar la configuracion con menor MAE interno (empate: mayor QWK).
# No usar el fold externo para elegir hiperparametros.
# Para CORAL, incluir tambien "beta" en cada diccionario.

HYPERPARAMETER_GRID = [
    # 1. Configuración base
    {
        "hidden_dim": 32,
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "beta": 0.99,
    },
    # 2. Mayor capacidad de red y mayor dropout (ayuda a generalizar)
    {
        "hidden_dim": 64,
        "dropout": 0.30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "beta": 0.99,
    },
    # 3. Red más densa con regularización agresiva y aprendizaje lento
    {
        "hidden_dim": 128,
        "dropout": 0.40,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "beta": 0.999,
    },
    # 4. Capacidad media con beta en 0.9 para probar otro umbral de desbalance
    {
        "hidden_dim": 64,
        "dropout": 0.15,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "beta": 0.90,
    },
    # 5. Red pequeña pero fuertemente regularizada
    {
        "hidden_dim": 32,
        "dropout": 0.30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "beta": 0.999,
    },
]