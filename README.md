# DL-2026-02-L01

Proyecto base para el Laboratorio 01 de Deep Learning (segundo semestre 2026).

La idea del laboratorio, siguiendo la presentacion, es tratar el problema como
seis experimentos independientes:

- `GDS`
- `GDS_R1`
- `GDS_R2`
- `GDS_R3`
- `GDS_R4`
- `GDS_R5`

Cada experimento toma una sola columna objetivo y la recodifica a indices
enteros `0, ..., K-1`. Esta version deja una base Softmax completa, con
metricas multiclase y ordinales, para que los estudiantes implementen CORAL
y la busqueda de hiperparametros en el loop interno.

## Que esta implementado

- Carga basica de datos desde `csv` o `sav`
- Seleccion de columnas de entrada
- Recodificacion del target activo a indices `0 .. K-1`
- `Dataset` de PyTorch
- Red neuronal poco profunda multiclase (`ShallowMultiClassNet`)
- Entrenamiento base con `CrossEntropyLoss`
- Validacion anidada 5x3 con una configuracion fija
- Metricas multiclase: `accuracy`, `precision_macro`, `recall_macro`,
  `f1_macro`, `balanced_accuracy`
- Metricas ordinales: `mae_ordinal`, `quadratic_weighted_kappa`,
  `accuracy_pm1`, `errores_graves`
- Reporte por clase y matriz de confusion del ultimo fold externo
- `--all-targets`: los seis experimentos, tablas, rankings (F1 y MAE) y graficos
- Ranking configurable con `--rank-metric` y `--rank-mode`
- Ejemplo de grid en `HYPERPARAMETER_GRID` (todavia no se recorre)

## Que queda como TODO

- Recorrer `HYPERPARAMETER_GRID` en el loop interno
  Elegir la configuracion con menor MAE interno (empate: mayor QWK).
  El loop externo solo reporta; no se reajusta sobre test.
  Esta busqueda es un resultado esperado del informe, no una extension opcional.
- Capa CORAL y red `MLPCoral`
- Perdida ordinal y pesos por numero efectivo de muestras
- Comparacion entre Softmax y CORAL en los seis experimentos
- Completar la tabla de resultados de la presentacion

## Estructura

```text
DL-2026-02-L01/
|-- dataset/
|   `-- README.md
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- evaluation.py
|   |-- losses.py
|   |-- models.py
|   |-- ordinal.py
|   |-- preprocessing.py
|   `-- reporting.py
|-- main.py
|-- results/
|   `-- README.md
|-- presentation/
|-- .gitignore
|-- environment.yml
`-- README.md
```

## Uso esperado

1. Copiar el dataset dentro de `dataset/`.
2. Activar el entorno Conda `lab_pytorch` (ya creado a partir de `environment.yml`).
3. Ejecutar un experimento base.

```bash
conda activate lab_pytorch
python main.py --data-path dataset/archivo.csv --target-name GDS_R2
python main.py --data-path dataset/archivo.sav --all-targets --epochs 5
```

`--all-targets`:

- entrena Softmax en `GDS` y `GDS_R1` ... `GDS_R5`,
- usa 2 folds en `GDS` para no romper la estratificacion,
- escribe CSV, Markdown y PNG en `results/`,
- e imprime rankings por F1 (maximizar) y MAE (minimizar).

Un experimento suelto (`--target-name`):

- carga el dataset,
- prepara `X` e `y`,
- entrena una red neuronal poco profunda,
- aplica validacion externa con folds estratificados,
- aplica validacion interna dentro de cada fold externo,
- y reporta media +/- std externa de todas las metricas.

## Flujo implementado

La implementacion sigue la idea general mostrada en la presentacion:

1. Elegir una columna objetivo, por ejemplo `GDS_R2`.
2. Recodificarla a indices enteros `0 .. K-1`.
3. Crear `5` folds externos estratificados para evaluacion final.
4. Crear `3` folds internos dentro de cada entrenamiento externo.
5. Entrenar la configuracion base en cada fold.
6. Reportar metricas en validacion interna (MAE/QWK) y prueba externa.

La busqueda de hiperparametros **no se ejecuta sola**. El grid de ejemplo esta
en `src/config.py`. Hay que activarlo en el loop interno de `main.py`.

Criterio de seleccion: minimizar MAE interno; si hay empate, maximizar QWK.
No seleccionar por accuracy.

## Advertencia sobre `GDS`

El objetivo `GDS` tiene 7 clases y la clase menos frecuente tiene solo 2
muestras. Con el valor por defecto de `5` folds externos, la validacion
estratificada falla. Para un experimento suelto hay que reducir
`--outer-folds` y `--inner-folds`, o trabajar primero con `GDS_R2`.

Con `--all-targets`, `GDS` usa `--gds-outer-folds` y `--gds-inner-folds`
(por defecto `2`). El resto de columnas usa `--outer-folds` / `--inner-folds`.
Si un fold interno no puede estratificar, se omite ese loop interno y se avisa.

## Argumentos utiles

- `--data-path`: ruta al archivo `csv` o `sav`.
- `--target-name`: un experimento. Por defecto `GDS_R2`. Se ignora con `--all-targets`.
- `--all-targets`: corre los seis objetivos y genera reportes.
- `--hidden-dim`: neuronas de la capa oculta.
- `--dropout`: dropout de la red.
- `--learning-rate`: learning rate de Adam.
- `--weight-decay`: weight decay de Adam.
- `--batch-size`: batch size de entrenamiento.
- `--epochs`: epocas por fold.
- `--outer-folds`: folds externos. Por defecto `5`.
- `--inner-folds`: folds internos. Por defecto `3`.
- `--gds-outer-folds` / `--gds-inner-folds`: folds de `GDS` con `--all-targets` (por defecto `2`).
- `--device`: `cpu`, `cuda` o `auto`.
- `--output-dir`: carpeta de CSV, Markdown y PNG. Por defecto `results/`.
- `--rank-metric`: metrica del ranking generico (`f1_macro`, `mae_ordinal`, ...).
- `--rank-mode`: `max`, `min` o `auto` (MAE y errores graves se minimizan).

Cambiar `--hidden-dim` a mano no es busqueda de hiperparametros. Recorrer
`HYPERPARAMETER_GRID` en el loop interno sigue como TODO.

`--coral`, `--use-weights` y `--beta` no existen todavia. CORAL es trabajo
del alumno. Las tablas ya tienen columna `algorithm` para agregar filas.

Para una primera prueba rapida:

```bash
python main.py --data-path dataset/archivo.csv --target-name GDS_R2 --epochs 5
python main.py --data-path dataset/archivo.sav --all-targets --epochs 5 --output-dir results
python main.py --data-path dataset/archivo.sav --all-targets --rank-metric mae_ordinal --rank-mode min
```

## Plan sugerido para alumnos

1. Comprender el problema y revisar las columnas del dataset.
2. Ejecutar el Softmax entregado en `GDS_R2` y leer las metricas.
3. Correr `--all-targets` y revisar rankings F1 y MAE en `results/`.
4. Activar la busqueda de hiperparametros en el loop interno (`HYPERPARAMETER_GRID`).
5. Implementar `CoralLayer`, `MLPCoral` y `coral_loss`.
6. Incorporar pesos por numero efectivo de muestras.
7. Agregar filas CORAL a las mismas tablas (`algorithm`) y completar el informe.

## Presentaciones

En `presentation/` (compilar dos veces con `pdflatex`):

- `Laboratorio_01_redes_poco_profundas_ordinal.tex`: Enunciado (27/08, entrega 24/09).
- `Laboratorio_01.2_redes_poco_profundas.tex`: Sesion dirigida del 03/09.
