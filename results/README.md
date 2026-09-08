# results

Aqui `main.py` escribe CSV, Markdown y PNG al usar `--output-dir results`
(por defecto).

Esos archivos no se suben a Git. Cada grupo genera los suyos con:

```bash
python main.py --data-path dataset/archivo.sav --all-targets --epochs 5
```
