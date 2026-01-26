# Resolución Jigsaw
En este repositorios implementamos y explicamos cómo resolver automáticamente un rompecabezas a partir de una imagen de las piezas desordenadas y sin información adicional.  
El objetivo principal es aplicar diferentes técnicas de Análisis de Imágenes (Visión por Computador) y un algoritmo de Backtracking para conseguir una configuración final correcta.

El repositorio está organizado en notebooks explicativos y un módulo .py

---

### Notebooks y módulo

#### `1_Carga_y_procesamiento.ipynb`
- Carga de la imagen original del puzzle
- Segmentación de las piezas
- Extracción de contornos y esquinas
- Almacenamiento de la información en un .pkl

#### `2_Explicacion_algoritmo_puzzle.ipynb`
Explicación del código del archivo.py:
- Explicación de las clases principales `Piece` y `Edge`
- Clasificación de bordes
- Definición de la métrica de disimilitud entre bordes
- Explicación algoritmo de backtracking para obtener el puzzle completo

#### `3_visualizacion_de_soluciones.ipynb`
- Visualización gráfica paso a paso del puzzle durante el algoritmo de backtracking, tanto por delante como por detrás

#### `jigsaw.py`
Contiene el código de todo lo explicado en los notebooks 2 y 3.

---

## Dependencias

- OpenCV
- NumPy
- Matplotlib
- Pandas
- Sklearn
- SciPy
- tqdm
- itertools
- IPython

---

Autores: 
- Jordi Hamberg Gallego
- Héctor Sancho Rodríguez

Este proyecto tiene un enfoque en el que priorizamos principalmente la claridad del proceso y que se vea nuestra experimentación, en vez de un código súper compacto y óptimo.


