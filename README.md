# 📊 EDA + ML Pipeline

Una aplicación web interactiva para **Exploratory Data Analysis (EDA)** y **Machine Learning** usando Streamlit. Permite cargar datasets, realizar limpieza y transformaciones, entrenar modelos y hacer predicciones de manera sencilla.

---

## Características

- **Carga de Datos**: Soporte para CSV y XLSX, o usar un dataset de ejemplo (Iris).
- **Vista previa del dataset**: Forma, tipos de datos, valores nulos, estadísticas descriptivas.
- **EDA Interactivo**:
  - Sugerencias automáticas para convertir tipos de columnas (numéricas, categóricas, fechas).
  - Eliminación y recuperación de columnas.
  - Visualizaciones interactivas: histogramas, boxplots, scatter, heatmaps, proporciones categóricas, comparaciones numéricas y categóricas.
- **Preprocesamiento**:
  - Eliminación de outliers.
  - Imputación de valores faltantes (numérica y categórica).
  - Escalado de variables numéricas y codificación de categóricas (ordinal / one-hot).
  - Split Train / Validation / Test.
  - Guardado y carga de pipelines de preprocesamiento.
- **Modelado**:
  - Entrenamiento de Decision Tree y Random Forest.
  - Reportes de clasificación en Train / Validation / Test.
  - Descarga de pipeline entrenado.
- **Predicción**:
  - Predicción con el modelo entrenado para nuevas entradas.
  - Interfaz de entrada interactiva para cada columna.

---

## Estructura de Archivos

```
.
├── app.py                # Script principal de Streamlit
├── eda.py                # Funciones de EDA y visualización
├── preprocessing.py      # Pipeline de limpieza y preprocesamiento
├── preprocessing_util.py # Transformers personalizados (imputación, outliers, encoding)
├── modeling.py           # Entrenamiento y predicción de modelos
├── utils.py              # Funciones auxiliares (lectura de archivos, splits, detección de tipos)
├── requirements.txt      # Dependencias del proyecto
└── README.md
```

---

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/aishagomez/Interactive-Web-App-with-Streamlit-Classification-with-Pipelines-and-Decision-Tree
cd Interactive-Web-App-with-Streamlit-Classification-with-Pipelines-and-Decision-Tree
```

2. Crea un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

---

## Uso

Ejecuta la app de Streamlit:

```bash
streamlit run app.py
```

1. Carga tu dataset en formato CSV/XLSX o selecciona el dataset de ejemplo.
2. Navega entre las pestañas:

- **Dataset Preview**: Vista rápida de tu dataset, tipos de columnas, nulos y estadísticas.
- **EDA**: Visualización interactiva, sugerencias de transformación, eliminación de columnas.
- **Cleaning & Transformations**: Limpieza de datos, eliminación de outliers, generación de pipeline.
- **Train Model**: Entrenamiento de Decision Tree o Random Forest, reportes de clasificación y descarga del pipeline.
- **Predict**: Predicción usando el modelo entrenado para nuevas entradas.

