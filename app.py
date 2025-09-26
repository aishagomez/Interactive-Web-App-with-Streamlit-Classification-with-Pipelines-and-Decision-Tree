import streamlit as st
import pandas as pd
import numpy as np
import eda
import preprocessing
import modeling
import utils

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="EDA + ML Pipeline", layout="wide")
st.title("📊 EDA + ML Pipeline")

# --- SIDEBAR: CARGA DE DATOS ---
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV/XLSX", type=["csv", "xlsx"])
use_example = st.sidebar.checkbox("Use example dataset")

if uploaded_file:
    df = utils.read_uploaded_file(uploaded_file)
elif use_example:
    df = utils.load_example_dataset()
else:
    st.warning("Upload a dataset or select example dataset")
    st.stop()

# Guardamos dataset en session_state para mantener persistencia entre tabs
if "df" not in st.session_state:
    st.session_state['df'] = df


# --- CREACIÓN DE TABS PRINCIPALES ---
tabs = st.tabs(["Dataset Preview", "EDA", "Cleaning & Transformations", "Train Model", "Predict"])

# --- TAB 0: PREVIEW ---
with tabs[0]:

    df = st.session_state['df']
    # --- PREVIEW DE DATOS  ---
    st.subheader("Dataset Preview")
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(5), use_container_width=True)
        
    # Información de tipo de datos
    st.subheader("Dataset Information")
    dtypes_df = df.dtypes.to_frame(name="Tipo de Dato").astype(str)
    nulls_df = df.isnull().sum().to_frame(name="Nulos")
    info_df = pd.concat([dtypes_df, nulls_df], axis=1)
    st.dataframe(info_df, use_container_width=True)

    # --- ESTADÍSTICAS DESCRITIVAS ---    
    st.subheader("Statistics Overview")
    # Dividir las columnas en numéricas y categóricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Mostrar estadísticas numéricas
    if numeric_columns:
        st.subheader("Numeric Columns Statistics")
        numeric_desc = df[numeric_columns].describe().T
        st.dataframe(numeric_desc, use_container_width=True)
    else:
        st.warning("No numeric columns found.")
    
    # Mostrar estadísticas categóricas
    if categorical_columns:
        st.subheader("Categorical Columns Statistics")
        categorical_desc = df[categorical_columns].describe(include=['object', 'category']).T
        st.dataframe(categorical_desc, use_container_width=True)
    else:
        st.warning("No categorical columns found.")

# --- TAB 1: EDA ---
with tabs[1]:
    eda.run_eda_tab(st.session_state['df'])
    st.markdown("---")
    target_col = st.selectbox("Select target column for analysis (optional)", [None]+st.session_state['df'].columns.tolist())
    st.session_state['target']=target_col
    eda.run_visualization_tab(st.session_state['df'], target=target_col)
    


# --- TAB 2: CLEANING & TRANSFORMATIONS ---
with tabs[2]:
    st.markdown("## Cleaning & Preprocessing")
    preprocessing.run_preprocessing_tab(st.session_state['df'])

# --- TAB 3: TRAIN MODEL ---
with tabs[3]:
    # 🔑 Aquí usamos los datos limpios y separados
    if 'splits' not in st.session_state:
        st.warning("You must run preprocessing first (Tab 2)")
    else:
        modeling.train_and_evaluate_model(st.session_state['splits'])

# --- TAB 4: PREDICT ---
with tabs[4]:
    if 'splits' not in st.session_state:
        st.warning("You must run preprocessing first (Tab 2)")
    else:
        modeling.run_prediction_tab(st.session_state['splits'], st.session_state['df'])
