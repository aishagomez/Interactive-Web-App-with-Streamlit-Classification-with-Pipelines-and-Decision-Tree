import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
import cloudpickle
import io
import zipfile
from preprocessing_util import OutlierRemover, CustomImputer, CustomEncoder
from utils import train_val_test_split  
import datetime


def run_preprocessing_tab(df):
    st.markdown("## 🛠 Data Preprocessing (Custom Transformers)")

    st.subheader("✂️ Train/Val/Test Split")
    target_col = st.selectbox("Select target column", [None] + df.columns.tolist())
    test_size = st.slider("Test size", 0.05, 0.5, 0.2, 0.05)
    val_size = st.slider("Validation size (from test)", 0.05, 0.6, 0.5, 0.05)
    stratify = st.checkbox("Stratify by target?", value=True)

    if st.button("🔀 Split Dataset") and target_col is not None:
        train_set, val_set, test_set = train_val_test_split(
            df, test_size=test_size, val_size=val_size,
            stratify=target_col if stratify else None
        )
        

        # Separar X e y
        X_train = train_set.drop(columns=[target_col])
        y_train = train_set[target_col].copy()
        X_val = val_set.drop(columns=[target_col])
        y_val = val_set[target_col].copy()
        X_test = test_set.drop(columns=[target_col])
        y_test = test_set[target_col].copy()

        # Guardar sets limpios en sesión
        st.session_state['splits'] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "target": target_col
        }
        st.success("✅ Dataset split successfully!")
    st.subheader("🧹 Data Cleaning")
    remove_outliers = st.checkbox("Remove outliers?", value=True)

    if st.button("🛠 Apply Cleaning"):
        X_train = st.session_state['splits']['X_train']
        y_train = st.session_state['splits']['y_train']
        
        numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
        
        if numeric_cols:  
            cleaner = Pipeline([
                ('remove_outliers', OutlierRemover(do=remove_outliers))
            ])
            X_train_num_clean = pd.DataFrame(
                cleaner.fit_transform(X_train[numeric_cols]),
                columns=numeric_cols,
                index=X_train.index
            )
            # Reemplazar solo las columnas numéricas en X_train
            X_train_clean = X_train.copy()
            X_train_clean[numeric_cols] = X_train_num_clean
            y_train_clean = y_train.loc[X_train_clean.index]

            # Guardar en session_state
            st.session_state['splits'].update({
                "X_train": X_train_clean,
                "y_train": y_train_clean
            })
            st.success("✅ Cleaning applied to numeric columns successfully!")
        else:
            st.warning("⚠️ No numeric columns found to clean.")

    # --- Cargar pipeline existente ---
    uploaded_pipeline = st.file_uploader("📂 Load Preprocessing Pipeline (.pkl)", type=["pkl"])
    if uploaded_pipeline:
        st.session_state['preprocessing_pipeline'] = cloudpickle.load(uploaded_pipeline)
        st.success("✅ Pipeline loaded successfully!")

        # Aplicar pipeline cargado a los splits existentes si ya se hicieron
        if 'splits' in st.session_state:
            pipeline = st.session_state['preprocessing_pipeline']
            splits = st.session_state['splits']

            X_train_transformed = pd.DataFrame(
                pipeline.transform(splits['X_train']),
                columns=splits['X_train'].columns,
                index=splits['X_train'].index
            )
            X_val_transformed = pd.DataFrame(
                pipeline.transform(splits['X_val']),
                columns=splits['X_val'].columns,
                index=splits['X_val'].index
            )
            X_test_transformed = pd.DataFrame(
                pipeline.transform(splits['X_test']),
                columns=splits['X_test'].columns,
                index=splits['X_test'].index
            )

            # Actualizar splits
            st.session_state['splits'].update({
                'X_train': X_train_transformed,
                'X_val': X_val_transformed,
                'X_test': X_test_transformed
            })
            st.success("✅ Splits updated with loaded pipeline!")

    # --- Configuración del pipeline ---
    st.subheader("⚙️ Preprocessing Pipeline")
    if 'splits' in st.session_state:
        X_train = st.session_state['splits']['X_train']
        X_val = st.session_state['splits']['X_val']
        X_test = st.session_state['splits']['X_test']

        X_all = pd.concat([X_train, X_val, X_test], axis=0)

        numeric_cols = X_all.select_dtypes(include=['int64','float64']).columns.tolist()
        categorical_cols = X_all.select_dtypes(include=['object','category']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in numeric_cols]
        num_strategy = st.selectbox("Numeric Imputation Strategy", ["median", "mean"], index=0)
        cat_strategy = st.selectbox("Categorical Imputation Strategy", ["most_frequent", "constant"], index=0)
        encoder_type = st.selectbox("Categorical Encoder", ["ordinal", "onehot"], index=0)
        
        if st.button("💾 Generate Preprocessing Pipeline"):
            num_pipeline = Pipeline([
                ('imputer', CustomImputer(strategy=num_strategy)),
                ('scaler', RobustScaler())
            ])
            cat_pipeline = Pipeline([
                ('imputer', CustomImputer(strategy=cat_strategy)),
                ('encoder', CustomEncoder(method=encoder_type))
            ])
            full_pipeline = ColumnTransformer([
                ('num', num_pipeline, numeric_cols),
                ('cat', cat_pipeline, categorical_cols)
            ])

            full_pipeline.fit(X_all)  # Solo sobre X
            st.session_state['preprocessing_pipeline'] = full_pipeline
            st.success("✅ Preprocessing pipeline fitted and ready!")

    # --- Descargar pipeline ---
    if 'preprocessing_pipeline' in st.session_state:
        buffer = io.BytesIO()
        cloudpickle.dump(st.session_state['preprocessing_pipeline'], buffer)
        buffer.seek(0)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇️ Download Preprocessing Pipeline",
            data=buffer,
            file_name=f"preprocessing_pipeline_{timestamp}.pkl",
            mime="application/octet-stream"
        )

    # --- Aplicar pipeline a X_train, X_val, X_test y actualizar splits ---
    if 'splits' in st.session_state and 'preprocessing_pipeline' in st.session_state:
        pipeline = st.session_state['preprocessing_pipeline']
        splits = st.session_state['splits']

        # Transformar solo las X
        X_train_transformed = pd.DataFrame(
            pipeline.transform(splits['X_train']),
            columns=splits['X_train'].columns,
            index=splits['X_train'].index
        )
        X_val_transformed = pd.DataFrame(
            pipeline.transform(splits['X_val']),
            columns=splits['X_val'].columns,
            index=splits['X_val'].index
        )
        X_test_transformed = pd.DataFrame(
            pipeline.transform(splits['X_test']),
            columns=splits['X_test'].columns,
            index=splits['X_test'].index
        )

        # Actualizar splits
        st.session_state['splits'].update({
            'X_train': X_train_transformed,
            'X_val': X_val_transformed,
            'X_test': X_test_transformed
        })

    # --- Botón para generar ZIP de los splits ---
    if 'splits' in st.session_state:
        if st.button("📦 Download Train/Val/Test as ZIP"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, mode="w") as zf:
                for split_name in ["train", "val", "test"]:
                    X = st.session_state['splits'][f'X_{split_name}']
                    y = st.session_state['splits'][f'y_{split_name}']
                    df_split = pd.concat([X, y], axis=1)
                    csv_bytes = df_split.to_csv(index=False).encode('utf-8')
                    zf.writestr(f"{split_name}.csv", csv_bytes)
            buffer.seek(0)
            st.download_button(
                label="⬇️ Download ZIP",
                data=buffer,
                file_name=f"dataset_splits_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )

    
    return st.session_state.get('splits', {})
