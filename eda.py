import streamlit as st
import utils
import pandas as pd
import plotly.express as px
import numpy as np



def run_eda_tab(df):
    # --- DETECCIÓN AUTOMÁTICA DE TIPOS ---
    auto_suggestions = utils.detect_column_types(df)

    st.markdown("### Column Type Suggestions")
    with st.expander("Check Automatic Suggestions"):
        if auto_suggestions["convert_to_numeric"]:
            st.info(f"Suggested numeric conversions: {auto_suggestions['convert_to_numeric']}")
        if auto_suggestions["convert_to_datetime"]:
            st.info(f"Suggested datetime conversions: {auto_suggestions['convert_to_datetime']}")
        if auto_suggestions["low_cardinality"]:
            st.info(f"Suggested categorical conversions (low cardinality): {auto_suggestions['low_cardinality']}")

    st.markdown("#### 🛠 Manual Adjustments")
    col_numeric = st.multiselect(
        "Select columns to convert to numeric",
        df.columns.tolist(),
        default=auto_suggestions["convert_to_numeric"]
    )
    col_categorical = st.multiselect(
        "Select columns to convert to categorical",
        df.columns.tolist(),
        default=auto_suggestions["low_cardinality"]
    )
    col_datetime = st.multiselect(
        "Select columns to convert to datetime",
        df.columns.tolist(),
        default=auto_suggestions["convert_to_datetime"]
    )
    if "df_snapshot_type" not in st.session_state:
        st.session_state['df_snapshot_type'] = st.session_state['df'].copy()

    if st.button("Apply Conversions"):
        df = utils.apply_conversions(df, col_numeric, col_categorical, col_datetime)
        st.session_state['df'] = df
        st.session_state['conversion_status'] = {
            "message": "Conversions applied successfully!",
            "dtypes": df.dtypes
        }
        if 'undo_type' in st.session_state:
            del  st.session_state['undo_type']
        st.rerun()
        
    if st.button("Undo Changes"):
        st.session_state['df'] = st.session_state['df_snapshot_type']
        st.session_state['undo_type'] = {
            "message": "Changes undone!"
        }
        del st.session_state['df_snapshot_type']
        if 'conversion_status' in st.session_state:
            del st.session_state['conversion_status']
        st.rerun()
    

    
    if 'conversion_status' in st.session_state:
        st.success(st.session_state['conversion_status']['message'])

    if 'undo_type' in st.session_state:
        st.warning(st.session_state['undo_type']['message'])








    st.markdown("### Column Removal Suggestions")

    # --- Detectar columnas ---
    n_rows = df.shape[0]
    n_null_threshold = 0.5  # 50%
    n_unique_threshold = 0.95  # 95%

    null_cols = [col for col in df.columns if df[col].isnull().mean() > n_null_threshold]
    unique_cols = [col for col in df.columns if not pd.api.types.is_float_dtype(df[col]) and df[col].nunique() / n_rows > n_unique_threshold]

    with st.expander("Suggested columns to remove"):
        if null_cols:
            st.warning(f"Columns with >50% nulls: {null_cols}")
        if unique_cols:
            st.warning(f"Columns with >95% unique values: {unique_cols}")
        if not null_cols and not unique_cols:
            st.info("No columns meet the removal criteria.")

    # --- Selección interactiva ---
    cols_to_remove = st.multiselect(
        "Select columns to remove",
        df.columns.tolist(),
        default=null_cols + unique_cols
    )

    # --- Inicializar snapshot si no existe ---
    if "df_snapshot_del" not in st.session_state:
        st.session_state['df_snapshot_del'] = st.session_state['df'].copy()

    # --- Botón Apply ---
    if st.button("Apply Column Removal"):
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            st.session_state['df'] = df
            st.session_state['del_status'] = {
                "message": f"Removed columns: {cols_to_remove}"
            }
            st.session_state['removed_columns'] = cols_to_remove
            if 'undo_del' in st.session_state:
                del  st.session_state['undo_del']
        else:
            st.info("No columns selected to remove.")
        
        st.rerun()

    if 'removed_columns' in st.session_state:
        cols_to_recuperate = st.multiselect(
            "Select columns to recuperate",
            st.session_state['removed_columns'],  
            default=[] 
        )
    else:
        cols_to_recuperate = [] 

    if st.button("Undo Column Removal"):
        df = st.session_state['df_snapshot_del']
        cols_to_remove_again = [col for col in st.session_state['removed_columns'] if col not in cols_to_recuperate and col in df.columns]
        if cols_to_remove_again:
            df = df.drop(columns=cols_to_remove_again)
        st.session_state['df'] = df
        st.session_state['undo_del'] = {
                "message": f"Recupered columns: {cols_to_recuperate}"
            }
        st.session_state['removed_columns']=cols_to_remove_again if cols_to_remove_again else []
        st.session_state['del_status'] = {
            "message": f"Removed columns still in dataset: {st.session_state['removed_columns']}"
            }
        
        if len(st.session_state['removed_columns']) == 0 or not st.session_state['removed_columns']: 
            del st.session_state['removed_columns']
            del st.session_state['df_snapshot_del']
            del st.session_state['del_status']
        st.rerun()
    
    if 'del_status' in st.session_state:
        st.success(st.session_state['del_status']['message'])
        
    if 'undo_del' in st.session_state:
        st.info(st.session_state['undo_del']['message'])







    # --- PREVIEW DE DATOS  ---
    st.subheader("Dataset Overview")
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(5), use_container_width=True)
        
    # Información de tipo de datos
    dtypes_df = df.dtypes.to_frame(name="Tipo de Dato").astype(str)
    nulls_df = df.isnull().sum().to_frame(name="Nulos")
    info_df = pd.concat([dtypes_df, nulls_df], axis=1)
    st.dataframe(info_df, use_container_width=True)

'''

    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), ax=ax, annot=True, cmap="coolwarm")
        st.pyplot(fig)

    st.subheader("Column Types & Suggested Transformations")
    for col in df.columns:
        st.write(f"{col} : {df[col].dtype} | Unique: {df[col].nunique()}")
'''


def run_visualization_tab(df, target=None):
    st.markdown("## Interactive Data Visualization")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    '''st.subheader("Descriptive Statistics")
    if numeric_cols:
        st.write("**Numeric Columns:**")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    if categorical_cols:
        st.write("**Categorical Columns:**")
        st.dataframe(df[categorical_cols].describe(include=['object', 'category']).T, use_container_width=True)
    '''
    st.subheader("Interactive Plots")
    plot_type = st.selectbox("Select Plot Type", [
        "Histogram", "Boxplot", "Scatter", 
        "Correlation Heatmap", "Categorical Proportion", 
        "Categorical vs Numeric", "Categorical vs Categorical"
    ])

    if plot_type == "Histogram":
        col = st.selectbox("Select numeric column", numeric_cols)
        fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Boxplot":
        col = st.selectbox("Select numeric column", numeric_cols)
        fig = px.box(df, y=col, points="all", title=f"Boxplot of {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter":
        x_col = st.selectbox("X-axis", numeric_cols, index=0)
        y_col = st.selectbox("Y-axis", numeric_cols, index=1)
        color_col = st.selectbox("Color (optional)", [None] + categorical_cols)
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, hover_data=df.columns)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Correlation Heatmap":
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Categorical Proportion":
        col = st.selectbox("Select categorical column", categorical_cols)
        counts = df[col].value_counts(normalize=True).reset_index()
        counts.columns = [col, "proportion"]
        fig = px.bar(counts, x=col, y="proportion", text="proportion", title=f"Proportion of categories in {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Categorical vs Numeric":
        cat_col = st.selectbox("Select categorical column", categorical_cols)
        num_col = st.selectbox("Select numeric column", numeric_cols)
        chart_type = st.radio("Chart type", ["Boxplot", "Violin", "Mean Bar"])
        if chart_type == "Boxplot":
            fig = px.box(df, x=cat_col, y=num_col, points="all", title=f"{num_col} by {cat_col}")
        elif chart_type == "Violin":
            fig = px.violin(df, x=cat_col, y=num_col, box=True, points="all", title=f"{num_col} by {cat_col}")
        elif chart_type == "Mean Bar":
            mean_df = df.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(mean_df, x=cat_col, y=num_col, title=f"Mean of {num_col} by {cat_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Categorical vs Categorical":
        cat1 = st.selectbox("Categorical X", categorical_cols, index=0)
        cat2 = st.selectbox("Categorical Y", categorical_cols, index=1)
        contingency = pd.crosstab(df[cat1], df[cat2])
        fig = px.imshow(contingency, text_auto=True, aspect="auto", title=f"{cat1} vs {cat2} Counts")
        st.plotly_chart(fig, use_container_width=True)

    if target and target in df.columns:
        st.subheader("Target Variable Analysis")
        st.write(f"Target: **{target}**")
        if df[target].dtype in ["object", "category"] or len(df[target].unique()) < 20:
            fig = px.histogram(df, x=target, title=f"Distribution of target {target}")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.write("Target is numeric")
            st.metric(label=f"Mean of Target", value=f"{df[target].mean():.2f}")
            st.write("**Correlation with numeric features**")
            corr_with_target = df[numeric_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)
            st.dataframe(corr_with_target, use_container_width=True)

            low_corr = corr_with_target[abs(corr_with_target) < 0.1].index.tolist()
            if low_corr:
                st.warning(f"Features weakly correlated with target (|corr| < 0.1): {low_corr}")
    st.subheader("Key Metrics & Target Insights")

    if target and target in df.columns and df[target].dtype not in ["object", "category"]:
        # Variables numéricas más correlacionadas
        corr_with_target = df[numeric_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)
        top_vars = corr_with_target.head(3).index.tolist()
        st.write("### Top 3 Features Correlated with Target")

        kpi_cols = st.columns(3)
        for i, var in enumerate(top_vars):
            with kpi_cols[i]:
                if df[var].dtype in ["object", "category"] or len(df[var].unique()) < 10:
                    counts = df[var].value_counts()
                    fig = px.pie(values=counts.values, names=counts.index, title=f"{var} Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(df, x=var, nbins=20, title=f"{var} Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                if pd.api.types.is_numeric_dtype(df[var]):
                    st.metric(label=f"Mean of {var}", value=f"{df[var].mean():.2f}")
                else:
                    st.metric(label=f"Most frequent of {var}", value=f"{df[var].mode()[0]}")

    else:
        st.write("### Select Variables to Inspect Metrics")
        selected_num = st.selectbox("Select numeric variable", [None] + numeric_cols)
        selected_cat = st.selectbox("Select categorical variable", [None] + categorical_cols)

        kpi_cols = st.columns(2)
        if selected_num:
            with kpi_cols[0]:
                fig = px.histogram(df, x=selected_num, nbins=20, title=f"{selected_num} Distribution")
                st.plotly_chart(fig, use_container_width=True)
                st.metric(label=f"Mean of {selected_num}", value=f"{df[selected_num].mean():.2f}")
        if selected_cat:
            with kpi_cols[1]:
                counts = df[selected_cat].value_counts()
                fig = px.pie(values=counts.values, names=counts.index, title=f"{selected_cat} Distribution")
                st.plotly_chart(fig, use_container_width=True)
                st.metric(label=f"Most frequent of {selected_cat}", value=f"{df[selected_cat].mode()[0]}")



