import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import cloudpickle
import io


def display_classification_report(y_true, y_pred, title="Report"):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    st.subheader(title)
    st.dataframe(df_report.round(3))  

def train_and_evaluate_model(splits):
    st.subheader("Train Model")

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    full_pipeline = st.session_state['preprocessing_pipeline']

    model_choice = st.radio("Select Model", ["Decision Tree", "Random Forest"])
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    clf_pipeline = Pipeline([
        ("preprocessor", full_pipeline),
        ("classifier", model)
    ])

    if st.button("Train Model"):
        clf_pipeline.fit(X_train, y_train)
        st.session_state['model_pipeline'] = clf_pipeline  # guardamos pipeline completo

        st.success("Model trained!")

        display_classification_report(y_train, clf_pipeline.predict(X_train), "📈 Train Set Report")
        display_classification_report(y_val, clf_pipeline.predict(X_val), "📈 Validation Set Report")
        display_classification_report(y_test, clf_pipeline.predict(X_test), "📈 Test Set Report")

        # Guardar pipeline entrenado y ofrecer descarga
        buffer = io.BytesIO()
        cloudpickle.dump(clf_pipeline, buffer)
        buffer.seek(0)

        st.download_button(
            label="Download Trained Pipeline",
            data=buffer,
            file_name=f"{model_choice.lower().replace(' ', '_')}_pipeline.pkl",
            mime="application/octet-stream"
        )

def run_prediction_tab(splits, df):
    st.subheader("Predict with Trained Model")
    pipeline = st.session_state.get('model_pipeline', None)

    if pipeline is None:
        st.warning("Train a model first")
        return

    input_cols = splits["X_train"].columns.tolist()
    default_row = df.iloc[0]

    inputs = {}
    for col in input_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            value = st.number_input(f"Enter value for {col}", value=float(default_row[col]))
        else:
            value = st.text_input(f"Enter value for {col}", value=str(default_row[col]))
        inputs[col] = value

    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        prediction = pipeline.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
