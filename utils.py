import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def load_example_dataset():
    return sns.load_dataset("iris")


def read_uploaded_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type")


def detect_column_types(df):
    convert_to_numeric = []
    convert_to_datetime = []
    low_cardinality = []

    for col in df.columns:
        series = df[col]

        if series.dtype == "object":
            numeric_like = pd.to_numeric(series, errors="coerce")
            if numeric_like.notna().mean() > 0.95:
                convert_to_numeric.append(col)

            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() > 0.95:
                convert_to_datetime.append(col)

        elif pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            nunique = series.nunique(dropna=True)
            if nunique <= 10:
                low_cardinality.append(col)

    return {
        "convert_to_numeric": convert_to_numeric,
        "convert_to_datetime": convert_to_datetime,
        "low_cardinality": low_cardinality,
    }

def apply_conversions(df, numeric_cols, categorical_cols, datetime_cols):
    df = df.copy()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def train_val_test_split(df, rstate=42, test_size=0.2, val_size=0.5, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=test_size, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=val_size, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)