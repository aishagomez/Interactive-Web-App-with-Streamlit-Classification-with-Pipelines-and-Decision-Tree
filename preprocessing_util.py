
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import streamlit as st  # Si usas Streamlit para la interfaz


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = None

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, do=True):
        self.do = do
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not self.do:
            return X
        X_copy = X.copy()
        num_cols = X_copy.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            Q1 = X_copy[col].quantile(0.25)
            Q3 = X_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X_copy = X_copy[(X_copy[col] >= lower) & (X_copy[col] <= upper)]
        return X_copy

"""### Transformers for categorical attributes"""

class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, method='onehot'):
        self.method = method
        self.encoder = None
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns
        if self.method == 'onehot':
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        elif self.method == 'ordinal':
            self.encoder = OrdinalEncoder()
        else:
            raise ValueError("Encoder no reconocido")
        self.encoder.fit(X)
        return self

    def transform(self, X):
        X_enc = self.encoder.transform(X)
        if self.method == 'onehot':
            cols = []
            for i, col in enumerate(X.columns):
                if hasattr(self.encoder, 'categories_'):
                    cats = self.encoder.categories_[i]
                    cols.extend([f"{col}_{c}" for c in cats])
            return pd.DataFrame(X_enc, columns=cols, index=X.index)
        else:
            return pd.DataFrame(X_enc, columns=X.columns, index=X.index)



def suggest_cleaning(df, null_threshold=0.5, corr_threshold=0.9):
    suggestions = {
        "high_null": [],
        "unique_ids": [],
        "zero_variance": [],
        "high_corr": []
    }

    # > X% nulos
    null_ratio = df.isna().mean()
    suggestions["high_null"] = null_ratio[null_ratio > null_threshold].index.tolist()

    # columnas con valores únicos (IDs)
    suggestions["unique_ids"] = [col for col in df.columns if df[col].nunique() == len(df)]

    # columnas con varianza cero
    suggestions["zero_variance"] = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if df[col].nunique() <= 1
    ]

    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        corr_matrix = num_df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_pairs = [
            col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)
        ]
        suggestions["high_corr"] = list(set(high_corr_pairs))

    return suggestions

def run_cleaning_tab(df):
    st_df = df.copy()
    return st_df
