import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str = "car_price_prediction_.csv", nrows: Optional[int] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    return pd.read_csv(path, nrows=nrows)


def _select_features(df: pd.DataFrame, target: str = "price"):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=[object, "category"]).columns.tolist()
    if target in numeric:
        numeric.remove(target)
    cat = [c for c in cat if df[c].nunique() <= 50]
    return numeric, cat


def train_demo_model(df: pd.DataFrame, target: str = "price", sample_frac: float = 0.25, random_state: int = 42):
    target_col = None
    for c in df.columns:
        if c.lower() == target.lower():
            target_col = c
            break
    if target_col is None:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    df = df.dropna(subset=[target_col]).copy()
    if sample_frac and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state)

    numeric_cols, cat_cols = _select_features(df, target=target_col)
    if not numeric_cols and not cat_cols:
        raise ValueError("No usable features found in dataframe")

    X = df[numeric_cols + cat_cols]
    y = df[target_col]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", onehot),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X, y)
    pipeline.feature_names = numeric_cols + cat_cols
    try:
        bg_size = min(200, X.shape[0])
        bg_df = X.sample(n=bg_size, random_state=random_state)
        bg_transformed = pipeline.named_steps["preprocessor"].transform(bg_df)
        pipeline.background_data = bg_transformed
        pipeline.background_raw = bg_df.reset_index(drop=True)
    except Exception:
        pipeline.background_data = None
        pipeline.background_raw = None
    return pipeline


def save_model(pipeline: Pipeline, path: str = "models/demo_model.joblib") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: str = "models/demo_model.joblib") -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    return joblib.load(path)
