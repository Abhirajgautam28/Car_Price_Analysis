import os
import sys

import pandas as pd
import streamlit as st

# Ensure local package path is first so `import scripts` resolves to the repository package
proj_root = os.path.abspath(os.path.dirname(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from scripts.utils import load_data, save_model, train_demo_model, load_model

import plotly.express as px
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import List

st.set_page_config(layout="wide", page_title="Car Price Prediction Demo")

DATA_PATH = os.environ.get("DATA_PATH", "car_price_prediction_.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/demo_model.joblib")


@st.cache_data
def _load_data(nrows: int = 5000):
    return load_data(DATA_PATH, nrows=nrows)


@st.cache_resource
def _get_or_train_model(df):
    try:
        model = load_model(MODEL_PATH)
        # If loaded model is missing feature metadata, retrain to produce a compatible artifact
        if not hasattr(model, "feature_names"):
            raise RuntimeError("Loaded model missing feature metadata; retraining")
    except Exception:
        model = train_demo_model(df, sample_frac=0.25)
        save_model(model, MODEL_PATH)
    return model


def main():
    st.title("Car Price Prediction — Interactive Demo")
    st.markdown("This interactive demo uses a compact RandomForest pipeline for fast inference. See `car_price_prediction.ipynb` for the full pipeline and experiments.")

    with st.sidebar:
        st.header("Inputs")
        nrows = st.number_input("Load rows for demo", min_value=100, max_value=20000, value=5000, step=100)

    df = _load_data(nrows=nrows)

    def _detect_target_col(df: pd.DataFrame) -> str:
        for c in df.columns:
            if c.lower() == "price":
                return c
        return "price"

    def _get_preprocessor_feature_names(preprocessor: ColumnTransformer, input_features: List[str]) -> List[str]:
        """Return the feature names after the ColumnTransformer to map importances back to original features.

        This attempts to use sklearn's get_feature_names_out when available, otherwise it falls back
        to constructing names from transformers' categories_ (for OneHotEncoder) and numeric names.
        """
        try:
            return preprocessor.get_feature_names_out(input_features).tolist()
        except Exception:
            # Manual construction
            feature_names = []
            for name, transformer, cols in preprocessor.transformers:
                if name == "remainder":
                    continue
                if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                    ohe = transformer.named_steps["onehot"]
                    cats = []
                    try:
                        cats = ohe.categories_
                    except Exception:
                        cats = []
                    for col, catvals in zip(cols, cats):
                        for v in catvals:
                            feature_names.append(f"{col}_{v}")
                else:
                    # assume numeric passthrough
                    for col in cols:
                        feature_names.append(col)
            return feature_names

    st.sidebar.markdown("---")

    # Try to load model to know which features were used for training
    model = None
    try:
        model = load_model(MODEL_PATH)
        # if model does not have feature list, force retrain (saved artifact is older)
        if not hasattr(model, "feature_names"):
            model = None
    except Exception:
        model = None

    # Determine feature list for the input form
    if model is not None:
        feature_list = [f for f in getattr(model, "feature_names", []) if f in df.columns]
    else:
        # Fallback: use first three numeric features available
        numeric_candidates = [c for c in df.select_dtypes(include=["number"]).columns if c.lower() not in ("price",)]
        feature_list = numeric_candidates[:3]

    inputs = {}
    st.sidebar.markdown("**Feature inputs (demo)**")
    for c in feature_list:
        if c in df.select_dtypes(include=["number"]).columns:
            mn = int(df[c].min()) if pd.api.types.is_integer_dtype(df[c]) else float(df[c].min())
            mx = int(df[c].max()) if pd.api.types.is_integer_dtype(df[c]) else float(df[c].max())
            default = int(df[c].median()) if pd.api.types.is_integer_dtype(df[c]) else float(df[c].median())
            inputs[c] = st.sidebar.number_input(c, value=default, min_value=mn, max_value=mx)
        else:
            # categorical: provide a selectbox with top unique values (capped)
            uniques = df[c].dropna().unique().tolist()[:20]
            default = uniques[0] if uniques else "missing"
            inputs[c] = st.sidebar.selectbox(c, options=uniques or ["missing"], index=0)

    st.sidebar.markdown("---")

    tabs = st.tabs(["Predict", "Exploratory Analysis", "Model Diagnostics"]) 

    # --------------------- Predict tab ---------------------
    with tabs[0]:
        if st.button("Predict"):
            # Load or train model (this will save model if it was missing or outdated)
            model = _get_or_train_model(df)

            # Build a complete input row matching the model's feature list
            feature_names = getattr(model, "feature_names", [])
            if not feature_names:
                st.error("Model feature list is missing; retrain the model with `scripts/train.py`.")
            else:
                row = {}
                for f in feature_names:
                    if f in inputs:
                        row[f] = inputs[f]
                    else:
                        # fill defaults from data if available
                        if f in df.columns:
                            if f in df.select_dtypes(include=["number"]).columns:
                                row[f] = float(df[f].median())
                            else:
                                # categorical default
                                nonnull = df[f].dropna().unique().tolist()
                                row[f] = nonnull[0] if nonnull else "missing"
                        else:
                            # unknown column not in dataframe; safe fallback
                            row[f] = 0.0

                Xpred = pd.DataFrame([row], columns=feature_names)
                pred = model.predict(Xpred)[0]
                st.metric("Predicted Price (demo)", f"${pred:,.0f}")

                # SHAP explanation (best-effort): prefer background saved during training
                # Explanation pipeline (SHAP -> model.feature_importances_ -> permutation importance)
                preprocessor = model.named_steps.get("preprocessor")
                model_step = model.named_steps.get("model")
                # 1) Try SHAP if installed and model is explainable
                # detect if shap is importable without raising during import
                try:
                    import importlib
                    shap_available = importlib.util.find_spec("shap") is not None
                except Exception:
                    shap_available = False

                explained = False
                # Use SHAP for tree models if available
                if shap_available and model_step is not None:
                    try:
                        # Build background in transformed space
                        background = getattr(model, "background_data", None)
                        if background is None and preprocessor is not None:
                            try:
                                bg_df = df[feature_names].dropna().sample(min(100, len(df)), random_state=42)
                                background = preprocessor.transform(bg_df)
                            except Exception:
                                background = None

                        if background is not None:
                            Xtrans = preprocessor.transform(Xpred) if preprocessor is not None else Xpred.values
                            try:
                                import shap
                                explainer = shap.Explainer(model_step, background)
                                shap_values = explainer(Xtrans)
                                shap_arr = shap_values.values[0]
                                shap_df = pd.DataFrame({"feature": feature_names, "shap": shap_arr})
                                st.subheader("Feature contribution (SHAP)")
                                st.table(shap_df.sort_values(by="shap", key=abs, ascending=False).reset_index(drop=True))
                                fig = px.bar(shap_df, x="feature", y="shap", title="SHAP feature contributions")
                                st.plotly_chart(fig, width="stretch")
                                # try to show a SHAP beeswarm if available (matplotlib)
                                try:
                                    import matplotlib.pyplot as plt
                                    import shap as _shap
                                    plt.figure(figsize=(6, 4))
                                    # shap.plots.beeswarm works with the shap.Explanation object
                                    _shap.plots.beeswarm(shap_values)
                                    st.pyplot(plt.gcf())
                                    plt.clf()
                                except Exception:
                                    # non-fatal: beeswarm optional
                                    pass
                                explained = True
                            except Exception:
                                explained = False
                    except Exception:
                        # outer SHAP attempt failed; mark as not explained
                        explained = False

                # 2) Try model's native feature importances (RandomForest/XGBoost)
                if (
                    not explained
                    and model_step is not None
                    and hasattr(model_step, "feature_importances_")
                    and preprocessor is not None
                ):
                    try:
                        # map importances back to transformed feature names
                        try:
                            transformed_names = _get_preprocessor_feature_names(preprocessor, feature_names)
                        except Exception:
                            transformed_names = feature_names
                        importances = model_step.feature_importances_
                        imp_df = pd.DataFrame({"feature": transformed_names, "importance": importances})
                        # aggregate if needed back to original names by splitting on '_' for OHE
                        st.subheader("Model feature importances (model-reported)")
                        fig = px.bar(imp_df.sort_values("importance", ascending=False).head(30), x="feature", y="importance", title="Model feature importances")
                        st.plotly_chart(fig, width="stretch")
                        explained = True
                    except Exception:
                        explained = False

                # 3) Permutation importance fallback (ensure correct y selection)
                if not explained and model_step is not None and preprocessor is not None:
                    try:
                        sample_df = df[feature_names].dropna().head(200)
                        Xs = preprocessor.transform(sample_df)
                        target_col = _detect_target_col(df)
                        yfull = df.loc[sample_df.index, target_col]
                        imp = permutation_importance(model_step, Xs, yfull, n_repeats=5, random_state=42)
                        imp_df = pd.DataFrame({"feature": feature_names, "importance": imp.importances_mean})
                        fig = px.bar(imp_df.sort_values("importance", ascending=False), x="feature", y="importance", title="Permutation importance (fallback)")
                        st.plotly_chart(fig, width="stretch")
                        explained = True
                    except Exception:
                        explained = False

                if not explained:
                    # provide actionable message to user
                    if not shap_available:
                        st.info("SHAP is not installed in this environment. Install it with `pip install shap` to enable richer explanations.")
                    else:
                        st.info("Unable to compute SHAP or feature importances with the current model/data; see logs for details.")

    # --------------------- EDA tab ---------------------
    with tabs[1]:
        st.header("Exploratory Data Analysis")
        st.markdown("Visualizations reproduced from `car_price_prediction.ipynb`.")

        # avg price by Brand
        try:
            grp = df.groupby('Brand')['Price'].mean().reset_index().sort_values('Price', ascending=False)
            fig = px.bar(grp, x='Brand', y='Price', title='Average Price by Brand')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Brand-price plot unavailable.")

        # avg price by Fuel Type
        try:
            grp = df.groupby('Fuel Type')['Price'].mean().reset_index().sort_values('Price', ascending=False)
            fig = px.bar(grp, x='Fuel Type', y='Price', title='Average Price by Fuel Type')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Fuel-type plot unavailable.")

        # avg price by Condition
        try:
            grp = df.groupby('Condition')['Price'].mean().reset_index().sort_values('Price', ascending=False)
            fig = px.bar(grp, x='Condition', y='Price', title='Average Price by Condition')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Condition plot unavailable.")

        # count Transmission
        try:
            cnt = df['Transmission'].value_counts().reset_index()
            cnt.columns = ['Transmission', 'count']
            fig = px.bar(cnt, x='Transmission', y='count', title='Transmission Counts')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Transmission counts unavailable.")

        # Brand pie
        try:
            counts = df['Brand'].value_counts().reset_index()
            counts.columns = ['Brand', 'count']
            fig = px.pie(counts, names='Brand', values='count', title='Brand distribution')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Brand distribution unavailable.")

        # top models
        try:
            top = df['Model'].value_counts().head(10).reset_index()
            top.columns = ['Model', 'count']
            fig = px.bar(top, x='Model', y='count', title='Top 10 Models')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Top models plot unavailable.")

    # --------------------- Diagnostics tab ---------------------
    with tabs[2]:
        st.header("Model Diagnostics")
        st.markdown("Permutation importance and residuals (where applicable).")

        try:
            model = _get_or_train_model(df)
            feature_names = getattr(model, 'feature_names', [])
            if feature_names:
                # show permutation importance on a held-out sample
                target_col = _detect_target_col(df)
                sample_df = df.dropna(subset=feature_names + [target_col])
                if sample_df.shape[0] < 10:
                    st.info(f"Insufficient rows ({sample_df.shape[0]}) to compute diagnostics reliably. Need at least 10 rows with all features + target present.")
                else:
                    sample_df = sample_df.sample(n=min(200, len(sample_df)), random_state=42)
                    Xs = model.named_steps['preprocessor'].transform(sample_df[feature_names])
                    ys = sample_df[target_col]
                    imp = permutation_importance(model.named_steps['model'], Xs, ys, n_repeats=5, random_state=42)
                    imp_df = pd.DataFrame({'feature': feature_names, 'importance': imp.importances_mean})
                    fig = px.bar(imp_df.sort_values('importance', ascending=False), x='feature', y='importance', title='Permutation importance')
                    st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Diagnostics unavailable.")

    st.header("Data snapshot")
    st.dataframe(df.head(50))


if __name__ == "__main__":
    main()
