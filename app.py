import os
import sys

import pandas as pd
import streamlit as st
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

        try:
            return preprocessor.get_feature_names_out(input_features).tolist()
        except Exception:
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
                    for col in cols:
                        feature_names.append(col)
            return feature_names

    st.sidebar.markdown("---")

    model = None
    try:
        model = load_model(MODEL_PATH)
        if not hasattr(model, "feature_names"):
            model = None
    except Exception:
        model = None

    if model is not None:
        feature_list = [f for f in getattr(model, "feature_names", []) if f in df.columns]
    else:
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
            uniques = df[c].dropna().unique().tolist()[:20]
            default = uniques[0] if uniques else "missing"
            inputs[c] = st.sidebar.selectbox(c, options=uniques or ["missing"], index=0)

    st.sidebar.markdown("---")

    tabs = st.tabs(["Predict", "Exploratory Analysis", "Model Diagnostics"]) 

    with tabs[0]:
        if st.button("Predict"):
            model = _get_or_train_model(df)

            feature_names = getattr(model, "feature_names", [])
            if not feature_names:
                st.error("Model feature list is missing; retrain the model with `scripts/train.py`.")
            else:
                row = {}
                for f in feature_names:
                    if f in inputs:
                        row[f] = inputs[f]
                    else:
                        if f in df.columns:
                            if f in df.select_dtypes(include=["number"]).columns:
                                row[f] = float(df[f].median())
                            else:
                                nonnull = df[f].dropna().unique().tolist()
                                row[f] = nonnull[0] if nonnull else "missing"
                        else:
                            row[f] = 0.0

                Xpred = pd.DataFrame([row], columns=feature_names)
                pred = model.predict(Xpred)[0]
                st.metric("Predicted Price (demo)", f"${pred:,.0f}")

                preprocessor = model.named_steps.get("preprocessor")
                model_step = model.named_steps.get("model")

                try:
                    import importlib
                    shap_available = importlib.util.find_spec("shap") is not None
                except Exception:
                    shap_available = False

                explained = False
                if shap_available and model_step is not None:
                    try:
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
                                try:
                                    import matplotlib.pyplot as plt
                                    import shap as _shap
                                    plt.figure(figsize=(6, 4))
                                    _shap.plots.beeswarm(shap_values)
                                    st.pyplot(plt.gcf())
                                    plt.clf()
                                except Exception:
                                    pass
                                explained = True
                            except Exception:
                                explained = False
                    except Exception:
                        explained = False

                if (
                    not explained
                    and model_step is not None
                    and hasattr(model_step, "feature_importances_")
                    and preprocessor is not None
                ):
                    try:
                        try:
                            transformed_names = _get_preprocessor_feature_names(preprocessor, feature_names)
                        except Exception:
                            transformed_names = feature_names
                        importances = model_step.feature_importances_
                        imp_df = pd.DataFrame({"feature": transformed_names, "importance": importances})
                        st.subheader("Model feature importances (model-reported)")
                        fig = px.bar(imp_df.sort_values("importance", ascending=False).head(30), x="feature", y="importance", title="Model feature importances")
                        st.plotly_chart(fig, width="stretch")
                        explained = True
                    except Exception:
                        explained = False

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

                    if not shap_available:
                        st.info("SHAP is not installed in this environment. Install it with `pip install shap` to enable richer explanations.")
                    else:
                        st.info("Unable to compute SHAP or feature importances with the current model/data; see logs for details.")

    with tabs[1]:
        st.header("Exploratory Data Analysis")
        st.markdown("Visualizations reproduced from `car_price_prediction.ipynb`.")

        try:
            grp = df.groupby('Brand')['Price'].mean().reset_index().sort_values('Price', ascending=False)
            fig = px.bar(grp, x='Brand', y='Price', title='Average Price by Brand')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Brand-price plot unavailable.")

        try:
            grp = df.groupby('Fuel Type')['Price'].mean().reset_index().sort_values('Price', ascending=False)
            fig = px.bar(grp, x='Fuel Type', y='Price', title='Average Price by Fuel Type')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Fuel-type plot unavailable.")

        try:
            grp = df.groupby('Condition')['Price'].mean().reset_index().sort_values('Price', ascending=False)
            fig = px.bar(grp, x='Condition', y='Price', title='Average Price by Condition')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Condition plot unavailable.")

        try:
            cnt = df['Transmission'].value_counts().reset_index()
            cnt.columns = ['Transmission', 'count']
            fig = px.bar(cnt, x='Transmission', y='count', title='Transmission Counts')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Transmission counts unavailable.")

        try:
            counts = df['Brand'].value_counts().reset_index()
            counts.columns = ['Brand', 'count']
            fig = px.pie(counts, names='Brand', values='count', title='Brand distribution')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Brand distribution unavailable.")

        try:
            top = df['Model'].value_counts().head(10).reset_index()
            top.columns = ['Model', 'count']
            fig = px.bar(top, x='Model', y='count', title='Top 10 Models')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Top models plot unavailable.")

    with tabs[2]:
        st.header("Model Diagnostics")
        st.markdown("Permutation importance and residuals (where applicable).")

        try:
            model = _get_or_train_model(df)
        except Exception as e:
            st.error(f"Could not load or train model for diagnostics: {e}")
        else:
            feature_names = getattr(model, 'feature_names', [])
            if not feature_names:
                st.info("Model artifact is missing `feature_names`; retrain and save pipeline with metadata.")
            else:
                target_col = _detect_target_col(df)
                if target_col not in df.columns:
                    st.info(f"Target column '{target_col}' not found in data; diagnostics require the target.")
                else:
                    sample_df = df.dropna(subset=feature_names + [target_col])
                    if sample_df.shape[0] < 10:
                        st.info(
                            f"Insufficient rows ({sample_df.shape[0]}) to compute diagnostics reliably."
                            " Need at least 10 rows with all features + target present."
                        )
                    else:
                        sample_df = sample_df.sample(n=min(200, len(sample_df)), random_state=42)
                        preproc = model.named_steps.get('preprocessor')
                        model_step = model.named_steps.get('model')
                        if preproc is None or model_step is None:
                            st.info("Saved pipeline is missing preprocessor or model steps required for diagnostics.")
                        else:
                            try:
                                Xs = preproc.transform(sample_df[feature_names])
                                ys = sample_df[target_col]
                                imp = permutation_importance(model_step, Xs, ys, n_repeats=5, random_state=42)
                                try:
                                    transformed_names = _get_preprocessor_feature_names(preproc, feature_names)
                                except Exception:
                                    transformed_names = None
                                if transformed_names and len(transformed_names) == imp.importances_mean.shape[0]:
                                    imp_df = pd.DataFrame({'feature': transformed_names, 'importance': imp.importances_mean})
                                else:
                                    imp_df = pd.DataFrame({'feature': feature_names[: imp.importances_mean.shape[0]], 'importance': imp.importances_mean})
                                fig = px.bar(imp_df.sort_values('importance', ascending=False), x='feature', y='importance', title='Permutation importance')
                                st.plotly_chart(fig, width="stretch")
                            except Exception as e:
                                st.error(f"Error computing diagnostics: {e}")

    st.header("Data snapshot")
    st.dataframe(df.head(50))


if __name__ == "__main__":
    main()
