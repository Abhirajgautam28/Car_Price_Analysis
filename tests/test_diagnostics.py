import os
import sys
import pandas as pd

# ensure project root is on sys.path so tests import local `scripts` package
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from scripts.utils import train_demo_model
from sklearn.inspection import permutation_importance


def test_diagnostics_permutation_importance():
    # create synthetic dataset with sufficient rows to compute diagnostics
    n = 60
    df = pd.DataFrame({
        "Price": (10000 + (pd.Series(range(n)) * 50)).astype(float),
        "year": [2010 + (i % 10) for i in range(n)],
        "mileage": [20000 + (i * 100) for i in range(n)],
        "engine_size": [1.2 + (i % 4) * 0.4 for i in range(n)],
        "make": ["makeA" if i % 3 == 0 else "makeB" for i in range(n)],
    })

    # train demo pipeline (use all rows)
    pipeline = train_demo_model(df, target="Price", sample_frac=1.0)

    assert hasattr(pipeline, "feature_names")
    feature_names = pipeline.feature_names
    assert len(feature_names) > 0

    # build a sample for permutation importance similar to app logic
    target_col = None
    for c in df.columns:
        if c.lower() == "price":
            target_col = c
            break
    assert target_col is not None

    sample_df = df.dropna(subset=feature_names + [target_col])
    assert sample_df.shape[0] >= 10

    sample_df = sample_df.sample(n=min(200, len(sample_df)), random_state=42)
    preprocessor = pipeline.named_steps["preprocessor"]
    model_step = pipeline.named_steps["model"]

    Xs = preprocessor.transform(sample_df[feature_names])
    ys = sample_df[target_col]

    imp = permutation_importance(model_step, Xs, ys, n_repeats=3, random_state=42)
    # determine transformed feature names length (ColumnTransformer may expand categoricals)
    try:
        transformed_names = preprocessor.get_feature_names_out(feature_names).tolist()
    except Exception:
        # best-effort fallback: if onehot encoder created extra columns, allow that
        transformed_names = None

    # sanity checks: importance vector length should match transformed matrix width
    if transformed_names is not None:
        assert imp.importances_mean.shape[0] == len(transformed_names)
    else:
        # if we couldn't resolve names, at minimum ensure importances length > 0
        assert imp.importances_mean.shape[0] > 0
    # allow tiny negative values due to sampling noise
    tol = 1e-6
    assert (imp.importances_mean >= -tol).all()
