import os
import sys
import pandas as pd

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from scripts.utils import train_demo_model
from sklearn.inspection import permutation_importance


def test_diagnostics_permutation_importance():
    n = 60
    df = pd.DataFrame({
        "Price": (10000 + (pd.Series(range(n)) * 50)).astype(float),
        "year": [2010 + (i % 10) for i in range(n)],
        "mileage": [20000 + (i * 100) for i in range(n)],
        "engine_size": [1.2 + (i % 4) * 0.4 for i in range(n)],
        "make": ["makeA" if i % 3 == 0 else "makeB" for i in range(n)],
    })

    pipeline = train_demo_model(df, target="Price", sample_frac=1.0)

    assert hasattr(pipeline, "feature_names")
    feature_names = pipeline.feature_names
    assert len(feature_names) > 0

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
    try:
        transformed_names = preprocessor.get_feature_names_out(feature_names).tolist()
    except Exception:
        transformed_names = None

    if transformed_names is not None:
        assert imp.importances_mean.shape[0] == len(transformed_names)
    else:

        assert imp.importances_mean.shape[0] > 0

    tol = 1e-6
    assert (imp.importances_mean >= -tol).all()
