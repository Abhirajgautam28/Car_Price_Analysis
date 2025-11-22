import os
import sys
import pandas as pd

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from scripts.utils import train_demo_model


def test_train_demo_model_basic():

    df = pd.DataFrame(
        {
            "price": [10000, 12000, 9000, 15000],
            "year": [2012, 2015, 2010, 2018],
            "mileage": [50000, 30000, 70000, 20000],
            "engine_size": [1.6, 2.0, 1.4, 2.4],
            "make": ["a", "b", "a", "c"],
        }
    )
    pipeline = train_demo_model(df, target="price", sample_frac=1.0)
    preds = pipeline.predict(df[[c for c in df.columns if c != "price"]])
    assert len(preds) == df.shape[0]
    assert preds.dtype.kind in "fiu"
