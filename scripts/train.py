import argparse
import logging
import os
import sys

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from scripts.utils import load_data, save_model, train_demo_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="car_price_prediction_.csv")
    parser.add_argument("--out", default="models/demo_model.joblib")
    parser.add_argument("--sample-frac", type=float, default=0.25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = load_data(args.data)
    pipeline = train_demo_model(df, sample_frac=args.sample_frac)
    save_model(pipeline, args.out)
    logging.info(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
