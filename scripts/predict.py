from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from spam_classifier.pipeline import process_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict spam/ham on a CSV with an 'email' column")
    parser.add_argument("--model", type=str, required=True, help="Path to spam_classifier.joblib")
    parser.add_argument("--in_csv", type=str, required=True, help="Input CSV with column: email")
    parser.add_argument("--out_csv", type=str, default="outputs/predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.in_csv)

    X = process_data(df)
    proba_spam = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    out = df.copy()
    out["pred_spam"] = pred
    out["proba_spam"] = proba_spam

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()