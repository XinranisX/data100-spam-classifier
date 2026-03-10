from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from spam_classifier.pipeline import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spam/ham classifier")
    parser.add_argument("--train_csv", type=str, required=True, help="CSV with columns: email, spam")
    parser.add_argument("--out_dir", type=str, default="models", help="Output directory for model artifacts")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    result = train_model(train_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.model, out_dir / "spam_classifier.joblib")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {"best_params": result.best_params, "feature_dim": result.feature_dim},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved model to:", out_dir / "spam_classifier.joblib")
    print("Best params:", result.best_params)


if __name__ == "__main__":
    main()