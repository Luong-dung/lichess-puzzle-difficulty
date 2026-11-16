"""Tune Logistic Regression trên train (mục 1.4c).

Giống tune_naive_bayes.py nhưng lưới siêu tham số khác và
`class_weight` được thử để đối phó mất cân bằng lớp.
"""
import argparse
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_sample, split_groups
from src.evaluation import bin_target, classification_metrics
from src.features import make_features
from src.models.logistic_regression import build_model as build_logistic


PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}


def _make_pipeline(seed, representation, dimensions, numeric):
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["turn"]),
    ])
    steps = [("preprocess", preprocess)]
    if representation == "pca":
        steps.append(("reduce", PCA(n_components=dimensions, random_state=seed)))
    elif representation == "svd":
        steps.append(("reduce", TruncatedSVD(n_components=dimensions, random_state=seed)))
    steps.append(("model", build_logistic(seed)))
    return Pipeline(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        choices=[0.2, 0.3, 0.4])
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--bins", type=int, default=3, choices=[3, 4])
    parser.add_argument("--output", default="outputs/tuning_logistic.json")
    args = parser.parse_args()

    frame = load_sample(args.data, args.sample, args.seed)
    X, y = make_features(frame), frame["Rating"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    train, valid = split_groups(frame, args.validation_fraction, args.seed)
    dimensions = max(1, (len(numeric) + 2) // 3)

    X_train, X_valid = X.iloc[train], X.iloc[valid]
    y_train_raw, y_valid_raw = y.iloc[train], y.iloc[valid]
    y_train_bin, y_valid_bin, thresholds = bin_target(y_train_raw, y_valid_raw, args.bins)
    groups_train = frame.iloc[train]["game_id"].to_numpy()

    results = []
    for representation in ("original", "pca", "svd"):
        pipeline = _make_pipeline(args.seed, representation, dimensions, numeric)
        grid = {f"model__{k}": v for k, v in PARAM_GRID.items()}
        search = GridSearchCV(
            pipeline, grid,
            cv=StratifiedGroupKFold(n_splits=args.cv),
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X_train, y_train_bin, groups=groups_train)

        best = search.best_estimator_
        best.fit(X_train, y_train_bin)
        train_metrics = classification_metrics(y_train_bin, best.predict(X_train))
        valid_metrics = classification_metrics(y_valid_bin, best.predict(X_valid))
        result = {
            "model": "logistic",
            "representation": representation,
            "validation_fraction": args.validation_fraction,
            "bins": args.bins,
            "thresholds": thresholds,
            "best_params": {
                k.split("__", 1)[-1]: str(v) for k, v in search.best_params_.items()
            },
            "best_cv_accuracy": float(search.best_score_),
            "train": train_metrics,
            "validation": valid_metrics,
            "overfit_gap_accuracy": float(train_metrics["accuracy"] - valid_metrics["accuracy"]),
        }
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()