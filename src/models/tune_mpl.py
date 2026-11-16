import argparse
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_sample, split_groups
from src.evaluation import regression_metrics
from src.features import make_features
from src.models.mlp import build_model as build_mlp


PARAM_GRID = {
    "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "learning_rate_init": [1e-3, 5e-3],
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
    steps.append(("model", build_mlp(seed)))
    return Pipeline(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        choices=[0.2, 0.3, 0.4])
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--output", default="outputs/tuning_mlp.json")
    args = parser.parse_args()

    frame = load_sample(args.data, args.sample, args.seed)
    X, y = make_features(frame), frame["Rating"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    train, valid = split_groups(frame, args.validation_fraction, args.seed)
    dimensions = max(1, (len(numeric) + 2) // 3)

    X_train, y_train = X.iloc[train], y.iloc[train]
    X_valid, y_valid = X.iloc[valid], y.iloc[valid]
    groups_train = frame.iloc[train]["game_id"].to_numpy()

    results = []
    for representation in ("original", "pca", "svd"):
        pipeline = _make_pipeline(args.seed, representation, dimensions, numeric)
        wrapped = TransformedTargetRegressor(
            regressor=pipeline, transformer=StandardScaler()
        )
        grid = {f"regressor__model__{k}": v for k, v in PARAM_GRID.items()}
        search = GridSearchCV(
            wrapped, grid,
            cv=GroupKFold(n_splits=args.cv),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train, groups=groups_train)

        best = search.best_estimator_
        best.fit(X_train, y_train)
        train_metrics = regression_metrics(y_train, best.predict(X_train))
        valid_metrics = regression_metrics(y_valid, best.predict(X_valid))
        result = {
            "model": "mlp",
            "representation": representation,
            "validation_fraction": args.validation_fraction,
            "best_params": {
                k.split("__", 2)[-1]: str(v) for k, v in search.best_params_.items()
            },
            "best_cv_rmse": float(-search.best_score_),
            "train": train_metrics,
            "validation": valid_metrics,
            "overfit_gap_rmse": float(valid_metrics["rmse"] - train_metrics["rmse"]),
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