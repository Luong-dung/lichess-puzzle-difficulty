"""Run baseline experiments, with preprocessing fitted only on training data."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import load_sample
from .features import make_features
from .evaluation import regression_metrics
from .models import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/metrics.csv")
    args = parser.parse_args()
    frame = load_sample(args.data, args.sample, args.seed)
    X, y = make_features(frame), frame["Rating"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    rows = []
    for ratio in (0.2, 0.3, 0.4):
        train, valid = next(GroupShuffleSplit(n_splits=1, test_size=ratio,
                           random_state=args.seed).split(X, y, groups=frame["game_id"]))
        for representation in ("original", "pca", "svd"):
            for name in args.models:
                preprocess = ColumnTransformer([
                    ("numeric", StandardScaler(), numeric),
                    ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["turn"])])
                steps = [("preprocess", preprocess)]
                dimensions = max(1, (len(numeric) + 2) // 3)
                if representation == "pca":
                    steps.append(("reduce", PCA(n_components=dimensions, random_state=args.seed)))
                elif representation == "svd":
                    steps.append(("reduce", TruncatedSVD(n_components=dimensions, random_state=args.seed)))
                steps.append(("model", MODELS[name](args.seed)))
                model = TransformedTargetRegressor(regressor=Pipeline(steps), transformer=StandardScaler())
                model.fit(X.iloc[train], y.iloc[train])
                for split, indices in (("train", train), ("validation", valid)):
                    rows.append(dict(model=name, representation=representation,
                                     validation_fraction=ratio, split=split, n=len(indices),
                                     **regression_metrics(y.iloc[indices], model.predict(X.iloc[indices]))))
                print(name, representation, ratio, flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
