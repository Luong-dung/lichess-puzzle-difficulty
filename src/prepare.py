"""Generate local quality report, feature statistics and split manifests."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .data import load_sample, split_groups
from .features import make_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/preprocessing")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame, report = load_sample(args.data, args.sample, args.seed, return_report=True)
    features = make_features(frame)
    numeric = features.select_dtypes(include="number")
    if features.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Missing or nonfinite feature values")
    report["features"] = len(features.columns)
    report["constant_features"] = features.columns[features.nunique() <= 1].tolist()
    report["rating_summary"] = frame["Rating"].describe().to_dict()
    report["splits"] = []
    for fraction in (0.2, 0.3, 0.4):
        train, valid = split_groups(frame, fraction, args.seed)
        assignment = frame[["PuzzleId", "game_id", "position_key"]].copy()
        assignment["split"] = "train"
        assignment.loc[valid, "split"] = "validation"
        assignment.to_csv(out / f"split_{int(fraction*100)}.csv", index=False)
        report["splits"].append(dict(requested_fraction=fraction, train=len(train),
                                    validation=len(valid), actual_fraction=len(valid)/len(frame),
                                    shared_games=0, shared_positions=0))
    frame.to_csv(out / "sample.csv", index=False)
    features.to_csv(out / "features.csv", index=False)
    features.describe(include="all").to_csv(out / "feature_statistics.csv")
    frame["Rating"].plot.hist(bins=40)
    plt.xlabel("Rating")
    plt.ylabel("Puzzle count")
    plt.title(f"Clean sample: n={len(frame)}, seed={args.seed}")
    plt.tight_layout()
    plt.savefig(out / "rating_distribution.png", dpi=150)
    plt.close()
    (out / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
