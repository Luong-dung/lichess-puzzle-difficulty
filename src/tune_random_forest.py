"""Thành viên 2: tuning Random Forest bằng GroupKFold trên train, rồi đánh giá trên validation.

Chạy:  python -m src.tune_random_forest --n-iter 20
Dùng lại mẫu đã làm sạch (outputs/preprocessing/sample.csv) nếu có để khỏi quét lại CSV gốc.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, ParameterSampler, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .data import load_sample, split_groups
from .evaluation import regression_metrics
from .features import make_features

GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [8, 12, 16, 24, None],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_features": [0.3, 0.5, 0.8, 1.0, "sqrt"],
}


def load_frame(args):
    cache = Path(args.cache)
    if cache.exists():
        frame = pd.read_csv(cache, dtype={"game_id": "string"})
        print(f"Đọc mẫu đã làm sạch từ {cache}: {len(frame)} dòng", flush=True)
        return frame
    print("Không có cache, quét CSV gốc (vài phút)...", flush=True)
    return load_sample(args.data, args.sample, args.seed)


def build_pipeline(numeric, seed, **params):
    preprocess = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["turn"])])
    model = RandomForestRegressor(n_jobs=-1, random_state=seed, **params)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--cache", default="outputs/preprocessing/sample.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--output", default="outputs/random_forest")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args)
    X, y = make_features(frame), frame["Rating"].to_numpy()
    numeric = X.select_dtypes(include="number").columns.tolist()
    train, valid = split_groups(frame, args.validation, args.seed)
    X_train, y_train, groups = X.iloc[train], y[train], frame["game_id"].iloc[train]
    print(f"{X.shape[1]} đặc trưng; train {len(train)}, validation {len(valid)}", flush=True)

    # 1. Tìm siêu tham số: chỉ dùng train, chia fold theo ván để không rò rỉ.
    folds = list(GroupKFold(n_splits=args.folds).split(X_train, y_train, groups))
    candidates = list(ParameterSampler(GRID, args.n_iter, random_state=args.seed))
    rows, start = [], time.time()
    for i, params in enumerate(candidates, 1):
        fold_rows = []
        for fit_idx, val_idx in folds:
            model = build_pipeline(numeric, args.seed, **params)
            model.fit(X_train.iloc[fit_idx], y_train[fit_idx])
            fold_rows.append({**{f"train_{k}": v for k, v in regression_metrics(
                                  y_train[fit_idx], model.predict(X_train.iloc[fit_idx])).items()},
                              **{f"cv_{k}": v for k, v in regression_metrics(
                                  y_train[val_idx], model.predict(X_train.iloc[val_idx])).items()}})
        summary = pd.DataFrame(fold_rows)
        row = {"candidate": i - 1, **params,**summary.mean().to_dict(), "cv_rmse_std": summary["cv_rmse"].std()}
        rows.append(row)
        elapsed = time.time() - start
        print(f"[{i}/{len(candidates)}] {params} cv_rmse={row['cv_rmse']:.1f} "
              f"({elapsed:.0f}s, còn ~{elapsed / i * (len(candidates) - i):.0f}s)", flush=True)
    results = pd.DataFrame(rows).sort_values("cv_rmse")
    # Quy tắc 1-SE: các cấu hình có CV RMSE trong 1 sai số chuẩn của cấu hình tốt nhất coi như
    # ngang nhau; chọn cấu hình ít overfit nhất (khoảng cách CV - train nhỏ nhất).
    results["overfit_gap"] = results["cv_rmse"] - results["train_rmse"]
    top = results.iloc[0]
    threshold = top["cv_rmse"] + top["cv_rmse_std"] / np.sqrt(args.folds)
    results["within_1se"] = results["cv_rmse"] <= threshold
    chosen = results[results["within_1se"]].sort_values("overfit_gap").iloc[0]
    results.to_csv(out / "cv_results.csv", index=False)
    best = candidates[int(chosen["candidate"])]
    selection = dict(rule="1-SE, nhỏ nhất overfit_gap", threshold_cv_rmse=float(threshold),
                     lowest_cv_params=candidates[int(top["candidate"])], lowest_cv_rmse=float(top["cv_rmse"]),
                     chosen_cv_rmse=float(chosen["cv_rmse"]), chosen_overfit_gap=float(chosen["overfit_gap"]),
                     n_within_1se=int(results["within_1se"].sum()))

    # 2. Fit lại trên toàn bộ train, đánh giá một lần trên validation.
    model = build_pipeline(numeric, args.seed, **best)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X.iloc[valid])
    metrics = {"train": regression_metrics(y_train, pred_train),
               "validation": regression_metrics(y[valid], pred_valid),
               "baseline_mean_validation": regression_metrics(
                   y[valid], np.full(len(valid), y_train.mean()))}
    report = dict(best_params=best, selection=selection, metrics=metrics, n_features=X.shape[1],
                  n_train=len(train), n_validation=len(valid), folds=args.folds,
                  n_iter=args.n_iter, seed=args.seed, tuning_seconds=round(time.time() - start))
    (out / "best.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                   encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)

    # 3. Ảnh hưởng từng siêu tham số (overfit: khoảng cách train - CV).
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key in zip(axes, GRID):
        g = results.assign(**{key: results[key].astype(str)}).groupby(key)[["train_rmse", "cv_rmse"]].mean()
        g.plot(marker="o", ax=ax, title=key)
        ax.set_ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out / "hyperparameter_effect.png", dpi=150)
    plt.close()

    # 4. Learning curve với tham số tốt nhất.
    print("Learning curve...", flush=True)
    sizes, tr, va = learning_curve(build_pipeline(numeric, args.seed, **best), X_train, y_train,
                                   groups=groups, cv=GroupKFold(n_splits=args.folds),
                                   train_sizes=np.linspace(0.1, 1.0, 6),
                                   scoring="neg_root_mean_squared_error")
    pd.DataFrame({"train_size": sizes, "train_rmse": -tr.mean(1), "cv_rmse": -va.mean(1)}).to_csv(
        out / "learning_curve.csv", index=False)
    plt.plot(sizes, -tr.mean(1), "o-", label="train")
    plt.plot(sizes, -va.mean(1), "o-", label="CV")
    plt.xlabel("Số mẫu train")
    plt.ylabel("RMSE")
    plt.title("Learning curve Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "learning_curve.png", dpi=150)
    plt.close()

    # 5. Phần dư trên validation.
    residual = y[valid] - pred_valid
    pd.DataFrame({"PuzzleId": frame["PuzzleId"].iloc[valid].to_numpy(), "rating": y[valid],
                  "prediction": pred_valid, "residual": residual}).to_csv(out / "residuals.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(y[valid], pred_valid, s=3, alpha=0.3)
    lims = [y.min(), y.max()]
    axes[0].plot(lims, lims, "r--")
    axes[0].set(xlabel="Rating thật", ylabel="Dự đoán", title="Dự đoán và thực tế")
    axes[1].scatter(pred_valid, residual, s=3, alpha=0.3)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set(xlabel="Dự đoán", ylabel="Phần dư", title="Phần dư theo dự đoán")
    axes[2].hist(residual, bins=50)
    axes[2].set(xlabel="Phần dư", title="Phân phối phần dư")
    plt.tight_layout()
    plt.savefig(out / "residuals.png", dpi=150)
    plt.close()

    # 6. Feature importance: impurity (trên train) và permutation (trên validation).
    names = model.named_steps["preprocess"].get_feature_names_out()
    impurity = pd.Series(model.named_steps["model"].feature_importances_, index=names)
    perm = permutation_importance(model, X.iloc[valid], y[valid], n_repeats=5, random_state=args.seed,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
    importance = pd.DataFrame({"feature": X.columns, "permutation_mean": perm.importances_mean,
                               "permutation_std": perm.importances_std}).sort_values(
                                   "permutation_mean", ascending=False)
    importance.to_csv(out / "permutation_importance.csv", index=False)
    impurity.sort_values(ascending=False).rename("impurity").to_csv(out / "impurity_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    impurity.sort_values().tail(15).plot.barh(ax=axes[0], title="Impurity importance (train)")
    importance.set_index("feature")["permutation_mean"].sort_values().tail(15).plot.barh(
        ax=axes[1], xerr=None, title="Permutation importance (validation, tăng RMSE)")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Đã lưu kết quả vào {out}", flush=True)


if __name__ == "__main__":
    main()
