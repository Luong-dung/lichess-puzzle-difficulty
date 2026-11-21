"""Member 1: OLS vs Ridge, alpha tuned by GroupKFold on train; overfit and residual analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from .data import load_sample, split_groups
from .evaluation import regression_metrics
from .features import make_features
from .models.linear import ALPHAS, build_model
from .preprocessing import build_regressor, clean_names
from .plot_style import plt, save, SERIES, SEQUENTIAL, MUTED, TEXT

ALPHA_PARAMETER = "regressor__model__alpha"


def tune_ridge(numeric, X, y, groups, representation, seed, folds=5):
    search = GridSearchCV(build_regressor(numeric, build_model(seed), representation, seed),
                          {ALPHA_PARAMETER: ALPHAS}, cv=GroupKFold(n_splits=folds),
                          scoring="neg_root_mean_squared_error", return_train_score=True)
    return search.fit(X, y, groups=groups)


def plot_tuning(search, out):
    results = pd.DataFrame(search.cv_results_)
    alphas = results[f"param_{ALPHA_PARAMETER}"].astype(float)
    figure, ax = plt.subplots(figsize=(7, 3.6))
    for color, split, label in ((SERIES[0], "train", "Train (trong fold)"), (SERIES[1], "test", "Validation fold")):
        mean, std = -results[f"mean_{split}_score"], results[f"std_{split}_score"]
        ax.fill_between(alphas, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(alphas, mean, color=color, marker="o", markersize=4, label=label)
    best = search.best_params_[ALPHA_PARAMETER]
    ax.axvline(best, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate(f"alpha tốt nhất = {best:g}", (best, 1), xycoords=("data", "axes fraction"),
                fontsize=8, color=TEXT, va="top", xytext=(4, -4), textcoords="offset points")
    ax.set(xscale="log", xlabel="alpha (thang log)", ylabel="RMSE (Rating)",
           title="Ridge: chọn alpha bằng GroupKFold 5 fold trên train")
    ax.legend(loc="upper left")
    save(figure, out / "ridge_tuning_curve.png")
    results.to_csv(out / "ridge_cv_results.csv", index=False)


def plot_learning_curve(numeric, X, y, frame, train, valid, alpha, seed, out):
    games = np.random.default_rng(seed).permutation(np.asarray(frame["game_id"].iloc[train].unique(), dtype=object))
    rows = []
    for share in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        chosen = set(games[:max(2, int(len(games) * share))])
        subset = train[frame["game_id"].iloc[train].isin(chosen).to_numpy()]
        model = build_regressor(numeric, build_model(seed, alpha), "original", seed)
        model.fit(X.iloc[subset], y.iloc[subset])
        rows.append(dict(train_rows=len(subset),
                         train_rmse=regression_metrics(y.iloc[subset], model.predict(X.iloc[subset]))["rmse"],
                         validation_rmse=regression_metrics(y.iloc[valid], model.predict(X.iloc[valid]))["rmse"]))
    curve = pd.DataFrame(rows)
    curve.to_csv(out / "ridge_learning_curve.csv", index=False)
    figure, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(curve["train_rows"], curve["train_rmse"], color=SERIES[0], marker="o", markersize=5, label="Train")
    ax.plot(curve["train_rows"], curve["validation_rmse"], color=SERIES[1], marker="s", markersize=5,
            label="Validation")
    ax.set(xlabel="Số bài trong tập train", ylabel="RMSE (Rating)",
           title=f"Ridge (alpha={alpha:g}): đường học theo cỡ tập train")
    ax.legend(loc="upper right")
    save(figure, out / "ridge_learning_curve.png")


def plot_residuals(y_true, prediction, out):
    y_true, prediction = np.asarray(y_true, float), np.asarray(prediction, float)
    residual = y_true - prediction
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    low, high = min(y_true.min(), prediction.min()), max(y_true.max(), prediction.max())
    image = axes[0].hexbin(y_true, prediction, gridsize=35, cmap=SEQUENTIAL, mincnt=1, bins="log", linewidths=0)
    axes[0].plot([low, high], [low, high], color=MUTED, linewidth=1, linestyle="--", label="Dự đoán đúng (y = x)")
    axes[0].set(xlabel="Rating thực tế", ylabel="Rating dự đoán", title="Thực tế và dự đoán")
    axes[0].legend(loc="upper left")
    figure.colorbar(image, ax=axes[0], label="Số bài (log)")
    image = axes[1].hexbin(prediction, residual, gridsize=35, cmap=SEQUENTIAL, mincnt=1, bins="log", linewidths=0)
    axes[1].axhline(0, color=MUTED, linewidth=1, linestyle="--")
    axes[1].set(xlabel="Rating dự đoán", ylabel="Phần dư (thực tế − dự đoán)", title="Phần dư theo giá trị dự đoán")
    figure.colorbar(image, ax=axes[1], label="Số bài (log)")
    bins = pd.qcut(y_true, 10, duplicates="drop")
    by_bin = pd.DataFrame({"bin": bins, "residual": residual}).groupby("bin", observed=True)["residual"]
    labels = [f"{interval.left:.0f}–{interval.right:.0f}" for interval in by_bin.mean().index]
    axes[2].bar(range(len(labels)), by_bin.mean().values, color=SERIES[0], width=0.7, label="Phần dư trung bình")
    axes[2].plot(range(len(labels)), by_bin.apply(lambda s: s.abs().mean()).values, color=SERIES[1],
                 marker="o", markersize=5, label="MAE")
    axes[2].axhline(0, color=MUTED, linewidth=0.8)
    axes[2].set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    axes[2].set(xlabel="Nhóm thập phân vị Rating thực tế", ylabel="Rating",
                title="Sai số theo mức Rating thực tế")
    axes[2].legend(loc="upper center")
    save(figure, out / "ridge_residuals.png")
    return pd.DataFrame({"rating_bin": labels, "mean_residual": by_bin.mean().values,
                         "mae": by_bin.apply(lambda s: s.abs().mean()).values, "n": by_bin.size().values})


def plot_coefficients(model, out):
    pipeline = model.regressor_
    names = clean_names(pipeline.named_steps["preprocess"].get_feature_names_out())
    # Rating points per one standard deviation of each standardized feature.
    coefficients = pd.Series(pipeline.named_steps["model"].coef_ * model.transformer_.scale_[0], index=names)
    coefficients = coefficients[coefficients.abs() > 1e-9].sort_values()
    figure, ax = plt.subplots(figsize=(6.5, 0.28 * len(coefficients) + 1))
    ax.barh(coefficients.index, coefficients.values, color=SERIES[0], height=0.7)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set(xlabel="Thay đổi Rating khi đặc trưng tăng 1 độ lệch chuẩn",
           title="Hệ số Ridge trên đặc trưng chuẩn hóa")
    ax.tick_params(axis="y", labelsize=7)
    save(figure, out / "ridge_coefficients.png")
    return coefficients


def run_linear(frame, features, out, seed=42, folds=5, detail_fraction=0.2):
    out.mkdir(parents=True, exist_ok=True)
    X, y = features, frame["Rating"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    rows, best_alphas, detail = [], {}, None
    for fraction in (0.2, 0.3, 0.4):
        train, valid = split_groups(frame, fraction, seed)
        for representation in ("original", "pca", "svd"):
            search = tune_ridge(numeric, X.iloc[train], y.iloc[train],
                                frame["game_id"].iloc[train], representation, seed, folds)
            alpha = float(search.best_params_[ALPHA_PARAMETER])
            best_alphas[f"{representation}_{fraction}"] = alpha
            candidates = {"mean_baseline": build_regressor(numeric, DummyRegressor(), representation, seed),
                          "ols": build_regressor(numeric, LinearRegression(), representation, seed),
                          "ridge_tuned": search.best_estimator_}
            for name, model in candidates.items():
                if name != "ridge_tuned":
                    model.fit(X.iloc[train], y.iloc[train])
                for split, indices in (("train", train), ("validation", valid)):
                    rows.append(dict(model=name, representation=representation, validation_fraction=fraction,
                                     alpha=alpha if name == "ridge_tuned" else None,
                                     cv_rmse=-search.best_score_ if name == "ridge_tuned" else None,
                                     split=split, n=len(indices),
                                     **regression_metrics(y.iloc[indices], model.predict(X.iloc[indices]))))
            if fraction == detail_fraction and representation == "original":
                detail = (search, train, valid)
            print("linear", representation, fraction, "alpha", alpha, flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "linear_metrics.csv", index=False)

    search, train, valid = detail
    plot_tuning(search, out)
    alpha = float(search.best_params_[ALPHA_PARAMETER])
    plot_learning_curve(numeric, X, y, frame, train, valid, alpha, seed, out)
    model = search.best_estimator_
    residuals = plot_residuals(y.iloc[valid], model.predict(X.iloc[valid]), out)
    residuals.to_csv(out / "ridge_residuals_by_rating.csv", index=False)
    coefficients = plot_coefficients(model, out)
    coefficients.to_csv(out / "ridge_coefficients.csv", header=["rating_per_std"])

    pivot = metrics.pivot_table(index=["validation_fraction", "representation", "model"],
                                columns="split", values=["rmse", "mae", "r2"])
    pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
    pivot["rmse_gap"] = pivot["rmse_validation"] - pivot["rmse_train"]
    pivot.round(4).to_csv(out / "linear_summary.csv")
    summary = dict(seed=seed, folds=folds, best_alpha=best_alphas,
                   detail=dict(validation_fraction=detail_fraction, representation="original", alpha=alpha),
                   validation=pivot.reset_index().query("model == 'ridge_tuned'")[
                       ["validation_fraction", "representation", "rmse_train", "rmse_validation",
                        "mae_validation", "r2_validation", "rmse_gap"]].round(4).to_dict("records"))
    (out / "linear_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="outputs/linear")
    args = parser.parse_args()
    frame = load_sample(args.data, args.sample, args.seed)
    summary = run_linear(frame, make_features(frame), Path(args.output), args.seed, args.folds)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()