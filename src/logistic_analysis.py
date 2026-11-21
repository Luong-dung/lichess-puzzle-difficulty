"""Thành viên 1: Logistic Regression trên nhãn khoảng Rating của Thành viên 3.

Nhãn: ngưỡng phân vị học trên train (src.classification.fit_rating_bins), 3 và 4 khoảng.
Biểu diễn: đặc trưng gốc và PCA còn 1/3 số chiều. C chọn bằng GroupKFold 5 fold theo ván
trên train, tiêu chí macro-F1; validation chỉ dùng một lần để báo cáo.

Chạy:  python -m src.logistic_analysis      -> outputs/logistic/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold

from .classification import (_build_pipeline, _plot_confusion_matrix, apply_rating_bins,
                             evaluate_classification, fit_rating_bins)
from .data import split_groups
from .features import make_features
from .models.logistic_regression import C_GRID, build_model
from .plot_style import MUTED, SERIES, TEXT, plt, save
from .preprocessing import reduced_dimensions
from .sample_cache import load_frame

C_PARAMETER = "model__C"
FRACTIONS = (0.2, 0.3, 0.4)
REPRESENTATIONS = ("original", "pca")


def tune_logistic(numeric, X, y, groups, representation, seed, folds=5):
    search = GridSearchCV(_build_pipeline(numeric, build_model, representation, seed),
                          {C_PARAMETER: C_GRID}, cv=GroupKFold(n_splits=folds),
                          scoring="f1_macro", return_train_score=True)
    return search.fit(X, y, groups=groups)


def plot_tuning(searches, n_bins, out):
    figure, ax = plt.subplots(figsize=(7, 3.6))
    for color, (representation, search) in zip(SERIES, searches.items()):
        results = pd.DataFrame(search.cv_results_)
        c = results[f"param_{C_PARAMETER}"].astype(float)
        ax.plot(c, results["mean_train_score"], color=color, linestyle="--", linewidth=1.2,
                label=f"{representation}: train")
        ax.plot(c, results["mean_test_score"], color=color, marker="o", markersize=4,
                label=f"{representation}: fold validation")
        best = search.best_params_[C_PARAMETER]
        ax.axvline(best, color=color, linewidth=0.8, linestyle=":")
    ax.set(xscale="log", xlabel="C (thang log, C lớn = phạt L2 yếu)", ylabel="Macro-F1",
           title=f"Logistic: chọn C bằng GroupKFold 5 fold ({n_bins} khoảng, val 20%)")
    ax.legend(loc="lower right")
    save(figure, out / f"lr_tuning_curve_{n_bins}bins.png")


def plot_coefficients(pipeline, n_bins, out, top=15):
    names = [n.split("__", 1)[-1] for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
    coef = pd.DataFrame(pipeline.named_steps["model"].coef_.T, index=names,
                        columns=[f"khoảng {i}" for i in range(n_bins)])
    coef.to_csv(out / f"lr_coefficients_{n_bins}bins.csv")
    # Hướng "khó hơn": hệ số khoảng cao nhất trừ khoảng thấp nhất (đặc trưng đã chuẩn hóa).
    contrast = (coef.iloc[:, -1] - coef.iloc[:, 0]).sort_values()
    contrast = pd.concat([contrast.head(top // 2), contrast.tail(top - top // 2)])
    figure, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(contrast.index, contrast.values,
            color=[SERIES[1] if v > 0 else SERIES[0] for v in contrast.values])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set(xlabel=f"Hệ số khoảng {n_bins - 1} trừ khoảng 0 (đặc trưng chuẩn hóa)",
           title="Logistic: đặc trưng đẩy về bài khó (cam) và bài dễ (xanh)")
    save(figure, out / f"lr_coefficients_{n_bins}bins.png")
    return contrast


def run_logistic(frame, features, out, seed=42, folds=5, n_bins_list=(3, 4)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    X, y = features, frame["Rating"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    rows, cv_tables, details = [], [], {}

    for fraction in FRACTIONS:
        train, valid = split_groups(frame, fraction, seed)
        groups = frame["game_id"].iloc[train]
        for n_bins in n_bins_list:
            labels = list(range(n_bins))
            thresholds, y_train = fit_rating_bins(y.iloc[train], n_bins)
            y_valid = apply_rating_bins(y.iloc[valid], thresholds)
            searches = {}
            for representation in REPRESENTATIONS:
                search = tune_logistic(numeric, X.iloc[train], y_train, groups, representation, seed, folds)
                searches[representation] = search
                best_c = float(search.best_params_[C_PARAMETER])
                table = pd.DataFrame(search.cv_results_)[
                    [f"param_{C_PARAMETER}", "mean_train_score", "mean_test_score", "std_test_score"]]
                cv_tables.append(table.rename(columns={f"param_{C_PARAMETER}": "C"}).assign(
                    validation_fraction=fraction, n_bins=n_bins, representation=representation))
                majority = DummyClassifier(strategy="most_frequent").fit(X.iloc[train], y_train)
                for model_name, model in (("logistic_regression", search.best_estimator_),
                                          ("majority_baseline", majority)):
                    for split, idx, target in (("train", train, y_train), ("validation", valid, y_valid)):
                        metrics = evaluate_classification(target, model.predict(X.iloc[idx]), labels)
                        rows.append(dict(model=model_name, representation=representation, n_bins=n_bins,
                                         validation_fraction=fraction, split=split, n=len(idx),
                                         C=best_c if model_name == "logistic_regression" else None,
                                         cv_macro_f1=(float(search.best_score_)
                                                      if model_name == "logistic_regression" else None),
                                         accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"]))
                print(f"logistic val={fraction} {n_bins} khoảng {representation}: C={best_c:g} "
                      f"cv_f1={search.best_score_:.4f} val_f1={rows[-3]['macro_f1']:.4f}", flush=True)
                if fraction == 0.2:
                    metrics = evaluate_classification(y_valid, search.predict(X.iloc[valid]), labels)
                    _plot_confusion_matrix(np.array(metrics["confusion_matrix"]), n_bins,
                                           f"Logistic - {representation} - {n_bins} khoảng (val 20%)",
                                           out / f"lr_confusion_{representation}_{n_bins}bins.png")
                    details[f"{representation}_{n_bins}bins"] = dict(
                        C=best_c, thresholds=thresholds,
                        n_components=reduced_dimensions(numeric) if representation == "pca" else None,
                        confusion_matrix_validation=metrics["confusion_matrix"])
            if fraction == 0.2:
                plot_tuning(searches, n_bins, out)
                plot_coefficients(searches["original"].best_estimator_, n_bins, out)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "lr_metrics.csv", index=False)
    pd.concat(cv_tables, ignore_index=True).to_csv(out / "lr_cv_results.csv", index=False)
    summary = metrics.pivot_table(index=["validation_fraction", "n_bins", "representation", "model"],
                                  columns="split", values=["accuracy", "macro_f1"])
    summary.columns = [f"{metric}_{split}" for metric, split in summary.columns]
    summary = summary.join(metrics[metrics.split == "validation"].set_index(
        ["validation_fraction", "n_bins", "representation", "model"])[["C", "cv_macro_f1"]])
    summary.round(4).to_csv(out / "lr_summary.csv")
    result = dict(seed=seed, folds=folds, scoring="f1_macro", C_grid=C_GRID.tolist(),
                  details_validation_0_2=details,
                  validation=summary.reset_index().round(4).to_dict("records"))
    (out / "lr_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression trên nhãn khoảng Rating.")
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--cache", default="outputs/features_v2/sample.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="outputs/logistic")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    frame = load_frame(args.data, args.cache, args.sample, args.seed)
    run_logistic(frame, make_features(frame), args.output, args.seed, args.folds)
    print(pd.read_csv(Path(args.output) / "lr_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
