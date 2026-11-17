"""Chia khoảng điểm Rating theo phân vị và phân loại bằng Gaussian Naive Bayes.
Cung cấp các hàm tiện ích xử lý nhãn dùng chung cho cả Naive Bayes (module này) và Logistic Regression
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

from .data import load_sample, split_groups
from .features import make_features
from .preprocessing import build_preprocess, reduced_dimensions
from .plot_style import plt, save, SEQUENTIAL, TEXT

# Tính threshold từ tập train và gán nhãn cho y_train
def fit_rating_bins(y_train, n_bins=3):
    if n_bins < 2:
        raise ValueError(f"n_bins phải >= 2")
    # Tính các mốc phân vị
    quantile_steps = [i / n_bins for i in range(1, n_bins)]
    thresholds = y_train.quantile(quantile_steps).tolist()
    # Đảm bảo các ngưỡng không bị trùng nhau
    if len(set(thresholds)) < len(thresholds):
        raise ValueError(
            f"Các ngưỡng thresholds không duy nhất ({thresholds})"
        )
    y_train_binned = apply_rating_bins(y_train, thresholds)
    return thresholds, y_train_binned

# Ánh xạ điểm Rating sang nhãn số nguyên dựa vào các ngưỡng đã học
def apply_rating_bins(y, thresholds):
    # Cắt dữ liệu thành các khoảng
    binned = pd.cut(y, bins=[-np.inf] + list(thresholds) + [np.inf], labels=False, right=True)
    if binned.isna().any():
        raise ValueError("Giá trị Rating bị thiếu")
    return binned.astype("int64")



def evaluate_classification(y_true, y_pred, labels=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

# Vẽ biểu đồ cho ma trận nhầm lẫn
def _plot_confusion_matrix(cm, n_bins, title, path):
    figure, ax = plt.subplots(figsize=(4.5, 3.8))
    image = ax.imshow(cm, cmap="Blues", aspect="auto")
    figure.colorbar(image, ax=ax, label="Số bài")
    ax.grid(False)
    ticks = range(n_bins)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"Dự đoán {i}" for i in ticks], rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels([f"Thực tế {i}" for i in ticks], fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(n_bins):
        for j in range(n_bins):
            text_color = "white" if cm[i, j] > thresh else "#1f2937"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=text_color, fontsize=9, fontweight="medium")
    ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thực tế", title=title)
    save(figure, path)

# Xây dựng pipeline cho Gaussian Naive Bayes
def _build_nb_pipeline(numeric, representation="original", seed=42):
    steps = [("preprocess", build_preprocess(numeric))]
    if representation == "pca":
        n_components = reduced_dimensions(numeric)
        steps.append(("reduce", PCA(n_components=n_components, random_state=seed)))
    steps.append(("model", GaussianNB()))
    return Pipeline(steps)

# Huấn luyện và đánh giá Gaussian Naive Bayes
def run_classification_nb(frame, features, out, seed=42, n_bins_list=(3, 4)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    X = features
    y = frame["Rating"]
    numeric = X.select_dtypes(include="number").columns.tolist()

    rows = []
    best_configs = {}

    for fraction in (0.2, 0.3, 0.4):
        train_idx, valid_idx = split_groups(frame, fraction, seed)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        for n_bins in n_bins_list:
            thresholds, y_train_binned = fit_rating_bins(y_train, n_bins)
            y_valid_binned = apply_rating_bins(y_valid, thresholds)

            for representation in ("original", "pca"):
                pipeline = _build_nb_pipeline(numeric, representation, seed)
                pipeline.fit(X_train, y_train_binned)

                for split, X_s, y_s in (
                    ("train", X_train, y_train_binned),
                    ("validation", X_valid, y_valid_binned),
                ):
                    y_pred = pipeline.predict(X_s)
                    metrics = evaluate_classification(y_s, y_pred, labels=range(n_bins))
                    rows.append(
                        dict(
                            model="naive_bayes",
                            representation=representation,
                            n_bins=n_bins,
                            validation_fraction=fraction,
                            split=split,
                            n=len(y_s),
                            accuracy=metrics["accuracy"],
                            macro_f1=metrics["macro_f1"],
                        )
                    )

                if fraction == 0.2:
                    cfg_key = f"nb_{representation}_{n_bins}bins"
                    y_pred_valid = pipeline.predict(X_valid)
                    val_metrics = evaluate_classification(y_valid_binned, y_pred_valid, labels=range(n_bins))
                    best_configs[cfg_key] = {
                        "representation": representation,
                        "n_bins": n_bins,
                        "n_components": (reduced_dimensions(numeric) if representation == "pca" else None),
                        "validation_fraction": fraction,
                        "thresholds": thresholds,
                        "validation_accuracy": val_metrics["accuracy"],
                        "validation_macro_f1": val_metrics["macro_f1"],
                    }
                    cm = np.array(val_metrics["confusion_matrix"])
                    _plot_confusion_matrix(
                        cm,
                        n_bins,
                        title=f"Naive Bayes - {representation} - {n_bins} khoảng (val frac=0.2)",
                        path=out / f"nb_confusion_{representation}_{n_bins}bins.png",
                    )
                    print(
                        f"classification nb {representation} {n_bins}bins frac={fraction}"
                        f"  acc={val_metrics['accuracy']:.4f}"
                        f"  f1={val_metrics['macro_f1']:.4f}",
                        flush=True,
                    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out / "nb_metrics.csv", index=False)

    summary = dict(
        seed=seed,
        n_bins_evaluated=list(n_bins_list),
        best_configs=best_configs,
        validation=metrics_df[metrics_df["split"] == "validation"]
        .round(4)
        .to_dict("records"),
    )
    (out / "nb_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary



def main():
    parser = argparse.ArgumentParser(
        description="Phân loại độ khó bài cờ vua bằng Naive Bayes."
    )
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/classification")
    args = parser.parse_args()

    frame = load_sample(args.data, args.sample, args.seed)
    summary = run_classification_nb(
        frame, make_features(frame), Path(args.output), seed=args.seed
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
