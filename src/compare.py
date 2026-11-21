"""Gom kết quả của ba thành viên trong outputs/ thành bảng so sánh cuối cùng.

Không huấn luyện lại; chỉ đọc file đã chạy. Thứ tự chạy trước đó (mặc định seed 42, mẫu 20.000):
    python -m src.linear_analysis
    python -m src.tune_random_forest && python -m src.rf_reduction
    python -m src.mlp_analysis
    python -m src.classification && python -m src.logistic_analysis
    python -m src.kmeans_analysis && python -m src.clustering_gmm
    python -m src.compare                     -> outputs/comparison/

Nguồn nào thiếu thì bỏ qua và in cảnh báo, không dừng chương trình.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .plot_style import SERIES, plt, save

REGRESSION_COLUMNS = ["validation_fraction", "representation", "model", "rmse_train", "rmse_validation",
                      "mae_validation", "r2_validation", "rmse_gap"]
CLASSIFICATION_COLUMNS = ["validation_fraction", "n_bins", "representation", "model", "accuracy_train",
                          "accuracy_validation", "macro_f1_train", "macro_f1_validation"]
MODEL_ORDER = ["mean_baseline", "ols", "ridge_tuned", "random_forest_tuned", "mlp_tuned",
               "majority_baseline", "naive_bayes", "logistic_regression"]


def _warn(message):
    print(f"Bỏ qua: {message}", flush=True)


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        _warn(f"không có {path}")
        return None
    return pd.read_csv(path)


def _wide(long, index):
    """Bảng dài (một dòng mỗi split) -> một dòng mỗi cấu hình, cột metric_split."""
    metrics = [m for m in ("mae", "rmse", "r2", "accuracy", "macro_f1") if m in long]
    wide = long.pivot_table(index=index, columns="split", values=metrics)
    wide.columns = [f"{metric}_{split}" for metric, split in wide.columns]
    return wide.reset_index()


def regression_table(linear_dir, rf_dir, mlp_dir):
    parts = []
    linear = _read_csv(Path(linear_dir) / "linear_summary.csv")
    if linear is not None:
        parts.append(linear.assign(source="Thành viên 1"))
    rf = _read_csv(Path(rf_dir) / "rf_by_representation.csv")
    if rf is not None:
        rf = _wide(rf, ["validation_fraction", "representation"]).assign(
            model="random_forest_tuned", source="Thành viên 2")
        rf["rmse_gap"] = rf["rmse_validation"] - rf["rmse_train"]
        parts.append(rf)
    mlp = _read_csv(Path(mlp_dir) / "mlp_summary.csv")
    if mlp is not None:
        # Baseline và OLS đã có ở bảng của Thành viên 1 với cùng phép chia.
        keep = mlp["model"] == "mlp_tuned" if linear is not None else slice(None)
        parts.append(mlp[keep].assign(source="Thành viên 3"))
    if not parts:
        return None
    table = pd.concat(parts, ignore_index=True)[REGRESSION_COLUMNS + ["source"]]
    return _sort(table, ["validation_fraction", "representation"])


def classification_table(nb_dir, logistic_dir):
    parts = []
    nb = _read_csv(Path(nb_dir) / "nb_metrics.csv")
    if nb is not None:
        parts.append(_wide(nb, ["validation_fraction", "n_bins", "representation", "model"])
                     .assign(source="Thành viên 3"))
    lr = _read_csv(Path(logistic_dir) / "lr_metrics.csv")
    if lr is not None:
        parts.append(_wide(lr, ["validation_fraction", "n_bins", "representation", "model"])
                     .assign(source="Thành viên 1"))
    if not parts:
        return None
    table = pd.concat(parts, ignore_index=True)[CLASSIFICATION_COLUMNS + ["source"]]
    return _sort(table, ["validation_fraction", "n_bins", "representation"])


def _sort(table, keys):
    order = {name: i for i, name in enumerate(MODEL_ORDER)}
    table = table.assign(_order=table["model"].map(order).fillna(len(order)))
    return table.sort_values(keys + ["_order"]).drop(columns="_order").reset_index(drop=True)


def clustering_table(kmeans_dir, gmm_dir):
    rows = []
    summaries = sorted(Path(kmeans_dir).glob("kmeans_k*_summary.json"), key=lambda p: p.stat().st_mtime)
    if summaries:
        km = json.loads(summaries[-1].read_text(encoding="utf-8"))
        rows.append(dict(method="K-Means", k=km["k"], chosen_by=km.get("chosen_by"),
                         silhouette_train=km.get("silhouette"), silhouette_validation=None,
                         ari_validation=None, nmi_validation=None,
                         eta2_rating_validation=km.get("eta2_rating_validation"), source="Thành viên 2"))
    else:
        _warn(f"không có kmeans_k*_summary.json trong {kmeans_dir}")
    gmm_path = Path(gmm_dir) / "gmm_summary.json"
    if gmm_path.exists():
        gmm = json.loads(gmm_path.read_text(encoding="utf-8"))
        rows.append(dict(method=f"GMM ({gmm['best_covariance_type']})", k=gmm["best_k"], chosen_by="min BIC",
                         silhouette_train=gmm.get("silhouette"),
                         silhouette_validation=gmm.get("silhouette_validation"),
                         ari_validation=gmm.get("ari_vs_rating_validation"),
                         nmi_validation=gmm.get("nmi_vs_rating_validation"),
                         eta2_rating_validation=gmm.get("eta2_rating_validation"), source="Thành viên 3"))
    else:
        _warn(f"không có {gmm_path}")
    return pd.DataFrame(rows) if rows else None


def _bar_chart(table, value, group, title, xlabel, path):
    pivot = table.pivot_table(index="model", columns=group, values=value)
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    figure, ax = plt.subplots(figsize=(7.5, 0.55 * len(pivot) * max(1, len(pivot.columns)) + 1.2))
    height = 0.8 / len(pivot.columns)
    positions = np.arange(len(pivot))
    for i, (column, color) in enumerate(zip(pivot.columns, SERIES * 3)):
        values = pivot[column].to_numpy()
        bars = ax.barh(positions + (i - (len(pivot.columns) - 1) / 2) * height, values, height=height * 0.92,
                       color=color, label=str(column))
        ax.bar_label(bars, fmt="%.3g", fontsize=7, padding=2)
    ax.set_yticks(positions, pivot.index)
    ax.invert_yaxis()
    ax.set(xlabel=xlabel, title=title)
    ax.legend(loc="lower right", title=group)
    save(figure, path)


def _markdown(table, digits=4):
    shown = table.copy()
    for column in shown.select_dtypes("number"):
        shown[column] = shown[column].round(digits)
    header = "| " + " | ".join(shown.columns) + " |"
    divider = "|" + "|".join("---" for _ in shown.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def build_comparison(out, linear_dir="outputs/linear", rf_dir="outputs/rf_reduction_v2", mlp_dir="outputs/mlp",
                     nb_dir="outputs/classification", logistic_dir="outputs/logistic",
                     kmeans_dir="outputs/clustering_v3", gmm_dir="outputs/clustering", fraction=0.2):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = dict(regression=regression_table(linear_dir, rf_dir, mlp_dir),
                  classification=classification_table(nb_dir, logistic_dir),
                  clustering=clustering_table(kmeans_dir, gmm_dir))
    lines = ["# Bảng so sánh cuối cùng", "",
             f"Sinh tự động bởi `python -m src.compare`. Bảng rút gọn dùng tỷ lệ validation {fraction:.0%}; "
             "bảng đầy đủ ở các file CSV cùng thư mục.", ""]
    regression = tables["regression"]
    if regression is not None:
        regression.to_csv(out / "regression.csv", index=False)
        short = regression[regression.validation_fraction == fraction]
        _bar_chart(short, "rmse_validation", "representation",
                   f"Hồi quy Rating: RMSE validation ({fraction:.0%})", "RMSE (Rating), thấp hơn là tốt hơn",
                   out / "regression_rmse.png")
        lines += ["## Hồi quy Rating", "",
                  "SVD của Random Forest là SVD chưa tâm hóa; SVD của Ridge và MLP là TruncatedSVD "
                  "sau chuẩn hóa có tâm (gần như trùng PCA).", "",
                  _markdown(short.drop(columns="validation_fraction"), 3), ""]
    classification = tables["classification"]
    if classification is not None:
        classification.to_csv(out / "classification.csv", index=False)
        short = classification[classification.validation_fraction == fraction]
        for n_bins, part in short.groupby("n_bins"):
            _bar_chart(part, "macro_f1_validation", "representation",
                       f"Phân loại {n_bins} khoảng Rating: macro-F1 validation ({fraction:.0%})",
                       "Macro-F1", out / f"classification_f1_{n_bins}bins.png")
        lines += ["## Phân loại khoảng Rating", "",
                  "Ngưỡng phân vị học trên train; đoán ngẫu nhiên cho accuracy 1/số khoảng.", "",
                  _markdown(short.drop(columns="validation_fraction")), ""]
    clustering = tables["clustering"]
    if clustering is not None:
        clustering.to_csv(out / "clustering.csv", index=False)
        lines += ["## Phân cụm", "", "eta² = tỷ lệ phương sai Rating giải thích bởi nhãn cụm (validation).", "",
                  _markdown(clustering), ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return tables


def main():
    parser = argparse.ArgumentParser(description="Gom outputs/ thành bảng so sánh cuối cùng.")
    parser.add_argument("--output", default="outputs/comparison")
    parser.add_argument("--fraction", type=float, default=0.2, help="Tỷ lệ validation cho bảng rút gọn.")
    for name, default in (("linear", "outputs/linear"), ("rf", "outputs/rf_reduction_v2"), ("mlp", "outputs/mlp"),
                          ("nb", "outputs/classification"), ("logistic", "outputs/logistic"),
                          ("kmeans", "outputs/clustering_v3"), ("gmm", "outputs/clustering")):
        parser.add_argument(f"--{name}-dir", default=default)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    build_comparison(args.output, args.linear_dir, args.rf_dir, args.mlp_dir, args.nb_dir, args.logistic_dir,
                     args.kmeans_dir, args.gmm_dir, args.fraction)
    print((Path(args.output) / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
