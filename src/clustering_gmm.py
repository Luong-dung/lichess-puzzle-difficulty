"""Member 3: Phân cụm thế cờ bằng Gaussian Mixture Models (GMM)."""
import argparse
import json
import sys
from pathlib import Path
import matplotlib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from .classification import apply_rating_bins, fit_rating_bins
from .data import load_sample, split_groups
from .features import make_features
from .preprocessing import build_preprocess
from .plot_style import MUTED, SERIES, TEXT, plt, save

#  Tuning GMM
def tune_gmm(
    X_scaled,
    k_range=range(2, 9),
    covariance_types=("full", "tied", "diag", "spherical"),
    seed=42,
):
    
    X = np.asarray(X_scaled)
    rows = []
    for cov_type in covariance_types:
        for k in k_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                random_state=seed,
                max_iter=200,
                n_init=3,
            )
            gmm.fit(X)
            rows.append(
                dict(
                    k=k,
                    covariance_type=cov_type,
                    bic=float(gmm.bic(X)),
                    aic=float(gmm.aic(X)),
                    n_iter=int(gmm.n_iter_),
                )
            )
            print(
                f"gmm cov={cov_type} k={k}  BIC={gmm.bic(X):.1f}  AIC={gmm.aic(X):.1f}",
                flush=True,
            )

    # Tìm cấu hình có BIC thấp nhất
    bic_table = pd.DataFrame(rows)
    best_idx = bic_table["bic"].idxmin()
    best_k = int(bic_table.loc[best_idx, "k"])
    best_cov = str(bic_table.loc[best_idx, "covariance_type"])
    bic_table.attrs["best_k"] = best_k
    bic_table.attrs["best_covariance_type"] = best_cov

    # Fit lại mô hình tối ưu với n_init lớn hơn để ổn định
    best_model = GaussianMixture(
        n_components=best_k,
        covariance_type=best_cov,
        random_state=seed,
        max_iter=300,
        n_init=5,
    ).fit(X)

    return bic_table, best_model

# Biểu đồ trực quan hóa chọn mô hình
def plot_gmm_selection(bic_table, out):
    out = Path(out)
    cov_types = bic_table["covariance_type"].unique()
    best_k = bic_table.attrs.get("best_k")
    best_cov = bic_table.attrs.get("best_covariance_type")

    n_cov = len(cov_types)
    figure, axes = plt.subplots(1, n_cov, figsize=(4 * n_cov, 3.8), sharey=False)
    if n_cov == 1:
        axes = [axes]

    for ax, cov_type in zip(axes, cov_types):
        sub = bic_table[bic_table["covariance_type"] == cov_type].sort_values("k")
        ax.plot(sub["k"], sub["bic"], color=SERIES[0], marker="o", markersize=5, label="BIC")
        ax.plot(sub["k"], sub["aic"], color=SERIES[1], marker="s", markersize=5,
                linestyle="--", label="AIC")
        if best_cov == cov_type and best_k is not None:
            best_bic = float(sub.loc[sub["k"] == best_k, "bic"].iloc[0])
            ax.axvline(best_k, color=MUTED, linewidth=0.9, linestyle=":")
            ax.annotate(
                f"k={best_k} (BIC tối ưu)",
                xy=(best_k, best_bic),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=TEXT,
            )
        ax.set(
            xlabel="Số thành phần k",
            ylabel="Tiêu chí thông tin",
            title=f"GMM - covariance={cov_type}",
        )
        ax.legend(loc="upper left")

    save(figure, out / "gmm_bic_aic.png")


# Biểu đồ phân bố các cụm và phân bố Rating
def plot_gmm_clusters(pca_2d, cluster_labels, rating, out):
    out = Path(out)
    pca_2d = np.asarray(pca_2d, float)
    cluster_labels = np.asarray(cluster_labels, int)
    rating = np.asarray(rating, float)
    k = int(cluster_labels.max()) + 1

    _get_cmap = getattr(matplotlib, "colormaps", None)
    if _get_cmap is not None:
        cmap = matplotlib.colormaps["tab10"]
    else:
        cmap = plt.cm.get_cmap("tab10") 

    figure, ax = plt.subplots(figsize=(7, 5))
    for c in range(k):
        mask = cluster_labels == c
        ax.scatter(
            pca_2d[mask, 0],
            pca_2d[mask, 1],
            s=12,
            alpha=0.55,
            color=cmap(c),
            label=f"Cụm {c}",
            linewidths=0,
        )
    ax.set(
        xlabel="PC1",
        ylabel="PC2",
        title=f"GMM - {k} cụm trên không gian PC1-PC2",
    )
    ax.legend(loc="upper right", markerscale=2)
    save(figure, out / "gmm_clusters_pc.png")

    # -- Boxplot Rating theo cụm --
    figure, ax = plt.subplots(figsize=(max(5, k * 1.4), 4))
    data_by_cluster = [rating[cluster_labels == c] for c in range(k)]
    bp = ax.boxplot(
        data_by_cluster,
        patch_artist=True,
        medianprops=dict(color=TEXT, linewidth=1.5),
        whiskerprops=dict(color=MUTED),
        capprops=dict(color=MUTED),
        flierprops=dict(marker=".", color=MUTED, markersize=3, alpha=0.4),
    )
    for patch, c in zip(bp["boxes"], range(k)):
        patch.set_facecolor(cmap(c))
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, k + 1))
    ax.set_xticklabels([f"Cụm {c}\n(n={int((cluster_labels == c).sum())})" for c in range(k)],
                       fontsize=8)
    ax.set(
        xlabel="Cụm GMM",
        ylabel="Rating",
        title=f"Phân bố Rating theo {k} cụm GMM",
    )
    save(figure, out / "gmm_rating_by_cluster.png")

# So sánh GMM với K-Means 
def _safe_silhouette(X, labels, seed=42, max_n=5000):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False) if len(X) > max_n else np.arange(len(X))
    if len(np.unique(labels[idx])) < 2:
        return float("nan")
    return float(silhouette_score(X[idx], labels[idx]))

def rating_eta2(rating, labels):
    rating = pd.Series(np.asarray(rating, float))
    means = rating.groupby(np.asarray(labels)).transform("mean")
    total = ((rating - rating.mean()) ** 2).sum()
    return float(((means - rating.mean()) ** 2).sum() / total) if total > 0 else float("nan")


def compare_with_kmeans(X_scaled, gmm_labels, kmeans_labels, true_rating_bins, seed=42):
    X = np.asarray(X_scaled, float)
    gmm_labels = np.asarray(gmm_labels, int)
    kmeans_labels = np.asarray(kmeans_labels, int)
    true_rating_bins = np.asarray(true_rating_bins, int)

    rows = [
        dict(metric="Silhouette Score",
             gmm=_safe_silhouette(X, gmm_labels, seed), kmeans=_safe_silhouette(X, kmeans_labels, seed)),
        dict(metric="ARI (vs Rating bins)",
             gmm=float(adjusted_rand_score(true_rating_bins, gmm_labels)),
             kmeans=float(adjusted_rand_score(true_rating_bins, kmeans_labels))),
        dict(metric="NMI (vs Rating bins)",
             gmm=float(normalized_mutual_info_score(true_rating_bins, gmm_labels)),
             kmeans=float(normalized_mutual_info_score(true_rating_bins, kmeans_labels))),
    ]
    return pd.DataFrame(rows)


def load_kmeans_labels(path, frame):
    path = Path(path) if path else None
    if path is None or not path.exists():
        return None
    kdf = pd.read_csv(path, dtype={"PuzzleId": "string"})
    aligned = kdf.drop_duplicates("PuzzleId").set_index("PuzzleId")["kmeans_cluster"].reindex(
        frame["PuzzleId"].astype("string"))
    if aligned.isna().any():
        print(f"{path} does not match current sample ({int(aligned.isna().sum())} missing PuzzleId, skipping K-Means comparison.", flush=True)
        return None
    return aligned.to_numpy(int)


def run_gmm(frame, features, out, seed=42, n_bins=3, k_range=range(2, 9), covariance_types=("full", "tied", "diag", "spherical"),
    kmeans_labels=None, kmeans_labels_path=None, validation_fraction=0.2,):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    X = features
    y = frame["Rating"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    
    # Chia dữ liệu và chuẩn hóa
    train_idx, valid_idx = split_groups(frame, validation_fraction=validation_fraction, seed=seed)
    preprocessor = build_preprocess(numeric).fit(X.iloc[train_idx])
    X_scaled = preprocessor.transform(X)
    Z_train, Z_valid = X_scaled[train_idx], X_scaled[valid_idx]

    # Tìm mô hình tối ưu
    bic_table, best_model = tune_gmm(Z_train, k_range, covariance_types, seed)
    best_k = bic_table.attrs["best_k"]
    best_cov = bic_table.attrs["best_covariance_type"]
    bic_table.to_csv(out / "gmm_bic_aic.csv", index=False)
    ks = list(k_range)
    at_boundary = best_k in (min(ks), max(ks)) and len(ks) > 1
    if at_boundary:
        print(f"best_k={best_k} is at the boundary of {min(ks)}-{max(ks)}.", flush=True)

    # Dự đoán nhãn cụm 
    cluster_labels = best_model.predict(X_scaled)

    plot_gmm_selection(bic_table, out)

    pca2 = PCA(n_components=2, random_state=seed).fit(Z_train)   # PCA fit trên train
    pca_2d = pca2.transform(X_scaled)
    plot_gmm_clusters(pca_2d[train_idx], cluster_labels[train_idx], y.iloc[train_idx].to_numpy(), out)

    # Chia khoảng Rating và tính chỉ số Silhouette
    thresholds, _ = fit_rating_bins(y.iloc[train_idx], n_bins)
    rating_bins_all = apply_rating_bins(y, thresholds).to_numpy()

    sil_train = _safe_silhouette(Z_train, cluster_labels[train_idx], seed)
    sil_valid = _safe_silhouette(Z_valid, cluster_labels[valid_idx], seed)

    split_col = np.full(len(frame), "train", dtype=object)
    split_col[valid_idx] = "validation"
    pd.DataFrame({"PuzzleId": frame["PuzzleId"].to_numpy(), "split": split_col,
                  "gmm_cluster": cluster_labels, "rating": y.to_numpy(),
                  "rating_bin": rating_bins_all}).to_csv(out / "gmm_labels.csv", index=False)

    # So sánh kết quả với K-Means
    if kmeans_labels is None:
        kmeans_labels = load_kmeans_labels(kmeans_labels_path, frame)
    compare_df = None
    eta2 = dict(gmm=rating_eta2(y.iloc[valid_idx], cluster_labels[valid_idx]))
    if kmeans_labels is not None:
        kmeans_arr = np.asarray(kmeans_labels, int)
        parts = []
        for split, idx in (("validation", valid_idx), ("train", train_idx)):
            part = compare_with_kmeans(X_scaled[idx], cluster_labels[idx], kmeans_arr[idx], rating_bins_all[idx], seed)
            if split == "validation":
                eta2["kmeans"] = rating_eta2(y.iloc[idx], kmeans_arr[idx])
                part = pd.concat([part, pd.DataFrame([dict(metric="eta² Rating", gmm=eta2["gmm"], kmeans=eta2["kmeans"])])], ignore_index=True)
            part.insert(0, "split", split)
            parts.append(part)
        compare_df = pd.concat(parts, ignore_index=True)
        compare_df.to_csv(out / "gmm_vs_kmeans.csv", index=False)
        print(compare_df.to_string(index=False), flush=True)

    summary = dict(
        seed=seed,
        best_k=best_k,
        best_covariance_type=best_cov,
        best_k_at_boundary=bool(at_boundary),
        n_bins_for_ari=n_bins,
        fit_on="train",
        silhouette=None if np.isnan(sil_train) else round(sil_train, 4),
        silhouette_validation=None if np.isnan(sil_valid) else round(sil_valid, 4),
        ari_vs_rating_validation=round(float(adjusted_rand_score(rating_bins_all[valid_idx], cluster_labels[valid_idx])), 4),
        nmi_vs_rating_validation=round(float(normalized_mutual_info_score(rating_bins_all[valid_idx], cluster_labels[valid_idx])), 4),
        eta2_rating_validation=round(eta2["gmm"], 4),
        compare_with_kmeans=(compare_df.round(4).to_dict("records") if compare_df is not None else None),
    )
    (out / "gmm_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"GMM best: k={best_k}, cov={best_cov}, silhouette(train)={sil_train:.4f}", flush=True)
    return summary

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Phân cụm thế cờ bằng GMM."
    )
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=3,
                        help="Số khoảng Rating khi tính ARI/NMI.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--output", default="outputs/clustering")
    parser.add_argument("--kmeans-labels", default="outputs/clustering/kmeans_labels.csv")
    args = parser.parse_args()

    frame = load_sample(args.data, args.sample, args.seed)
    summary = run_gmm(
        frame,
        make_features(frame),
        Path(args.output),
        seed=args.seed,
        n_bins=args.n_bins,
        k_range=range(args.k_min, args.k_max + 1),
        kmeans_labels_path=args.kmeans_labels,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()