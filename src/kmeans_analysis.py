"""Thành viên 2: K-Means trên đặc trưng chuẩn hóa (fit trên train), đối chiếu Rating.

Chạy:  python -m src.kmeans_analysis            (tự chọn k theo silhouette)
       python -m src.kmeans_analysis --k 4      (chọn k cố định)
"""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .data import split_groups
from .features import make_features
from .sample_cache import load_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--cache", default="outputs/features_v2/sample.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation", type=float, default=0.2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--silhouette-sample", type=int, default=5000)
    parser.add_argument("--output", default="outputs/clustering_v3")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args.data, args.cache, args.sample, args.seed)
    X = make_features(frame)
    X = pd.get_dummies(X, columns=["turn"], dtype=float)
    train, valid = split_groups(frame, args.validation, args.seed)
    X = X.loc[:, X.iloc[train].nunique() > 1]  # cột hằng xác định trên train
    scaler = StandardScaler().fit(X.iloc[train])
    Z_train, Z_valid = scaler.transform(X.iloc[train]), scaler.transform(X.iloc[valid])
    rating_train, rating_valid = frame["Rating"].iloc[train], frame["Rating"].iloc[valid]
    print(f"{X.shape[1]} đặc trưng; train {len(train)}, validation {len(valid)}", flush=True)

    rows = []
    for k in range(2, args.k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(Z_train)
        sil = silhouette_score(Z_train, km.labels_, sample_size=min(args.silhouette_sample, len(train)),
                               random_state=args.seed)
        rows.append(dict(k=k, inertia=km.inertia_, silhouette=sil))
        print(f"k={k} inertia={km.inertia_:.0f} silhouette={sil:.3f}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "kmeans_scores.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(scores.k, scores.inertia, "o-")
    axes[0].set(xlabel="k", ylabel="Inertia", title="Elbow")
    axes[1].plot(scores.k, scores.silhouette, "o-")
    axes[1].set(xlabel="k", ylabel="Silhouette", title="Silhouette")
    plt.tight_layout()
    plt.savefig(out / "kmeans_elbow_silhouette.png", dpi=150)
    plt.close()

    k = args.k or int(scores.loc[scores.silhouette.idxmax(), "k"])
    km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(Z_train)
    label_train, label_valid = km.labels_, km.predict(Z_valid)

    def rating_by_cluster(labels, rating):
        return rating.groupby(labels).agg(["count", "mean", "std", "median"]).rename_axis("cluster")
    stats = pd.concat({"train": rating_by_cluster(label_train, rating_train),
                       "validation": rating_by_cluster(label_valid, rating_valid)}, names=["split"])
    stats.to_csv(out / f"kmeans_k{k}_rating_by_cluster.csv")
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=X.columns).rename_axis("cluster")
    centers.to_csv(out / f"kmeans_k{k}_centers.csv")
    # Mức độ cụm giải thích Rating: tỉ lệ phương sai Rating giữa các cụm (eta²) trên validation.
    means = rating_valid.groupby(label_valid).transform("mean")
    eta2 = float(((means - rating_valid.mean()) ** 2).sum() / ((rating_valid - rating_valid.mean()) ** 2).sum())

    pca = PCA(n_components=2, random_state=args.seed).fit(Z_train)
    P = pca.transform(Z_train)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(P[:, 0], P[:, 1], c=label_train, cmap="tab10", s=3, alpha=0.4)
    axes[0].legend(*sc.legend_elements(), title="Cụm", fontsize=8)
    axes[0].set(xlabel="PC1", ylabel="PC2", title=f"K-Means k={k} trên PC1–PC2 (train)")
    sc = axes[1].scatter(P[:, 0], P[:, 1], c=rating_train, cmap="viridis", s=3, alpha=0.4)
    plt.colorbar(sc, ax=axes[1], label="Rating")
    axes[1].set(xlabel="PC1", ylabel="PC2", title="Rating trên PC1–PC2 (train)")
    plt.tight_layout()
    plt.savefig(out / f"kmeans_k{k}_pca.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.boxplot([rating_valid[label_valid == c] for c in range(k)], showfliers=False)
    plt.xticks(range(1, k + 1), range(k))
    plt.xlabel("Cụm")
    plt.ylabel("Rating")
    plt.title(f"Rating theo cụm (validation), eta² = {eta2:.3f}")
    plt.tight_layout()
    plt.savefig(out / f"kmeans_k{k}_rating_boxplot.png", dpi=150)
    plt.close()

    summary = dict(k=k, chosen_by="--k" if args.k else "max silhouette",
                   silhouette=float(scores.set_index("k").loc[k, "silhouette"]),
                   eta2_rating_validation=eta2, n_features=X.shape[1],
                   pca_explained_variance=pca.explained_variance_ratio_.tolist(), seed=args.seed)
    (out / f"kmeans_k{k}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(stats)


if __name__ == "__main__":
    main()
