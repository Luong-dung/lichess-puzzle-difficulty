"""Member 1: exploratory analysis and PCA; PCA is fitted on the training split only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from .data import load_sample, split_groups
from .features import make_features
from .preprocessing import build_preprocess, clean_names, reduced_dimensions
from .plot_style import plt, save, SERIES, SEQUENTIAL, DIVERGING, MUTED, TEXT


def plot_rating(frame, features, out):
    figure, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(frame["Rating"].min(), frame["Rating"].max(), 41)
    for color, (turn, label) in zip(SERIES, (("w", "Trắng đi"), ("b", "Đen đi"))):
        ax.hist(frame.loc[features["turn"] == turn, "Rating"], bins=bins, histtype="step",
                linewidth=2, color=color, label=f"{label} (n={(features['turn'] == turn).sum()})")
    for q in (0.25, 0.5, 0.75):
        value = frame["Rating"].quantile(q)
        ax.axvline(value, color=MUTED, linewidth=0.8, linestyle="--")
        ax.annotate(f"Q{int(q*4)}={value:.0f}" if q != 0.5 else f"Trung vị={value:.0f}",
                    (value, 1), xycoords=("data", "axes fraction"), fontsize=7, color=MUTED,
                    ha="left", va="top", xytext=(3, 0), textcoords="offset points")
    ax.set(xlabel="Rating", ylabel="Số bài", title=f"Phân bố Rating theo bên tới lượt (n={len(frame)})")
    ax.legend(loc="center right")
    save(figure, out / "rating_distribution.png")


def plot_feature_histograms(data, varying, out):
    columns = 4
    rows = int(np.ceil(len(varying) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(11, 2.1 * rows))
    for ax, name in zip(axes.flat, varying):
        values = data[name]
        if values.nunique() <= 12:
            counts = values.value_counts().sort_index()
            ax.bar(counts.index, counts.values, width=0.8, color=SERIES[0])
        else:
            ax.hist(values, bins=30, color=SERIES[0], edgecolor="white", linewidth=0.5)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(varying):]:
        ax.set_visible(False)
    figure.suptitle("Phân bố các đặc trưng số (bỏ cột hằng)", color=TEXT, fontsize=11)
    save(figure, out / "feature_histograms.png")


def plot_correlations(data, out):
    corr = data.corr(method="spearman").fillna(0)
    figure, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    ax.grid(False)
    figure.colorbar(image, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("Tương quan Spearman giữa các đặc trưng và Rating")
    save(figure, out / "correlation_heatmap.png")
    with_rating = corr["Rating"].drop("Rating").sort_values()
    figure, ax = plt.subplots(figsize=(6.5, 0.28 * len(with_rating) + 1))
    ax.barh(with_rating.index, with_rating.values, color=SERIES[0], height=0.7)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    for y, value in enumerate(with_rating.values):
        ax.annotate(f"{value:+.2f}", (value, y), fontsize=7, color=MUTED, va="center",
                    ha="left" if value >= 0 else "right",
                    xytext=(3 if value >= 0 else -3, 0), textcoords="offset points")
    ax.set(xlabel="Spearman ρ với Rating", title="Mức liên hệ đơn biến của từng đặc trưng với Rating")
    ax.tick_params(axis="y", labelsize=7)
    save(figure, out / "rating_correlation.png")
    return corr


def plot_rating_by_level(data, out, min_count=30):
    panels = (("Tổng vật chất hai bên", data["white_material"] + data["black_material"]),
              ("Số nước (fullmove, cắt ở phân vị 99)", data["fullmove"].clip(upper=data["fullmove"].quantile(.99))))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (label, level) in zip(axes, panels):
        if level.nunique() > 40:
            level = pd.cut(level, 30).map(lambda interval: interval.mid).astype(float)
        stats = data["Rating"].groupby(level).agg(["median", lambda s: s.quantile(.25),
                                                   lambda s: s.quantile(.75), "count"])
        stats.columns = ["median", "q1", "q3", "count"]
        stats = stats[stats["count"] >= min_count]
        ax.fill_between(stats.index, stats["q1"], stats["q3"], color=SERIES[0], alpha=0.18,
                        linewidth=0, label="Khoảng Q1–Q3")
        ax.plot(stats.index, stats["median"], color=SERIES[0], label="Trung vị Rating")
        ax.set(xlabel=label, title=f"Rating theo {label.split(' (')[0].lower()}")
    axes[0].set_ylabel("Rating")
    axes[0].legend(loc="upper left")
    save(figure, out / "rating_by_material_and_move.png")


def compare_svd(features, numeric, train, valid, k, cumulative, svd_cumulative, out):
    """PCA centres the data; SVD without centring spends its first axis on the mean direction."""
    scaler = build_preprocess(numeric, center=False).fit(features.iloc[train])
    x_train, x_valid = scaler.transform(features.iloc[train]), scaler.transform(features.iloc[valid])
    mean = x_train.mean(axis=0)
    _, _, pca_axes = np.linalg.svd(x_train - mean, full_matrices=False)
    _, _, svd_axes = np.linalg.svd(x_train, full_matrices=False)
    dims = np.arange(1, x_train.shape[1] + 1)
    total = ((x_valid - mean) ** 2).sum()

    def unexplained(axes, shift):
        shifted = x_valid - shift
        return np.array([((shifted - shifted @ axes[:d].T @ axes[:d]) ** 2).sum() / total for d in dims])

    pca_error, svd_error = unexplained(pca_axes, mean), unexplained(svd_axes, 0)
    # Same definition as TruncatedSVD.explained_variance_ratio_: variance of the scores over total variance.
    svd_ratio = np.cumsum((x_train @ svd_axes.T).var(axis=0)) / x_train.var(axis=0).sum()
    table = pd.DataFrame({"dimensions": dims, "pca_cumulative_train": cumulative[:len(dims)],
                          "svd_centered_cumulative_train": np.pad(svd_cumulative, (0, len(dims) - len(svd_cumulative)),
                                                                  constant_values=np.nan),
                          "svd_uncentered_cumulative_train": svd_ratio,
                          "pca_unexplained_validation": pca_error,
                          "svd_uncentered_unexplained_validation": svd_error})
    table.to_csv(out / "pca_vs_svd.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(dims, cumulative[:len(dims)], color=SERIES[0], marker="o", markersize=4, label="PCA")
    ax.plot(dims[:len(svd_cumulative)], svd_cumulative, color=SERIES[0], linestyle="--", linewidth=1,
            label="TruncatedSVD, đã tâm hóa (như runner)")
    ax.plot(dims, svd_ratio, color=SERIES[1], marker="s", markersize=4, label="TruncatedSVD, chưa tâm hóa")
    ax.set(xlabel="Số chiều giữ lại", ylabel="Phương sai tích lũy", ylim=(0, 1.02),
           title="Phương sai giải thích trên train")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(dims, pca_error, color=SERIES[0], marker="o", markersize=4, label="PCA")
    ax.plot(dims, svd_error, color=SERIES[1], marker="s", markersize=4, label="TruncatedSVD, chưa tâm hóa")
    ax.set(xlabel="Số chiều giữ lại", ylabel="Tỷ lệ phương sai không tái tạo được",
           ylim=(0, None), title="Sai số tái tạo trên validation")
    ax.legend(loc="upper right")
    for ax in axes:
        ax.axvline(k, color=MUTED, linewidth=0.8, linestyle=":")
    save(figure, out / "pca_vs_svd_variance.png")
    return dict(svd_uncentered_cumulative_at_k=float(svd_ratio[k-1]),
                pca_validation_unexplained_at_k=float(pca_error[k-1]),
                svd_uncentered_validation_unexplained_at_k=float(svd_error[k-1]),
                svd_uncentered_dims_to_match_pca_at_k=int(np.argmax(svd_error <= pca_error[k-1] + 1e-12) + 1),
                svd=x_valid @ svd_axes[:2].T)


def plot_pca_svd_projection(pca, z_valid, svd_scores, rating, out):
    pca_scores = pca.transform(z_valid)[:, :2]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    image = None
    for ax, scores, name in zip(axes, (pca_scores, svd_scores), ("PC", "SV")):
        image = ax.hexbin(scores[:, 0], scores[:, 1], C=rating, gridsize=30, reduce_C_function=np.mean,
                          cmap=SEQUENTIAL, mincnt=3, linewidths=0)
        ax.set(xlabel=f"{name}1", ylabel=f"{name}2",
               title="PCA (đã tâm hóa)" if name == "PC" else "TruncatedSVD (chưa tâm hóa)")
        ax.grid(False)
    figure.colorbar(image, ax=axes, shrink=0.8, label="Rating trung bình trong ô")
    figure.suptitle("Tập validation trên hai thành phần đầu (ô có ≥3 bài)", color=TEXT, fontsize=11)
    figure.savefig(out / "pca_vs_svd_projection.png", dpi=150)
    plt.close(figure)


def run_pca(frame, features, numeric, out, seed, fraction):
    train, valid = split_groups(frame, fraction, seed)
    preprocess = build_preprocess(numeric).fit(features.iloc[train])
    names = clean_names(preprocess.get_feature_names_out())
    z_train, z_valid = preprocess.transform(features.iloc[train]), preprocess.transform(features.iloc[valid])
    pca = PCA(random_state=seed).fit(z_train)
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    k = reduced_dimensions(numeric)
    svd_k = min(z_train.shape[1] - 1, z_train.shape[0] - 1)
    svd = TruncatedSVD(n_components=svd_k, random_state=seed).fit(z_train)
    svd_cumulative = np.cumsum(svd.explained_variance_ratio_)

    figure, ax = plt.subplots(figsize=(8, 3.8))
    positions = np.arange(1, len(ratio) + 1)
    ax.bar(positions, ratio, color=SERIES[0], width=0.7, label="Từng thành phần")
    ax.step(positions, cumulative, where="mid", color=SERIES[1], label="Tích lũy")
    ax.axhline(0.9, color=MUTED, linewidth=0.8, linestyle="--")
    ax.axvline(k + 0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate(f"k={k} (1/3 số chiều): {cumulative[k-1]:.1%}", (k + 0.6, cumulative[k-1]),
                fontsize=8, color=TEXT, va="top", xytext=(4, -6), textcoords="offset points")
    ax.set(xlabel="Thành phần chính", ylabel="Tỷ lệ phương sai giải thích", ylim=(0, 1.02),
           title="PCA: phương sai giải thích (fit trên train)")
    ax.set_xticks(positions)
    ax.legend(loc="center right")
    save(figure, out / "pca_explained_variance.png")

    comparison = compare_svd(features, numeric, train, valid, k, cumulative, svd_cumulative, out)
    plot_pca_svd_projection(pca, z_valid, comparison.pop("svd"), frame["Rating"].iloc[valid].to_numpy(), out)

    shown = min(6, len(ratio))
    loadings = pd.DataFrame(pca.components_[:shown], columns=names,
                            index=[f"PC{i+1}" for i in range(shown)])
    limit = np.abs(loadings.to_numpy()).max()
    figure, ax = plt.subplots(figsize=(10, 3.4))
    image = ax.imshow(loadings, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(shown), [f"{i} ({r:.1%})" for i, r in zip(loadings.index, ratio)], fontsize=8)
    ax.grid(False)
    figure.colorbar(image, ax=ax, label="Hệ số tải")
    ax.set_title(f"Hệ số tải của {shown} thành phần chính đầu tiên")
    save(figure, out / "pca_loadings.png")

    pairs = min(4, len(ratio))
    projected = pca.transform(z_valid)[:, :pairs]
    rating = frame["Rating"].iloc[valid].to_numpy()
    figure, axes = plt.subplots(pairs - 1, pairs - 1, figsize=(9, 8), squeeze=False)
    image = None
    for row in range(1, pairs):
        for column in range(pairs - 1):
            ax = axes[row - 1, column]
            if column >= row:
                ax.set_visible(False)
                continue
            image = ax.hexbin(projected[:, column], projected[:, row], C=rating, gridsize=30,
                              reduce_C_function=np.mean, cmap=SEQUENTIAL, mincnt=3, linewidths=0)
            ax.set_xlabel(f"PC{column+1}") if row == pairs - 1 else ax.set_xticklabels([])
            ax.set_ylabel(f"PC{row+1}") if column == 0 else ax.set_yticklabels([])
            ax.grid(False)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.6, label="Rating trung bình trong ô")
    figure.suptitle(f"Tập validation chiếu lên PC1–PC{pairs} (ô có ≥3 bài)", color=TEXT, fontsize=11)
    figure.savefig(out / "pca_components_by_rating.png", dpi=150)
    plt.close(figure)

    train_scores = pca.transform(z_train)
    rating_corr = [float(pd.Series(train_scores[:, i]).corr(
        pd.Series(frame["Rating"].iloc[train].to_numpy()), method="spearman")) for i in range(len(ratio))]
    table = pd.DataFrame({"component": [f"PC{i}" for i in positions], "explained_variance_ratio": ratio,
                          "cumulative": cumulative, "spearman_with_rating_train": rating_corr})
    table.to_csv(out / "pca_variance.csv", index=False)
    pd.DataFrame(pca.components_, columns=names, index=table["component"]).to_csv(out / "pca_loadings.csv")
    def needed(threshold):
        return int(np.searchsorted(cumulative, threshold) + 1)
    return dict(validation_fraction=fraction, train_rows=len(train), input_dimensions=len(names),
                reduced_k=k, pca_cumulative_at_k=float(cumulative[k-1]),
                svd_cumulative_at_k=float(svd_cumulative[k-1]),
                components_for_80=needed(.8), components_for_90=needed(.9), components_for_95=needed(.95),
                first_six_ratio=[float(r) for r in ratio[:6]], **comparison)


def run_eda(frame, features, out, seed=42, fraction=0.2):
    out.mkdir(parents=True, exist_ok=True)
    numeric = features.select_dtypes(include="number").columns.tolist()
    constant = [c for c in numeric if features[c].nunique() <= 1]
    varying = [c for c in numeric if c not in constant]
    data = features[varying].assign(Rating=frame["Rating"].to_numpy())
    data.describe().T.to_csv(out / "feature_summary.csv")
    plot_rating(frame, features, out)
    plot_feature_histograms(data, varying, out)
    corr = plot_correlations(data, out)
    corr.to_csv(out / "spearman_correlation.csv")
    plot_rating_by_level(data, out)
    strongest = corr["Rating"].drop("Rating").abs().sort_values(ascending=False)
    summary = dict(rows=len(frame), seed=seed, constant_features=constant,
                   turn_share=features["turn"].value_counts(normalize=True).round(4).to_dict(),
                   top_rating_correlations={k: round(float(corr.loc[k, "Rating"]), 4)
                                            for k in strongest.index[:8]},
                   pca=run_pca(frame, features, numeric, out, seed, fraction))
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output", default="outputs/eda")
    args = parser.parse_args()
    frame = load_sample(args.data, args.sample, args.seed)
    summary = run_eda(frame, make_features(frame), Path(args.output), args.seed, args.validation_fraction)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
