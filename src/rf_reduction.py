"""Thành viên 2: Random Forest trên dữ liệu gốc, PCA và SVD; so sánh hai cách giảm chiều.

PCA = chuẩn hóa có tâm rồi PCA. SVD = chuẩn hóa không trừ trung bình rồi TruncatedSVD
(SVD chưa tâm hóa, đúng nghĩa khác PCA). Mọi bước fit trên train.

Chạy:  python -m src.rf_reduction
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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from .data import split_groups
from .evaluation import regression_metrics
from .features import make_features
from .sample_cache import load_frame
from .preprocessing import build_preprocess, reduced_dimensions


def build(numeric, method, k, params, seed):
    steps = [("preprocess", build_preprocess(numeric, center=method != "svd"))]
    if method == "pca":
        steps.append(("reduce", PCA(n_components=k, random_state=seed)))
    elif method == "svd":
        steps.append(("reduce", TruncatedSVD(n_components=k, random_state=seed)))
    steps.append(("model", RandomForestRegressor(n_jobs=-1, random_state=seed, **params)))
    return Pipeline(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--cache", default="outputs/features_v2/sample.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params", default="outputs/random_forest_v4/best.json",
                        help="best.json của tune_random_forest; không có thì dùng tham số mặc định")
    parser.add_argument("--output", default="outputs/rf_reduction_v2")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args.data, args.cache, args.sample, args.seed)
    X, y = make_features(frame), frame["Rating"].to_numpy()
    numeric = X.select_dtypes(include="number").columns.tolist()
    params_path = Path(args.params)
    if params_path.exists():
        tuned = json.loads(params_path.read_text(encoding="utf-8"))
        params, params_source = tuned["best_params"], str(params_path)
        if tuned.get("seed") != args.seed or tuned.get("n_features") != X.shape[1]:
            print(f"CẢNH BÁO: {params_path} được tune với seed={tuned.get('seed')}, "
                  f"{tuned.get('n_features')} đặc trưng; hiện tại seed={args.seed}, {X.shape[1]} đặc trưng. "
                  f"Nên chạy lại src.tune_random_forest trước.", flush=True)
    else:
        params = dict(n_estimators=400, max_depth=12, min_samples_leaf=20, max_features=0.8)
        params_source = "mặc định (chưa chạy src.tune_random_forest)"
        print(f"CẢNH BÁO: không có {params_path}; dùng tham số RF mặc định. "
              f"Chạy python -m src.tune_random_forest trước để dùng tham số đã tune.", flush=True)
    print(f"Tham số RF lấy từ: {params_source}", flush=True)
    k_third = reduced_dimensions(numeric)
    dims = len(numeric) + 2
    print(f"{X.shape[1]} đặc trưng ({dims} cột sau one-hot), k = 1/3 = {k_third}; RF {params}", flush=True)

    # 1. Ba tỷ lệ validation x {gốc, PCA k, SVD k}, k = 1/3 số chiều như yêu cầu đề.
    rows, start = [], time.time()
    jobs = [(f, m) for f in (0.2, 0.3, 0.4) for m in ("original", "pca", "svd")]
    for i, (fraction, method) in enumerate(jobs, 1):
        train, valid = split_groups(frame, fraction, args.seed)
        model = build(numeric, method, k_third, params, args.seed).fit(X.iloc[train], y[train])
        for split, idx in (("train", train), ("validation", valid)):
            rows.append(dict(validation_fraction=fraction, representation=method,
                             k=dims if method == "original" else k_third, split=split, n=len(idx),
                             **regression_metrics(y[idx], model.predict(X.iloc[idx]))))
        print(f"[{i}/{len(jobs)}] {method} val={fraction} ({time.time() - start:.0f}s)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "rf_by_representation.csv", index=False)

    # 2. Quét số chiều k cho PCA và SVD (validation 20%), kèm phương sai giải thích trên train.
    train, valid = split_groups(frame, 0.2, args.seed)
    ks = sorted({1, 2, 3, 5, 8, k_third, 15, 20, 25, dims - 1})
    ks = [k for k in ks if k < dims]
    sweep = []
    for method in ("pca", "svd"):
        for k in ks:
            model = build(numeric, method, k, params, args.seed).fit(X.iloc[train], y[train])
            z = model[:-1].transform(X.iloc[train])
            full = model.named_steps["preprocess"].transform(X.iloc[train])
            explained = float(z.var(axis=0).sum() / full.var(axis=0).sum())
            sweep.append(dict(method=method, k=k, explained_variance_train=explained,
                              **{f"val_{m}": v for m, v in
                                 regression_metrics(y[valid], model.predict(X.iloc[valid])).items()}))
            print(f"sweep {method} k={k} explained={explained:.3f} val_rmse={sweep[-1]['val_rmse']:.1f} "
                  f"({time.time() - start:.0f}s)", flush=True)
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "rf_k_sweep.csv", index=False)
    original = table.query("validation_fraction == 0.2 and representation == 'original' and split == 'validation'")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, label in (("pca", "PCA (có tâm)"), ("svd", "SVD (không tâm)")):
        s = sweep[sweep.method == method]
        axes[0].plot(s.k, s.explained_variance_train, "o-", label=label)
        axes[1].plot(s.k, s.val_rmse, "o-", label=label)
    axes[1].axhline(float(original.rmse.iloc[0]), color="gray", ls="--", label="Dữ liệu gốc")
    for ax in axes:
        ax.axvline(k_third, color="gray", ls=":")
        ax.set_xlabel("Số chiều k")
        ax.legend()
    axes[0].set(ylabel="Tỷ lệ phương sai giữ lại (train)", title="Phương sai giải thích")
    axes[1].set(ylabel="RMSE validation", title="Random Forest theo số chiều")
    plt.tight_layout()
    plt.savefig(out / "rf_pca_vs_svd.png", dpi=150)
    plt.close()

    summary = dict(params=params, params_source=params_source, k_third=k_third, input_dimensions=dims,
                   validation=table[table.split == "validation"].round(4).to_dict("records"),
                   train=table[table.split == "train"].round(4).to_dict("records"),
                   seconds=round(time.time() - start))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pivot = table.pivot_table(index=["validation_fraction", "representation"], columns="split",
                              values=["rmse", "r2"]).round(3)
    print(pivot.to_string())
    print(sweep.round(3).to_string())
    print(f"Đã lưu vào {out}")


if __name__ == "__main__":
    main()
