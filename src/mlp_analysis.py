"""Mô hình hồi quy Multi-Layer Perceptron (MLP) và tinh chỉnh siêu tham số."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV  

from .data import load_sample, split_groups
from .evaluation import regression_metrics
from .features import make_features
from .models.mlp import DEFAULT_PARAM_GRID, build_model
from .preprocessing import build_regressor
from .plot_style import plt, save, SERIES, SEQUENTIAL, MUTED

PARAM_PREFIX = "regressor__model__"


def tune_mlp(numeric, X, y, groups, representation="original", seed=42, folds=5, param_grid=None, n_jobs=-1, n_iter=15, epochs=300):
    # Tìm kiếm siêu tham số bằng GroupKFold trên tập huấn luyện.
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID
    search_grid = {f"{PARAM_PREFIX}{k}": v for k, v in grid.items()}
    base_model = build_regressor(numeric, build_model(seed, max_iter=epochs), representation, seed)
    # thử ngẫu nhiên n_iter tổ hợp tham số
    search = RandomizedSearchCV(
         base_model,
        search_grid,
        n_iter=n_iter,              
        cv=GroupKFold(n_splits=folds),      # Sử dụng GroupKFold để đảm bảo cùng 1 ván cờ
        scoring="neg_root_mean_squared_error",      # Sử dụng RMSE làm tiêu chí đánh giá
        return_train_score=True,
        n_jobs=n_jobs,
        random_state=seed, 
        refit=True,     # Huấn luyện lại mô hình tốt nhất trên toàn bộ tập train
        error_score="raise",
        verbose=1,
    )
    return search.fit(X, y, groups=groups)

# Vẽ biểu đồ hàm mất mát tập train và điểm validation qua từng epoch.
def plot_loss_curve(model, out):
    mlp = model.regressor_.named_steps["model"]
    figure, ax = plt.subplots(figsize=(7, 3.6))
    epochs = np.arange(1, len(mlp.loss_curve_) + 1)
    ax.plot(epochs, mlp.loss_curve_, color=SERIES[0], label="Train loss")
    ax.set(xlabel="Số epoch", ylabel="Loss", title="MLP: đường suy giảm hàm loss qua các epochs")

    # Nếu mô hình có bật early_stopping, vẽ thêm trục Y thứ 2 hiển thị điểm R2 trên tập validation nội bộ
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None and len(mlp.validation_scores_) > 0:
        val_scores = mlp.validation_scores_
        ax2 = ax.twinx()
        ax2.plot(epochs, val_scores, color=SERIES[1], linestyle="--", label="Validation score (R2)")
        ax2.set_ylabel("Validation R2", color=SERIES[1])
        ax2.tick_params(axis="y", labelcolor=SERIES[1])
        ax2.grid(False)

    ax.legend(loc="upper right")
    save(figure, out / "mlp_loss_curve.png")
    
# Vẽ biểu đồ so sánh RMSE giữa train và validation của từng lần thử
def plot_tuning_overfit(search, out):
    results = pd.DataFrame(search.cv_results_)
    train_rmse = -results["mean_train_score"].astype(float).to_numpy()
    val_rmse = -results["mean_test_score"].astype(float).to_numpy()
    best = search.best_index_

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    ax.scatter(train_rmse, val_rmse, s=30, color=SERIES[0], alpha=0.6, label="Trial")
    low = min(train_rmse.min(), val_rmse.min())
    high = max(train_rmse.max(), val_rmse.max())
    ax.plot([low, high], [low, high], color=MUTED, linewidth=1, linestyle="--", label="y = x (không overfit)")
    ax.scatter([train_rmse[best]], [val_rmse[best]], s=150, marker="*",
               color=SERIES[1], label="Lần thử tốt nhất", zorder=5)
    ax.set(xlabel="RMSE train", ylabel="RMSE validation",
           title="Điểm càng xa đường y = x mức độ overfit càng cao")
    ax.legend(loc="upper left")

    ax = axes[1]
    positions = np.arange(len(results))
    ax.barh(positions - 0.2, train_rmse, height=0.4, color=SERIES[0], label="Train")
    ax.barh(positions + 0.2, val_rmse, height=0.4, color=SERIES[1], label="Validation")
    ax.set_yticks(positions, [f"Trial {i}" for i in positions])
    ax.tick_params(axis="y", labelsize=7)
    ax.invert_yaxis()
    ax.set(xlabel="RMSE (Rating)", title="So sánh RMSE train và validation cho từng cấu hình tham số")
    ax.legend(loc="lower right")

    save(figure, out / "mlp_tuning_overfit.png")
    results.to_csv(out / "mlp_cv_results.csv", index=False)
  
# Vẽ đường học theo kích thước tập train
def plot_learning_curve(numeric, X, y, frame, train, valid, best_params, seed, out):
    games = np.random.default_rng(seed).permutation(np.asarray(frame["game_id"].iloc[train].unique(), dtype=object))
    rows = []
    for share in (0.1, 0.25, 0.5, 0.75, 1.0):
        target_count = max(2, int(len(games) * share))
        chosen = set(games[:target_count])
        subset = train[frame["game_id"].iloc[train].isin(chosen).to_numpy()]
        if len(subset) < 10:
            continue
        
        params = dict(best_params)
        params["early_stopping"] = len(subset) >= 50
        model = build_regressor(numeric, build_model(seed, **params), "original", seed)
        model.fit(X.iloc[subset], y.iloc[subset])
        rows.append(
            dict(
                train_rows=len(subset),
                train_rmse=regression_metrics(y.iloc[subset], model.predict(X.iloc[subset]))["rmse"],
                validation_rmse=regression_metrics(y.iloc[valid], model.predict(X.iloc[valid]))["rmse"],
            )
        )
    curve = pd.DataFrame(rows)
    curve.to_csv(out / "mlp_learning_curve.csv", index=False)
    figure, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(curve["train_rows"], curve["train_rmse"], color=SERIES[0], marker="o", markersize=5, label="Train")
    ax.plot(curve["train_rows"], curve["validation_rmse"], color=SERIES[1], marker="s", markersize=5, label="Validation")
    ax.set(xlabel="Số bài trong tập train", ylabel="RMSE (Rating)", title="MLP: đường học theo cỡ tập train (kiểm tra overfit)")
    ax.legend(loc="upper right")
    save(figure, out / "mlp_learning_curve.png")

# Biểu đồ phân tích sai số và độ khớp dự đoán của MLP
def plot_residuals(y_true, prediction, out):
    y_true, prediction = np.asarray(y_true, float), np.asarray(prediction, float)
    residual = y_true - prediction
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    low, high = min(y_true.min(), prediction.min()), max(y_true.max(), prediction.max())

    # Mật độ phân bố giữa Rating thực tế và Rating dự đoán
    image = axes[0].hexbin(y_true, prediction, gridsize=35, cmap=SEQUENTIAL, mincnt=1, bins="log", linewidths=0)
    axes[0].plot([low, high], [low, high], color=MUTED, linewidth=1, linestyle="--", label="Dự đoán đúng (y = x)")
    axes[0].set(xlabel="Rating thực tế", ylabel="Rating dự đoán", title="Thực tế và dự đoán")
    axes[0].legend(loc="upper left")
    figure.colorbar(image, ax=axes[0], label="Số bài (log)")

    # Mật độ phân bố phần dư theo giá trị dự đoán
    image = axes[1].hexbin(prediction, residual, gridsize=35, cmap=SEQUENTIAL, mincnt=1, bins="log", linewidths=0)
    axes[1].axhline(0, color=MUTED, linewidth=1, linestyle="--")
    axes[1].set(xlabel="Rating dự đoán", ylabel="Phần dư", title="Phần dư theo giá trị dự đoán")
    figure.colorbar(image, ax=axes[1], label="Số bài (log)")

    # Sai số trung bình ở từng mức độ khó
    bins = pd.qcut(y_true, 10, duplicates="drop")
    by_bin = pd.DataFrame({"bin": bins, "residual": residual}).groupby("bin", observed=True)["residual"]
    labels = [f"{interval.left:.0f}-{interval.right:.0f}" for interval in by_bin.mean().index]
    axes[2].bar(range(len(labels)), by_bin.mean().values, color=SERIES[0], width=0.7, label="Phần dư trung bình")
    axes[2].plot(range(len(labels)), by_bin.apply(lambda s: s.abs().mean()).values, color=SERIES[1],
                 marker="o", markersize=5, label="MAE")
    axes[2].axhline(0, color=MUTED, linewidth=0.8)
    axes[2].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes[2].tick_params(axis="x", labelsize=7)
    axes[2].set(xlabel="Khoảng Rating thực tế", ylabel="Rating", title="Sai số theo mức Rating thực tế")
    axes[2].legend(loc="upper center")
    save(figure, out / "mlp_residuals.png")
    return pd.DataFrame(
        {"rating_bin": labels, "mean_residual": by_bin.mean().values,
         "mae": by_bin.apply(lambda s: s.abs().mean()).values, "n": by_bin.size().values}
    )

# Trích xuất bộ tham số tốt nhất
def extract_best_params(search):
    return {k.replace(PARAM_PREFIX, ""): v for k, v in search.best_params_.items()}

# Chạy toàn bộ quy trình: tinh chỉnh tham số, so sánh với mô hình cơ sở và xuất biểu đồ
def run_mlp(frame, features, out, seed=42, folds=5, detail_fraction=0.2, param_grid=None, n_jobs=-1, n_iter=15, epochs=300):
    out.mkdir(parents=True, exist_ok=True)
    X, y = features, frame["Rating"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    rows, best_configs, detail = [], {}, None

    # Thử nghiệm trên 3 tỷ lệ chia tập validation
    for fraction in (0.2, 0.3, 0.4):
        train, valid = split_groups(frame, fraction, seed)
        for representation in ("original", "pca", "svd"):
            current_iter = n_iter if fraction == detail_fraction else max(3, n_iter // 3)
            search = tune_mlp(
                numeric,
                X.iloc[train],
                y.iloc[train],
                frame["game_id"].iloc[train],
                representation=representation,
                seed=seed,
                folds=folds,
                param_grid=param_grid,
                n_jobs=n_jobs,
                n_iter=current_iter,
                epochs=epochs
            )
            clean_params = extract_best_params(search)
            best_configs[f"{representation}_{fraction}"] = {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in clean_params.items()
            }
            candidates = {
                "mean_baseline": build_regressor(numeric, DummyRegressor(), representation, seed),
                "ols": build_regressor(numeric, LinearRegression(), representation, seed),
                "mlp_tuned": search.best_estimator_,
            }
            for name, model in candidates.items():
                if name != "mlp_tuned":
                    model.fit(X.iloc[train], y.iloc[train])
                for split, indices in (("train", train), ("validation", valid)):
                    rows.append(
                        dict(
                            model=name,
                            representation=representation,
                            validation_fraction=fraction,
                            cv_rmse=-search.best_score_ if name == "mlp_tuned" else None,
                            split=split,
                            n=len(indices),
                            **regression_metrics(y.iloc[indices], model.predict(X.iloc[indices])),
                        )
                    )
            if fraction == detail_fraction and representation == "original":
                detail = (search, clean_params, train, valid, -search.best_score_)
            print(f"Hoàn tất MLP ({representation}, tỷ lệ val={fraction}), tham số tốt nhất={clean_params}", flush=True)

    if detail is None:
        raise ValueError(
            f"detail_fraction={detail_fraction} không có trong {{0.2, 0.3, 0.4}}."
        )
    
    # Lưu chỉ số đánh giá
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "mlp_metrics.csv", index=False)

    # Vẽ và lưu biểu đồ
    search, clean_params, train, valid, cv_rmse = detail
    model = search.best_estimator_
    plot_tuning_overfit(search, out)
    plot_loss_curve(model, out)
    plot_learning_curve(numeric, X, y, frame, train, valid, clean_params, seed, out)
    residuals = plot_residuals(y.iloc[valid], model.predict(X.iloc[valid]), out)
    residuals.to_csv(out / "mlp_residuals_by_rating.csv", index=False)

    # Bảng tổng hợp so sánh chỉ số train và validation
    pivot = metrics.pivot_table(
        index=["validation_fraction", "representation", "model"],
        columns="split",
        values=["rmse", "mae", "r2"],
    )
    pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
    pivot["rmse_gap"] = pivot["rmse_validation"] - pivot["rmse_train"]
    pivot.round(4).to_csv(out / "mlp_summary.csv")

    # Xuất tóm tắt dạng JSON
    summary = dict(
        seed=seed,
        folds=folds,
        best_configs=best_configs,
        detail=dict(validation_fraction=detail_fraction, representation="original", best_params=clean_params, cv_rmse=cv_rmse),
        validation=pivot.reset_index().query("model == 'mlp_tuned'")[
            ["validation_fraction", "representation", "rmse_train", "rmse_validation",
             "mae_validation", "r2_validation", "rmse_gap"]
        ].round(4).to_dict("records"),
    )
    (out / "mlp_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="lichess_db_puzzle.csv")
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output", default="outputs/mlp")
    args = parser.parse_args()
    
    # Đọc mẫu dữ liệu, tạo đặc trưng và chạy thực nghiệm
    frame = load_sample(args.data, args.sample, args.seed)
    summary = run_mlp(frame, make_features(frame), Path(args.output), args.seed, args.folds, n_iter=args.n_iter, epochs=args.epochs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()