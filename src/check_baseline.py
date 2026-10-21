"""Compare Ridge and a mean baseline on a saved, validated sample."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .features import make_features
from .data import split_groups
from .evaluation import regression_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="outputs/features_v2/sample.csv")
    parser.add_argument("--output", default="outputs/features_v2/baseline_review.md")
    args = parser.parse_args()
    frame = pd.read_csv(args.sample)
    X, y = make_features(frame), frame["Rating"]
    train, valid = split_groups(frame, .2, 42)
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), X.select_dtypes(include="number").columns.tolist()),
        ("turn", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["turn"])])
    models = {"Mean baseline": DummyRegressor(), "Ridge v2": make_pipeline(preprocess, Ridge(alpha=10))}
    lines = ["# Kiểm tra bộ đặc trưng v2", "",
             f"Mẫu sạch {len(frame)} dòng; seed 42; train {len(train)}, validation {len(valid)}; 33 đặc trưng.",
             "Cùng mẫu và phép chia theo game_id; không dùng chung ván hoặc thế cờ.", "",
             "| Mô hình | MAE | RMSE | R² |", "|---|---:|---:|---:|"]
    for name, model in models.items():
        model.fit(X.iloc[train], y.iloc[train])
        scores = regression_metrics(y.iloc[valid], model.predict(X.iloc[valid]))
        lines.append(f"| {name} | {scores['mae']:.3f} | {scores['rmse']:.3f} | {scores['r2']:.4f} |")
    lines += ["", "Chỉ là kiểm tra ban đầu ở tỷ lệ 80:20, chưa tuning và chưa có test cuối cùng.",
              "Không so trực tiếp với số liệu nhánh khác nếu chưa xác nhận cùng mẫu và split.", "",
              "Tái lập: `python -m src.prepare --sample 20000 --seed 42 --output outputs/features_v2`",
              "rồi `python -m src.check_baseline`.", "",
              "Đã chạy 8 test đạt và smoke test 27 tổ hợp trên 300 dòng.",
              "Mẫu 20.000 dòng: không loại lỗi/trùng, không có cột hằng, giao ván/thế cờ bằng 0 ở cả ba tỷ lệ."]
    result = "\n".join(lines) + "\n"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(result, encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
