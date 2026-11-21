import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.compare import build_comparison
from src.data import load_sample
from src.features import make_features
from src.logistic_analysis import run_logistic
from src.models.logistic_regression import build_model
from tests import test_data


class LogisticTests(unittest.TestCase):
    def test_build_model_uses_given_c(self):
        model = build_model(seed=7, C=0.5)
        self.assertEqual((model.C, model.random_state), (0.5, 7))

    def test_run_logistic_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.csv"
            test_data.DataTests().fixture(200).to_csv(path, index=False)
            frame = load_sample(path, 200, 42)
            run_logistic(frame, make_features(frame), Path(folder) / "out", folds=3, n_bins_list=(3,))
            metrics = pd.read_csv(Path(folder) / "out" / "lr_metrics.csv")
            # 3 tỷ lệ x 2 biểu diễn x 2 mô hình x 2 split
            self.assertEqual(len(metrics), 24)
            self.assertTrue(metrics["macro_f1"].between(0, 1).all())
            self.assertTrue(metrics.loc[metrics.model == "logistic_regression", "C"].notna().all())
            for name in ("lr_summary.json", "lr_confusion_original_3bins.png", "lr_tuning_curve_3bins.png",
                         "lr_coefficients_3bins.csv"):
                self.assertTrue((Path(folder) / "out" / name).exists(), name)


class CompareTests(unittest.TestCase):
    def test_build_comparison_from_partial_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "linear").mkdir()
            pd.DataFrame([dict(validation_fraction=0.2, representation="original", model=m, mae_train=1,
                               mae_validation=2, r2_train=0.1, r2_validation=r2, rmse_train=3,
                               rmse_validation=rmse, rmse_gap=1)
                          for m, r2, rmse in (("mean_baseline", 0, 10), ("ridge_tuned", 0.06, 9))]
                         ).to_csv(root / "linear" / "linear_summary.csv", index=False)
            (root / "logistic").mkdir()
            pd.DataFrame([dict(model="logistic_regression", representation="original", n_bins=3,
                               validation_fraction=0.2, split=s, n=10, accuracy=0.4, macro_f1=0.39)
                          for s in ("train", "validation")]).to_csv(root / "logistic" / "lr_metrics.csv", index=False)
            (root / "gmm").mkdir()
            (root / "gmm" / "gmm_summary.json").write_text(json.dumps(dict(
                best_k=8, best_covariance_type="full", silhouette=0.07, silhouette_validation=0.07,
                ari_vs_rating_validation=0.0, nmi_vs_rating_validation=0.0, eta2_rating_validation=0.01)))
            missing = root / "missing"
            tables = build_comparison(root / "out", root / "linear", missing, missing, missing,
                                      root / "logistic", missing, root / "gmm")
            self.assertEqual(list(tables["regression"]["model"]), ["mean_baseline", "ridge_tuned"])
            self.assertEqual(tables["classification"]["macro_f1_validation"].iloc[0], 0.39)
            self.assertEqual(tables["clustering"]["k"].iloc[0], 8)
            self.assertIn("Phân cụm", (root / "out" / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
