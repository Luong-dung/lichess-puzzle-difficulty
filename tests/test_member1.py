import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.data import load_sample
from src.eda import run_eda
from src.features import make_features
from src.linear_analysis import run_linear
from tests import test_data


class MemberOneTests(unittest.TestCase):
    def test_eda_and_linear_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"fixture.csv"
            test_data.DataTests().fixture(150).to_csv(path, index=False)
            frame = load_sample(path, 150, 42)
            features = make_features(frame)
            eda = run_eda(frame, features, Path(folder)/"eda")
            self.assertNotIn("white_k", features.columns)
            self.assertIsInstance(eda["constant_features"], list)
            self.assertGreaterEqual(eda["pca"]["pca_cumulative_at_k"], 0)
            linear = run_linear(frame, features, Path(folder)/"linear", folds=3)
            metrics = pd.read_csv(Path(folder)/"linear"/"linear_metrics.csv")
            self.assertEqual(len(metrics), 3 * 3 * 3 * 2)
            self.assertEqual(len(linear["best_alpha"]), 9)
            pca = eda["pca"]
            self.assertGreaterEqual(pca["pca_validation_unexplained_at_k"], 0)
            self.assertLessEqual(pca["pca_validation_unexplained_at_k"], 1)
            for name in ("pca_explained_variance.png", "pca_components_by_rating.png", "correlation_heatmap.png",
                         "pca_vs_svd_variance.png", "pca_vs_svd_projection.png", "pca_vs_svd.csv"):
                self.assertTrue((Path(folder)/"eda"/name).exists())
            for name in ("ridge_residuals.png", "ridge_tuning_curve.png", "ridge_learning_curve.png"):
                self.assertTrue((Path(folder)/"linear"/name).exists())
            json.loads((Path(folder)/"linear"/"linear_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
