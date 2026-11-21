import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.classification import (
    apply_rating_bins,
    evaluate_classification,
    fit_rating_bins,
    run_classification_nb,
)
from src.clustering_gmm import (
    compare_with_kmeans,
    run_gmm,
    tune_gmm,
)
from src.data import load_sample
from src.features import make_features
from src.mlp_analysis import run_mlp
from src.models.mlp import build_model
from src.preprocessing import build_regressor
from tests import test_data

def _load_fixture(folder, n=150, seed=42):
    path = Path(folder) / "fixture.csv"
    test_data.DataTests().fixture(n).to_csv(path, index=False)
    return load_sample(path, n, seed)

class MLPTests(unittest.TestCase):
    def test_mlp_build_model(self):
        model = build_model(seed=42, hidden_layer_sizes=(32, 16), alpha=0.01)
        self.assertIsInstance(model, MLPRegressor)
        self.assertEqual(model.hidden_layer_sizes, (32, 16))
        self.assertEqual(model.alpha, 0.01)
        self.assertEqual(model.random_state, 42)

    def test_mlp_pipeline_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = _load_fixture(folder)
            features = make_features(frame)
            numeric = features.select_dtypes(include="number").columns.tolist()

            model = build_model(
                seed=42, hidden_layer_sizes=(16,), max_iter=20, early_stopping=False
            )
            regressor = build_regressor(numeric, model, representation="original", seed=42)
            regressor.fit(features, frame["Rating"])
            preds = regressor.predict(features)
            self.assertEqual(len(preds), len(frame))
            self.assertTrue(np.isfinite(preds).all())

    def test_mlp_run_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = _load_fixture(folder)
            features = make_features(frame)

            quick_grid = {
                "hidden_layer_sizes": [(16,)],
                "alpha": [1e-3],
                "learning_rate_init": [1e-3],
            }
            out = Path(folder) / "mlp"
            summary = run_mlp(
                frame,
                features,
                out,
                seed=42,
                folds=2,
                param_grid=quick_grid,
                n_jobs=1,
                n_iter=1,
                epochs=30,
            )

            self.assertIn("seed", summary)
            self.assertIn("details_frac_02", summary)
            self.assertIn("validation", summary)

            for name in (
                "mlp_metrics.csv",
                "mlp_summary.csv",
                "mlp_summary.json",
                "mlp_cv_results.csv",
                "mlp_learning_curve.csv",
                "mlp_residuals_by_rating.csv",
                "mlp_loss_curve.png",
                "mlp_tuning_overfit.png",
                "mlp_learning_curve.png",
                "mlp_residuals.png",
            ):
                self.assertTrue(
                    (out / name).exists(), f"Thiếu artifact: {name}"
                )

            loaded = json.loads((out / "mlp_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["seed"], 42)
            self.assertEqual(loaded["folds"], 2)

class ClassificationTests(unittest.TestCase):
    def test_fit_rating_bins_from_train(self):
        y_train = pd.Series([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        thresholds, binned = fit_rating_bins(y_train, n_bins=3)
        self.assertEqual(len(thresholds), 2)
        self.assertTrue(np.all(np.diff(thresholds) > 0))  # tăng dần
        self.assertEqual(set(binned.unique()), {0, 1, 2})

    def test_fit_rating_bins_rejects_invalid_n_bins(self):
        with self.assertRaises(ValueError):
            fit_rating_bins(pd.Series([100, 200, 300]), n_bins=1)

    def test_fit_rating_bins_rejects_duplicate_thresholds(self):
        with self.assertRaises(ValueError):
            fit_rating_bins(pd.Series([100, 100, 100, 100]), n_bins=3)

    def test_apply_rating_bins(self):
        thresholds = [200, 400]
        y = pd.Series([100, 200, 300, 400, 500])
        bins = apply_rating_bins(y, thresholds)
        self.assertEqual(bins.tolist(), [0, 0, 1, 1, 2])

    def test_apply_rating_bins_rejects_nan(self):
        with self.assertRaises(ValueError):
            apply_rating_bins(pd.Series([100, np.nan, 300]), [200])

    def test_evaluate_classification(self):
        metrics = evaluate_classification([0, 0, 1, 1], [0, 1, 1, 1], labels=[0, 1])
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)

    def test_run_classification_nb_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = _load_fixture(folder)
            features = make_features(frame)
            out = Path(folder) / "classification"

            summary = run_classification_nb(
                frame, features, out, seed=42, n_bins_list=(3,)
            )
            self.assertIn("details_frac_02", summary)
            self.assertIn("validation", summary)

            for name in (
                "nb_metrics.csv",
                "nb_summary.json",
                "nb_confusion_original_3bins.png",
                "nb_confusion_pca_3bins.png",
            ):
                self.assertTrue(
                    (out / name).exists(), f"Thiếu artifact: {name}"
                )


class GMMTests(unittest.TestCase):
    def test_tune_gmm_small_range(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))

        bic_table, best_model = tune_gmm(
            X,
            k_range=range(2, 4),
            covariance_types=("full", "diag"),
            seed=42,
        )

        self.assertEqual(len(bic_table), 4)
        self.assertIn("k", bic_table.columns)
        self.assertIn("covariance_type", bic_table.columns)
        self.assertIn("bic", bic_table.columns)
        self.assertIn("aic", bic_table.columns)

        self.assertIn("best_k", bic_table.attrs)
        self.assertIn("best_covariance_type", bic_table.attrs)
        self.assertEqual(best_model.n_components, bic_table.attrs["best_k"])
        self.assertEqual(
            best_model.covariance_type, bic_table.attrs["best_covariance_type"]
        )

    def test_compare_with_kmeans(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 4))

        labels_a = np.array([0, 1] * 30)         
        labels_b = np.array([0, 0, 1, 1] * 15)         
        true_bins = labels_a.copy()                    

        df = compare_with_kmeans(X, labels_a, labels_b, true_bins, seed=42)

        self.assertEqual(len(df), 3)
        self.assertEqual(
            set(df["metric"]),
            {"Silhouette Score", "ARI (vs Rating bins)", "NMI (vs Rating bins)"},
        )
        self.assertIn("gmm", df.columns)
        self.assertIn("kmeans", df.columns)

        ari_row = df[df["metric"] == "ARI (vs Rating bins)"].iloc[0]
        self.assertAlmostEqual(ari_row["gmm"], 1.0, places=7)
        self.assertLess(ari_row["kmeans"], 1.0)

        nmi_row = df[df["metric"] == "NMI (vs Rating bins)"].iloc[0]
        self.assertAlmostEqual(nmi_row["gmm"], 1.0, places=7)
        self.assertLess(nmi_row["kmeans"], 1.0)

        sil_row = df[df["metric"] == "Silhouette Score"].iloc[0]
        self.assertFalse(np.isnan(sil_row["gmm"]))
        self.assertFalse(np.isnan(sil_row["kmeans"]))
        
    def test_run_gmm_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = _load_fixture(folder)
            features = make_features(frame)
            out = Path(folder) / "clustering"

            summary = run_gmm(
                frame,
                features,
                out,
                seed=42,
                n_bins=3,
                k_range=range(2, 4),
                covariance_types=("full", "diag"),
                kmeans_labels_path=None, 
            )

            self.assertIn("best_k", summary)
            self.assertIn("best_covariance_type", summary)
            self.assertIn("fit_on", summary)
            self.assertEqual(summary["fit_on"], "train")
            self.assertIsNone(summary["compare_with_kmeans"])

            for name in (
                "gmm_bic_aic.csv",
                "gmm_labels.csv",
                "gmm_summary.json",
                "gmm_bic_aic.png",
                "gmm_clusters_pc.png",
                "gmm_rating_by_cluster.png",
            ):
                self.assertTrue(
                    (out / name).exists(), f"Thiếu artifact: {name}"
                )

            labels_df = pd.read_csv(out / "gmm_labels.csv")
            for col in ("PuzzleId", "split", "gmm_cluster", "rating", "rating_bin"):
                self.assertIn(col, labels_df.columns)
            self.assertEqual(set(labels_df["split"].unique()), {"train", "validation"})