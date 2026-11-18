import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from src import kmeans_analysis, rf_reduction, tune_random_forest
from src.sample_cache import cache_problem
from tests import test_data


def run(module, *args):
    with mock.patch.object(sys, "argv", ["prog", *args]):
        module.main()


class MemberTwoTests(unittest.TestCase):
    def test_cache_used_only_when_seed_and_sample_match(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)/"sample.csv"
            self.assertIn("không có", cache_problem(cache, 100, 42))
            cache.write_text("x\n", encoding="utf-8")
            self.assertIn("quality_report.json", cache_problem(cache, 100, 42))
            (Path(folder)/"quality_report.json").write_text(
                json.dumps({"seed": 42, "requested_sample": 100}), encoding="utf-8")
            self.assertIsNone(cache_problem(cache, 100, 42))
            self.assertIn("seed=42", cache_problem(cache, 100, 7))
            self.assertIn("sample=100", cache_problem(cache, 200, 42))

    def test_scripts_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/"fixture.csv"
            test_data.DataTests().fixture(150).to_csv(data, index=False)
            common = ["--data", str(data), "--cache", str(root/"missing.csv"), "--sample", "150"]
            run(tune_random_forest, *common, "--n-iter", "2", "--folds", "2", "--output", str(root/"rf"))
            best = json.loads((root/"rf"/"best.json").read_text(encoding="utf-8"))
            self.assertIn("selection", best)
            self.assertEqual(best["n_features"], 34)
            for name in ("cv_results.csv", "learning_curve.png", "residuals.png", "feature_importance.png"):
                self.assertTrue((root/"rf"/name).exists(), name)
            run(rf_reduction, *common, "--params", str(root/"rf"/"best.json"), "--output", str(root/"red"))
            summary = json.loads((root/"red"/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["params"], best["best_params"])
            self.assertEqual(len(summary["validation"]), 9)
            run(kmeans_analysis, *common, "--k-max", "3", "--output", str(root/"km"))
            self.assertTrue(list((root/"km").glob("kmeans_k*_summary.json")))


if __name__ == "__main__":
    unittest.main()
