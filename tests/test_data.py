import tempfile
import unittest
from pathlib import Path
import chess
import numpy as np
import pandas as pd
from src.data import clean_sample, load_sample, split_groups
from src.features import extract_fen, position_key, make_features


class DataTests(unittest.TestCase):
    def fixture(self, n=120):
        rows = []
        board = chess.Board()
        rng = np.random.default_rng(12)
        for i in range(n):
            if board.is_game_over():
                board.reset()
            rows.append(dict(PuzzleId=str(i), FEN=board.fen(), Rating=1000+i,
                             GameUrl=f"https://lichess.org/{i//3:08d}"))
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        return pd.DataFrame(rows)

    def test_invalid_fen_rejected(self):
        for fen in ("bad", "8/8/8/8/8/8/8/8 w - - 0 1"):
            with self.assertRaises(ValueError):
                extract_fen(fen)

    def test_features_and_clock_independent_key(self):
        fen = chess.STARTING_FEN
        features = extract_fen(fen)
        self.assertEqual(len(features), 23)
        self.assertEqual(features["white_p"], 8)
        self.assertEqual(features["material_difference"], 0)
        self.assertEqual(position_key(fen), position_key(fen.rsplit(" ", 2)[0]+" 7 10"))

    def test_bad_rows_and_duplicates_accounted(self):
        source = self.fixture(5)
        duplicate = source.iloc[[0]].copy()
        duplicate["PuzzleId"] = "different"
        bad = source.iloc[[1]].copy()
        bad["PuzzleId"] = "bad"
        bad["FEN"] = "invalid"
        cleaned, report = clean_sample(pd.concat([source, source.iloc[[0]], duplicate, bad], ignore_index=True))
        self.assertEqual(report["duplicate_puzzle_id"], 1)
        self.assertEqual(report["duplicate_position"], 1)
        self.assertEqual(report["invalid_fen"], 1)
        self.assertEqual(len(cleaned), 5)

    def test_sampling_reproducible_and_aligned(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"fixture.csv"
            self.fixture().to_csv(path, index=False)
            a = load_sample(path, 100, 42)
            b = load_sample(path, 100, 42)
            pd.testing.assert_frame_equal(a, b)
            features = make_features(a)
            self.assertTrue(features.index.equals(a["Rating"].index))
            self.assertFalse(features.isna().any().any())
            self.assertTrue(np.isfinite(features.select_dtypes(include="number").to_numpy()).all())
            for ratio in (.2, .3, .4):
                train, valid = split_groups(a, ratio)
                for column in ("game_id", "position_key"):
                    self.assertFalse(set(a.iloc[train][column]) & set(a.iloc[valid][column]))

    def test_missing_rating_and_url_removed(self):
        source = self.fixture(5)
        source.loc[0, "Rating"] = np.nan
        source.loc[1, "Rating"] = float("inf")
        source.loc[2, "GameUrl"] = "https://example.com/abcdefgh"
        cleaned, report = clean_sample(source)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["missing_required"], 1)
        self.assertEqual(report["invalid_rating"], 1)
        self.assertEqual(report["invalid_game_url"], 1)


if __name__ == "__main__":
    unittest.main()
