"""Reproducible bounded-memory sample across the complete CSV."""
import numpy as np
import pandas as pd


def load_sample(path, size=20000, seed=42):
    if size < 100:
        raise ValueError("Use at least 100 rows.")
    rng = np.random.default_rng(seed)
    selected = None
    for chunk in pd.read_csv(path, usecols=["PuzzleId", "FEN", "Rating", "GameUrl"],
                             chunksize=100000):
        chunk["_priority"] = rng.random(len(chunk))
        selected = chunk if selected is None else pd.concat([selected, chunk], ignore_index=True)
        selected = selected.nsmallest(size, "_priority")
    if selected is None:
        raise ValueError("Empty CSV")
    selected = selected.drop(columns="_priority").dropna().drop_duplicates("PuzzleId")
    selected["Rating"] = pd.to_numeric(selected["Rating"], errors="raise")
    selected["game_id"] = selected["GameUrl"].str.extract(r"lichess\.org/([A-Za-z0-9]+)")[0]
    return selected.dropna(subset=["game_id"]).reset_index(drop=True)
