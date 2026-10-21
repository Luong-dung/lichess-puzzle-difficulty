"""Bounded-memory sample; quality counters explicitly scoped to source or sample."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .features import position_key


def clean_sample(frame):
    frame = frame.copy()
    report = {"sample_rows_before": len(frame)}
    def remove(mask, name):
        nonlocal frame
        report[name] = int(mask.sum())
        frame = frame.loc[~mask].copy()
    remove(frame[["PuzzleId", "FEN", "Rating", "GameUrl"]].isna().any(axis=1), "missing_required")
    frame["Rating"] = pd.to_numeric(frame["Rating"], errors="coerce")
    remove(~np.isfinite(frame["Rating"]) | (frame["Rating"] <= 0), "invalid_rating")
    frame["game_id"] = frame["GameUrl"].str.extract(r"^https://lichess\.org/([A-Za-z0-9]{8})(?:[/#?]|$)")[0]
    remove(frame["game_id"].isna(), "invalid_game_url")
    remove(frame.duplicated("PuzzleId"), "duplicate_puzzle_id")
    keys = []
    for fen in frame["FEN"]:
        try:
            keys.append(position_key(fen))
        except (ValueError, TypeError):
            keys.append(None)
    frame["position_key"] = keys
    remove(frame["position_key"].isna(), "invalid_fen")
    remove(frame.duplicated("position_key"), "duplicate_position")
    report["sample_rows_after"] = len(frame)
    return frame.reset_index(drop=True), report


def load_sample(path, size=20000, seed=42, return_report=False):
    if size < 100:
        raise ValueError("Use at least 100 rows.")
    rng = np.random.default_rng(seed)
    selected = None
    total, missing, dtypes = 0, {}, {}
    for chunk in pd.read_csv(path, chunksize=100000, dtype={"PuzzleId": "string", "FEN": "string", "GameUrl": "string"}):
        total += len(chunk)
        for column in chunk:
            missing[column] = missing.get(column, 0) + int(chunk[column].isna().sum())
            dtypes.setdefault(column, set()).add(str(chunk[column].dtype))
        chunk = chunk[["PuzzleId", "FEN", "Rating", "GameUrl"]].copy()
        chunk["_priority"] = rng.random(len(chunk))
        selected = chunk if selected is None else pd.concat([selected, chunk], ignore_index=True)
        selected = selected.nsmallest(size, "_priority")
    if selected is None:
        raise ValueError("Empty CSV")
    cleaned, report = clean_sample(selected.drop(columns="_priority"))
    report.update(seed=seed, requested_sample=size, source_rows=total, source_missing=missing,
                  source_dtypes={k: sorted(v) for k, v in dtypes.items()},
                  detailed_validation_scope="sample only; removals are sequential")
    if len(cleaned) < 2:
        raise ValueError("Not enough valid unique rows after cleaning")
    return (cleaned, report) if return_report else cleaned


def split_groups(frame, validation_fraction=0.2, seed=42):
    train, valid = next(GroupShuffleSplit(n_splits=1, test_size=validation_fraction,
                       random_state=seed).split(frame, groups=frame["game_id"]))
    for key in ("game_id", "position_key"):
        if set(frame.iloc[train][key]) & set(frame.iloc[valid][key]):
            raise ValueError(f"Leakage across split: {key}")
    return train, valid
