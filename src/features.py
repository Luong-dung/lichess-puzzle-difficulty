"""Features available from FEN; never use rating or solution moves as inputs."""
import pandas as pd


def extract_fen(fen):
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError(f"Invalid FEN: {fen}")
    board, turn, castling, ep, halfmove, fullmove = fields
    features = {}
    for color, pieces in (("white", "PNBRQK"), ("black", "pnbrqk")):
        for piece in pieces:
            features[f"{color}_{piece.lower()}"] = board.count(piece)
        features[f"{color}_material"] = sum(
            board.count(p) * value for p, value in zip(pieces, (1, 3, 3, 5, 9, 0)))
    for flag in "KQkq":
        features[f"castle_{flag}"] = int(flag in castling)
    features.update(turn=turn, en_passant=int(ep != "-"),
                    halfmove=int(halfmove), fullmove=int(fullmove))
    features["material_difference"] = features["white_material"] - features["black_material"]
    return features


def make_features(frame):
    return pd.DataFrame([extract_fen(fen) for fen in frame["FEN"]], index=frame.index)
