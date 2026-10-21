"""Features available from FEN; never use rating or solution moves as inputs."""
import pandas as pd
import chess


def validated_board(fen):
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise ValueError("FEN must have six fields")
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"Invalid board status: {board.status()}")
    return board


def position_key(fen):
    return " ".join(validated_board(fen).fen(en_passant="legal").split()[:4])



def extract_fen(fen):
    validated_board(fen)
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
