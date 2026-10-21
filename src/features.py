"""Features of the position shown to the solver, never solution continuation."""
import chess
import pandas as pd


def validated_board(fen):
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise ValueError("FEN must have six fields")
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"Invalid board status: {board.status()}")
    return board


def puzzle_fen(fen, moves):
    board = validated_board(fen)
    if not isinstance(moves, str) or not moves.split():
        raise ValueError("Missing setup move")
    move = chess.Move.from_uci(moves.split()[0])
    if move not in board.legal_moves:
        raise ValueError("Illegal setup move")
    board.push(move)
    return board.fen(en_passant="legal")


def position_key(fen):
    return " ".join(validated_board(fen).fen(en_passant="legal").split()[:4])


def extract_fen(fen):
    board = validated_board(fen)
    features = {}
    phase = 0
    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        for piece in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            features[f"{label}_{chess.piece_symbol(piece)}"] = len(board.pieces(piece, color))
        features[f"{label}_material"] = sum(len(board.pieces(p, color))*v for p,v in
            ((chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)))
        occupied = chess.SquareSet(board.occupied_co[color])
        nonking = [s for s in occupied if board.piece_type_at(s) != chess.KING]
        features[f"{label}_pinned"] = sum(board.is_pinned(color, s) for s in nonking)
        features[f"{label}_hanging"] = sum(board.is_attacked_by(not color, s) and
            not board.is_attacked_by(color, s) for s in nonking)
        attacks = chess.SquareSet()
        for square in occupied:
            attacks |= board.attacks(square)
        features[f"{label}_attacked_squares"] = len(attacks)
        zone = chess.SquareSet(chess.BB_KING_ATTACKS[board.king(color)])
        features[f"{label}_king_zone_attacked"] = sum(board.is_attacked_by(not color, s) for s in zone)
        phase += sum(len(board.pieces(p,color))*w for p,w in
                     ((chess.KNIGHT,1),(chess.BISHOP,1),(chess.ROOK,2),(chess.QUEEN,4)))
    features.update(castle_K=int(board.has_kingside_castling_rights(chess.WHITE)),
                    castle_Q=int(board.has_queenside_castling_rights(chess.WHITE)),
                    castle_k=int(board.has_kingside_castling_rights(chess.BLACK)),
                    castle_q=int(board.has_queenside_castling_rights(chess.BLACK)),
                    turn="w" if board.turn else "b", en_passant=int(board.has_legal_en_passant()),
                    halfmove=board.halfmove_clock, fullmove=board.fullmove_number)
    features["material_difference"] = features["white_material"] - features["black_material"]
    legal = list(board.legal_moves)
    features["legal_moves"] = len(legal)
    features["checking_moves"] = sum(board.gives_check(m) for m in legal)
    features["in_check"] = int(board.is_check())
    features["phase_material"] = min(phase, 24) / 24
    return features


def make_features(frame):
    return pd.DataFrame([extract_fen(fen) for fen in frame["PuzzleFEN"]], index=frame.index)
