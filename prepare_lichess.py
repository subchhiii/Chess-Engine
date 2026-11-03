"""Build extra training data from the Lichess evaluation database (lichess_db_eval.jsonl.zst).

Streams the compressed file (no need to unpack ~100+ GB), keeps every STRIDE-th position, picks the deepest
evaluation, converts it to the same format as your dataset (side-to-move relative, clipped to +-1000 cp, /1000),
removes any position that is already in your train/validation/test files, and saves  lichess_extra_cnn.npz.
Install once:  pip install zstandard
"""
import io, os, glob, json, sys, numpy as np, chess, zstandard
import pandas as pd

# ======================= SETTINGS =======================
ZST_PATH = r"C:\Work\lichess_db_eval.jsonl.zst"      # where you saved the download
DATA_DIR = r"C:\Work\Fen analysis"                    # your existing dataset (CSV files)
OUT_DIR = DATA_DIR
STRIDE = 180               # keep every 180th line (the file has several hundred million; 180 spreads ~2M picks over almost all of it)
MAX_POSITIONS = 2_000_000  # stop after this many kept positions (if the file ends first you simply get fewer)
MIN_DEPTH = 20             # ignore evaluations shallower than this
MAX_EXTREME_SHARE = 0.07   # at most this share of kept positions may be at the +-1 clip (your original data: ~5%)
# ========================================================

def board_to_tensor(board):                          # identical to the encoding used for your dataset
    x = np.zeros((18, 8, 8), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        c = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        x[c, 7 - chess.square_rank(sq), chess.square_file(sq)] = 1
    if board.turn == chess.WHITE: x[12] = 1
    if board.has_kingside_castling_rights(chess.WHITE): x[13] = 1
    if board.has_queenside_castling_rights(chess.WHITE): x[14] = 1
    if board.has_kingside_castling_rights(chess.BLACK): x[15] = 1
    if board.has_queenside_castling_rights(chess.BLACK): x[16] = 1
    if board.ep_square is not None:
        x[17, 7 - chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    return x

def key4(fen):                                       # position identity: pieces, side, castling, en passant
    return " ".join(fen.split()[:4])

VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
def material_balance(board):                         # white minus black
    return sum(VALS.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1) for p in board.piece_map().values())

def existing_keys():
    keys = set()
    for name in ("train", "validation", "test"):
        for path in glob.glob(os.path.join(DATA_DIR, f"*{name}.csv")):
            keys.update(key4(f) for f in pd.read_csv(path, usecols=["FEN"])["FEN"])
    print(f"existing positions to exclude: {len(keys)}")
    return keys

def main():
    exclude = existing_keys()
    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    kept, n_lines, skipped_depth, skipped_dupe, skipped_extreme, n_extreme = [], 0, 0, 0, 0, 0
    with open(ZST_PATH, "rb") as fh:
        reader = io.BufferedReader(dctx.stream_reader(fh), buffer_size=1 << 20)
        for i, line in enumerate(reader):
            n_lines = i + 1
            if i % STRIDE: continue
            rec = json.loads(line)
            best = max(rec["evals"], key=lambda e: e["depth"])
            if best["depth"] < MIN_DEPTH: skipped_depth += 1; continue
            pv = best["pvs"][0]
            if pv.get("cp") is not None: value = pv["cp"]
            elif pv.get("mate") is not None: value = 100000 if pv["mate"] > 0 else -100000
            else: continue
            if key4(rec["fen"]) in exclude: skipped_dupe += 1; continue
            extreme = abs(value) >= 1000
            if extreme and n_extreme >= MAX_EXTREME_SHARE * (len(kept) + 1):      # too many lopsided positions already
                skipped_extreme += 1; continue
            n_extreme += extreme
            kept.append((rec["fen"], value))
            if len(kept) % 100000 == 0: print(f"  kept {len(kept)} (read {n_lines:,} lines)", flush=True)
            if len(kept) >= MAX_POSITIONS: break
    print(f"read {n_lines:,} lines, kept {len(kept)}, skipped for depth {skipped_depth}, already in your data {skipped_dupe}, "
          f"thinned-out extreme positions {skipped_extreme}")

    if not kept: sys.exit("No positions kept - check ZST_PATH and settings.")

    # --- which perspective are the scores in?  Test on black-to-move positions with a material imbalance ---
    vals = np.array([v for _, v in kept], dtype=np.float64)
    turn_white = np.array([f.split()[1] == "w" for f, _ in kept])
    probe = [i for i in range(min(len(kept), 100000)) if not turn_white[i]]
    mb = np.array([material_balance(chess.Board(kept[i][0] + " 0 1")) for i in probe], dtype=np.float64)
    corr = np.corrcoef(mb, np.clip(vals[probe], -1000, 1000))[0, 1]    # clip so mate scores don't swamp the test
    print(f"correlation between score and (white-black) material on black-to-move positions: {corr:.3f}")
    if abs(corr) < 0.2:
        sys.exit("Could not tell which perspective the scores use (correlation too weak). Stopping - tell Claude.")
    white_pov = corr > 0                                # black to move, white ahead -> positive score means white's point of view
    print("scores are from WHITE's point of view -> converting to side-to-move" if white_pov
          else "scores are already from the SIDE TO MOVE's point of view")
    sign = np.where(turn_white | (not white_pov), 1.0, -1.0)
    y = (np.clip(vals * sign, -1000, 1000) / 1000).astype(np.float32)

    # --- encode one position at a time into a preallocated uint8 array (keeps memory use small) ---
    X = np.zeros((len(kept), 18, 8, 8), dtype=np.uint8)
    for i, (f, _) in enumerate(kept):
        X[i] = board_to_tensor(chess.Board(f + " 0 1"))
        if (i + 1) % 250000 == 0: print(f"  encoded {i + 1}", flush=True)
    out = os.path.join(OUT_DIR, "lichess_extra_cnn.npz")
    np.savez_compressed(out, X=X, y=y)
    print(f"saved {out}: X {X.shape}, y mean {y.mean():.3f}, std {y.std():.3f}, share at +-1: {(np.abs(y) >= 0.999).mean():.3f}")
    print("(your original training data: mean 0.029, std 0.426, about 5% at +-1)")

if __name__ == "__main__":
    main()
