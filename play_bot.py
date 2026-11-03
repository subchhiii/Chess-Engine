import os, random, numpy as np, chess, torch, torch.nn as nn

DATA_DIR = r"C:\Work\Fen analysis"
DEPTH = 2            # search depth in half-moves; 3 is stronger but slower
GAMES = 10           # games vs the random bot (alternating colours)

# ---- load the trained MLP (same architecture as in train_chess.py) ----
model = nn.Sequential(nn.Flatten(), nn.Linear(18*64, 256), nn.ReLU(),
                      nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "mlp.pt"), map_location="cpu"))
model.eval()

# ---- board -> 18x8x8 tensor, identical to the dataset encoding ----
def board_to_tensor(board):
    x = np.zeros((18, 8, 8), dtype=np.float32)
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

# ---- evaluation: score from the point of view of the side to move ----
@torch.no_grad()
def evaluate(board):
    if board.is_checkmate():
        return -1.0                       # side to move has been mated
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    return model(x).item()

# ---- negamax with alpha-beta (the model is side-to-move relative, so we negate each level) ----
def negamax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    best = -2.0
    for move in sorted(board.legal_moves, key=board.is_capture, reverse=True):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best

def best_move(board, depth=DEPTH):
    best, best_score = None, -2.0
    alpha = -2.0
    moves = list(board.legal_moves)
    random.shuffle(moves)                 # variety between equal moves
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -2.0, -alpha)
        board.pop()
        if score > best_score:
            best, best_score = move, score
        alpha = max(alpha, score)
    return best

# ---- test: bot vs random mover ----
def play_game(bot_is_white):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.fullmove_number < 150:
        if (board.turn == chess.WHITE) == bot_is_white:
            board.push(best_move(board))
        else:
            board.push(random.choice(list(board.legal_moves)))
    if board.is_checkmate():
        return "bot" if (board.turn == chess.WHITE) != bot_is_white else "random"
    return "draw"

if __name__ == "__main__":
    random.seed(0)
    tally = {"bot": 0, "random": 0, "draw": 0}
    for g in range(GAMES):
        r = play_game(bot_is_white=(g % 2 == 0))
        tally[r] += 1
        print(f"game {g+1}/{GAMES}: {r}   running: {tally}", flush=True)
