"""Stronger search engines for the chess bot (they plug into eval_bots.py as 'name@ab2' and 'name@mcts').

FastSearcher  - alpha-beta with the standard engine upgrades: principal-variation search (null-window re-searches),
                null-move pruning, late-move reductions, check extensions, delta pruning in quiescence.
                Goal: reach a deeper search with the same number of (expensive) network evaluations.
MCTSSearcher  - Monte-Carlo tree search (PUCT) with the network as the value function and cheap rule-based move priors
                (captures, checks, promotions), because the network has no policy head."""
import math, time, random
import chess
import eval_bots as eb
from eval_bots import Searcher, _Timeout, MATE, EXACT, LOWER, UPPER, PIECE_VAL

EPS = 1e-5            # width of a 'null window' in network-score units (scores are centipawns / 1000)
NULL_R = 2            # null-move depth reduction
DELTA = 0.2           # quiescence delta-pruning margin (200 cp)


class FastSearcher(Searcher):
    def has_material(self, board):                       # side to move has more than pawns and king (avoids zugzwang errors)
        return bool(board.occupied_co[board.turn] & ~(board.pawns | board.kings))

    def qsearch(self, board, alpha, beta, qd, ply):
        self.tick()
        stand = None
        if board.is_check():
            moves = list(board.legal_moves)
            if not moves: return -MATE + ply
            if qd <= 0: return self.static(board)
            best = -9e9
        else:
            stand = self.static(board)
            if stand >= beta or qd <= 0: return stand
            alpha, best = max(alpha, stand), stand
            moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        for m in self.order(board, moves):
            if stand is not None and board.is_capture(m):                 # delta pruning: capture cannot lift the score to alpha
                victim = board.piece_type_at(m.to_square) or chess.PAWN
                if stand + 0.1 * PIECE_VAL[victim] + DELTA < alpha: continue
            board.push(m)
            s = -self.qsearch(board, -beta, -alpha, qd - 1, ply + 1)
            board.pop()
            if s > best: best = s
            if best > alpha: alpha = best
            if alpha >= beta: break
        return best

    def negamax(self, board, depth, alpha, beta, ply, allow_null=True):
        self.tick()
        if board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material():
            return 0.0
        in_check = board.is_check()
        if in_check and ply < 16: depth += 1                              # check extension
        key = board._transposition_key()
        alpha0, beta0, tt_move = alpha, beta, None
        hit = self.tt.get(key)
        if hit:
            d, flag, s, tt_move = hit
            if d >= depth:
                if flag == EXACT: return s
                if flag == LOWER: alpha = max(alpha, s)
                else: beta = min(beta, s)
                if alpha >= beta: return s
        if depth <= 0:
            return self.qsearch(board, alpha, beta, self.qdepth, ply)
        if allow_null and depth >= 3 and not in_check and abs(beta) < MATE - 100 and self.has_material(board):
            board.push(chess.Move.null())                                  # null-move pruning: if passing still beats beta, cut
            s = -self.negamax(board, depth - 1 - NULL_R, -beta, -beta + EPS, ply + 1, False)
            board.pop()
            if s >= beta: return beta
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0.0
        best, best_move = -9e9, None
        for i, m in enumerate(self.order(board, moves, tt_move, ply)):
            quiet = not board.is_capture(m) and not m.promotion
            reduce = 0
            if depth >= 3 and i >= 3 and quiet and not in_check and not board.gives_check(m):
                reduce = 1                                                  # late move reduction
            board.push(m)
            if i == 0:
                s = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:                                                           # principal-variation search: null window first
                s = -self.negamax(board, depth - 1 - reduce, -alpha - EPS, -alpha, ply + 1)
                if s > alpha and reduce:
                    s = -self.negamax(board, depth - 1, -alpha - EPS, -alpha, ply + 1)
                if s > alpha and s < beta:
                    s = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if s > best: best, best_move = s, m
            if best > alpha: alpha = best
            if alpha >= beta:
                if not board.is_capture(m):
                    k = self.killers.setdefault(ply, [])
                    if m not in k: k.insert(0, m); del k[2:]
                    h = m.from_square * 64 + m.to_square
                    self.history[h] = min(400_000, self.history.get(h, 0) + depth * depth)
                break
        if abs(best) < MATE - 100:
            self.tt[key] = (depth, UPPER if best <= alpha0 else (LOWER if best >= beta0 else EXACT), best, best_move)
        return best


class _Node:
    __slots__ = ("move", "prior", "children", "N", "W", "terminal", "tval")
    def __init__(self, move, prior):
        self.move, self.prior, self.children, self.N, self.W, self.terminal, self.tval = move, prior, None, 0, 0.0, False, 0.0


class MCTSSearcher(Searcher):
    def __init__(self, evalf, simulations=400, c_puct=0.5, time_limit=None, qdepth=2, **kw):
        super().__init__(evalf, max_depth=99, time_limit=time_limit, qdepth=4, adaptive=False)
        self.simulations, self.c, self.leaf_q = simulations, c_puct, qdepth

    def move_priors(self, board, moves):                  # no policy network: cheap rules say which moves look interesting
        raw = []
        for m in moves:
            r = 0.0
            if board.is_capture(m):
                victim = board.piece_type_at(m.to_square) or chess.PAWN
                r += 1.0 + 0.4 * PIECE_VAL[victim] - 0.05 * PIECE_VAL[board.piece_type_at(m.from_square)]
            if m.promotion: r += 2.0
            if board.gives_check(m): r += 0.8
            raw.append(r)
        mx = max(raw); ex = [math.exp(r - mx) for r in raw]; z = sum(ex)
        return [e / z for e in ex]

    def expand(self, node, b):
        moves = list(b.legal_moves)
        if not moves:
            node.terminal, node.tval = True, (-1.0 if b.is_check() else 0.0); return node.tval
        if b.is_repetition(2) or b.is_fifty_moves() or b.is_insufficient_material():
            node.terminal, node.tval = True, 0.0; return 0.0
        v = self.qsearch(b, -9e9, 9e9, self.leaf_q, 0) if self.leaf_q > 0 else self.static(b)   # value for the side to move
        node.children = [_Node(m, p) for m, p in zip(moves, self.move_priors(b, moves))]
        return v

    def select(self, node):
        sq = math.sqrt(node.N + 1)
        parent_q = (node.W / node.N) if node.N else 0.0
        best, best_s = None, -9e9
        for c in node.children:
            q = -(c.W / c.N) if c.N else parent_q - 0.1          # first-play urgency: unvisited moves look a bit worse than their parent
            s = q + self.c * c.prior * sq / (1 + c.N)
            if s > best_s: best, best_s = c, s
        return best

    def search(self, board):
        t0 = time.time(); self.nodes = 0
        moves = list(board.legal_moves)
        if len(moves) == 1:
            self.last_info = {"depth": 1, "score": 0.0, "nodes": 0, "time": 0.0}; return moves[0]
        root = _Node(None, 1.0); max_depth = 0; sims = 0
        try:
            self.expand(root, board.copy())
            for _ in range(self.simulations):
                if self.time_limit and time.time() - t0 > self.time_limit: break
                node, b, path = root, board.copy(), [root]
                while node.children is not None and not node.terminal:
                    node = self.select(node); b.push(node.move); path.append(node)
                v = node.tval if node.terminal else self.expand(node, b)
                max_depth = max(max_depth, len(path) - 1)
                for n in reversed(path):
                    n.N += 1; n.W += v; v = -v
                sims += 1
        except _Timeout:
            pass
        if not root.children or all(c.N == 0 for c in root.children):
            move, score = random.choice(moves), 0.0
        else:
            c = max(root.children, key=lambda c: (c.N, -(c.W / c.N) if c.N else -9))
            move, score = c.move, -(c.W / c.N)
        elapsed = time.time() - t0
        self.last_info = {"depth": max_depth, "score": score, "nodes": sims, "time": elapsed}
        self.moves_played += 1; self.depth_sum += max_depth; self.time_sum += elapsed; self.nodes_sum += sims
        return move
