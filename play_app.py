"""Play against your GNN chess bot in a web browser.

Runs locally or over local Wi-Fi so friends can play on their phones/PCs.
Run with:
    python play_app.py
"""
import os, sys, json, time, uuid, socket, threading, datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import chess, chess.pgn
import numpy as np
import torch

# Robust import: prefer eval_bots_ab3, fall back to eval_bots
try:
    import eval_bots_ab3 as eb
except ImportError:
    import eval_bots as eb

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DATA_DIR = getattr(eb, "DATA_DIR", r"C:\Work\Fen analysis")

# Hard time limit per bot move (seconds) to prevent web app freezing during complex midgames
BOT_TIME_LIMIT = 3.0

# Map UI menu options to evaluators
BOTS = {
    "gph_b":    ("Graph Neural Network (gph_b) - Best Engine", "gph_b"),
    "gph_tta":  ("Graph GNN + Flip TTA - Maximum Precision", "gph_b+tta"),
    "material": ("Material Count Only - Weak Baseline", "material"),
}

BOT_LOCK = threading.Lock()          # Bot searches one move at a time
GAMES, GAMES_LOCK = {}, threading.Lock()
_players = {}

# ---- Test Time Augmentation (TTA) Wrapper ----
def make_tta_eval(base_eval):
    def color_flip_board(board):
        flipped = chess.Board()
        flipped.clear_board()
        for sq, piece in board.piece_map().items():
            new_sq = chess.square(7 - chess.square_file(sq), 7 - chess.square_rank(sq))
            new_color = chess.BLACK if piece.color == chess.WHITE else chess.WHITE
            flipped.set_piece_at(new_sq, chess.Piece(piece.piece_type, new_color))
        flipped.turn = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
        return flipped

    def tta_eval(board):
        e1 = base_eval(board)
        e2 = base_eval(color_flip_board(board))
        return (e1 - e2) / 2.0

    return tta_eval


# ---- Extended Searcher with UI Stats & Hard Time Cap ----
class AppSearcher(eb.FastSearcher):
    def tick(self):
        """Check deadline frequently (every 16 nodes) to ensure snappy responses."""
        self.nodes += 1
        if self.deadline is not None and self.nodes % 16 == 0 and time.time() > self.deadline:
            raise eb._Timeout

    def search(self, board):
        if len(self.tt) > 2_000_000:
            self.tt.clear()
        self.nodes = 0
        t0, base = time.time(), len(board.move_stack)
        
        # Enforce deadline timer immediately from root search
        self.deadline = (t0 + self.time_limit) if self.time_limit else None
        
        self.killers, self.history = {}, {}
        moves = list(board.legal_moves)
        import random
        random.shuffle(moves)
        best_move, first, reached = random.choice(moves), None, 0
        last_score = 0.0

        for depth in range(1, self.max_depth + 1):
            try:
                alpha, it_best, it_score = -9e9, None, -9e9
                for m in self.order(board, moves, first):
                    board.push(m)
                    sc = -self.negamax(board, depth - 1, -9e9, -alpha, 1)
                    board.pop()
                    if sc > it_score:
                        it_score, it_best = sc, m
                    alpha = max(alpha, sc)
                best_move = first = it_best
                reached = depth
                last_score = it_score
                if abs(it_score) > eb.MATE - 100:
                    break
            except eb._Timeout:
                # Safely abort incomplete search and fallback to best move from previous complete depth
                while len(board.move_stack) > base:
                    board.pop()
                break

        elapsed = time.time() - t0
        self.moves_played += 1
        self.depth_sum += reached
        self.time_sum += elapsed
        self.nodes_sum += self.nodes

        self.last_info = {
            "depth": reached,
            "nodes": self.nodes,
            "time": elapsed,
            "score": last_score
        }
        return best_move


def available_bots():
    out = {}
    for key, (label, ev) in BOTS.items():
        base = ev[:-4] if ev.endswith("+tta") else ev
        has_pt = os.path.exists(os.path.join(DATA_DIR, base + ".pt"))
        has_pth = os.path.exists(os.path.join(DATA_DIR, base + ".pth"))
        if base == "material" or has_pt or has_pth:
            out[key] = label
    return out


def get_player(bot_key, depth, adaptive):
    k = (bot_key, depth, adaptive)
    if k not in _players:
        raw_ev_name = BOTS[bot_key][1]
        if raw_ev_name.endswith("+tta"):
            base_ev_name = raw_ev_name[:-4]
            evalf = make_tta_eval(eb.get_eval(base_ev_name))
        else:
            evalf = eb.get_eval(raw_ev_name)

        qdepth = getattr(eb, "QDEPTH", 4)
        # Enforce BOT_TIME_LIMIT (3.0s max per move)
        _players[k] = AppSearcher(evalf, max_depth=depth, time_limit=BOT_TIME_LIMIT, qdepth=qdepth, adaptive=adaptive)
    return _players[k]


class Game:
    def __init__(self, human_white, bot_key, depth, adaptive, name):
        self.id = uuid.uuid4().hex[:8]
        self.board = chess.Board()
        self.human_white, self.bot_key, self.depth, self.adaptive = human_white, bot_key, depth, adaptive
        self.name = (name or "Player").strip()[:30] or "Player"
        self.sans, self.last, self.info, self.over, self.result_text, self.saved = [], None, None, False, "", False
        self.lock = threading.Lock()
        self.created = time.time()

    def human_turn(self):
        return (self.board.turn == chess.WHITE) == self.human_white

    def finish_if_over(self, resigned=False):
        if self.over: return
        if resigned:
            self.over, self.result_text, self.human_result, self.reason = True, "You resigned", "L", "resignation"
        else:
            out = self.board.outcome(claim_draw=True)
            if out is None: return
            self.over = True
            self.reason = out.termination.name.lower().replace("_", " ")
            if out.winner is None:
                self.result_text, self.human_result = f"Draw ({self.reason})", "D"
            elif out.winner == self.human_white:
                self.result_text, self.human_result = f"You win ({self.reason})", "W"
            else:
                self.result_text, self.human_result = f"Bot wins ({self.reason})", "L"
        self.save()

    def save(self):
        if self.saved: return
        self.saved = True
        csv_path = os.path.join(DATA_DIR, "games_vs_humans.csv")
        pgn_path = os.path.join(DATA_DIR, "games_vs_humans.pgn")
        try:
            if not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    f.write("time,player,bot,depth,adaptive,human_colour,result_for_human,plies,reason\n")
            with open(csv_path, "a") as f:
                f.write(f'{datetime.datetime.now():%Y-%m-%d %H:%M},"{self.name}",{self.bot_key},{self.depth},{self.adaptive},'
                        f'{"white" if self.human_white else "black"},{self.human_result},{len(self.board.move_stack)},{self.reason}\n')
            g = chess.pgn.Game.from_board(self.board)
            bot_label = BOTS[self.bot_key][0]
            g.headers.update({
                "Event": "Friends vs GNN Chess Bot",
                "Date": f"{datetime.date.today():%Y.%m.%d}",
                "White": self.name if self.human_white else bot_label,
                "Black": bot_label if self.human_white else self.name,
                "Result": {"W": "1-0" if self.human_white else "0-1", "L": "0-1" if self.human_white else "1-0", "D": "1/2-1/2"}[self.human_result]
            })
            with open(pgn_path, "a") as f:
                f.write(str(g) + "\n\n")
        except OSError as e:
            print("Could not save the game:", e)

    def to_json(self):
        legal = {}
        if not self.over and self.human_turn():
            for m in self.board.legal_moves:
                legal.setdefault(chess.square_name(m.from_square), []).append(m.uci())
        check = None
        if self.board.is_check():
            check = chess.square_name(self.board.king(self.board.turn))
        if self.over: status = self.result_text
        elif self.human_turn(): status = "Your move" + (" (you are in check)" if check else "")
        else: status = "Bot is thinking..."
        return {"id": self.id, "fen": self.board.fen(), "legal": legal, "myturn": bool(legal), "last": self.last, "check": check,
                "history": self.sans, "over": self.over, "status": status, "human_white": self.human_white,
                "bot_label": BOTS[self.bot_key][0], "bot": self.info, "player": self.name}


def bot_move(game):
    with game.lock:
        with BOT_LOCK:
            player = get_player(game.bot_key, game.depth, game.adaptive)
            board = game.board.copy()
            move = player.search(board)
            info = getattr(player, "last_info", {"depth": game.depth, "nodes": 0, "time": 0.1, "score": 0.0})

        white_pov = info["score"] if game.board.turn == chess.WHITE else -info["score"]
        mate = abs(info["score"]) > 900
        game.info = {
            "depth": info["depth"],
            "nodes": info["nodes"],
            "seconds": round(info["time"], 2),
            "eval_white": None if mate else round(max(-1.0, min(1.0, white_pov)), 3),
            "mate_for": ("white" if white_pov > 0 else "black") if mate else None
        }
        game.sans.append(game.board.san(move))
        game.board.push(move)
        game.last = move.uci()
        game.finish_if_over()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)

    def send_json(self, obj, code=200): self.send(json.dumps(obj), code=code)

    def do_GET(self):
        if self.path == "/": return self.send(PAGE, "text/html; charset=utf-8")
        if self.path == "/api/bots": return self.send_json([{"key": k, "label": v} for k, v in available_bots().items()])
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 10000: return self.send_json({"error": "request too large"}, 400)
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/new": return self.send_json(self.new_game(req))
            with GAMES_LOCK: game = GAMES.get(str(req.get("id", "")))
            if game is None: return self.send_json({"error": "game not found - start a new one"}, 400)
            
            if self.path == "/api/state":
                return self.send_json(game.to_json())

            if not game.lock.acquire(blocking=False): return self.send_json({"error": "the bot is still thinking"}, 400)
            try:
                if self.path == "/api/move": out = self.move(game, req)
                elif self.path == "/api/undo": out = self.undo(game)
                elif self.path == "/api/resign": game.finish_if_over(resigned=True); out = game.to_json()
                else: out = {"error": "not found"}
            finally:
                game.lock.release()
            self.send_json(out, 400 if "error" in out else 200)
        except Exception as e:
            print("Request error:", repr(e)); self.send_json({"error": "server error: " + str(e)}, 500)

    def new_game(self, req):
        bots = available_bots()
        bot_key = req.get("bot") if req.get("bot") in bots else next(iter(bots))
        depth = int(req.get("depth", 4)); depth = depth if depth in (2, 3, 4, 5) else 4
        side = req.get("side", "white")
        human_white = (side == "white") if side in ("white", "black") else bool(int(time.time()) % 2)
        g = Game(human_white, bot_key, depth, bool(req.get("adaptive")), req.get("name", ""))
        with GAMES_LOCK:
            if len(GAMES) > 300:
                for k in sorted(GAMES, key=lambda k: GAMES[k].created)[:100]: GAMES.pop(k, None)
            GAMES[g.id] = g
        if not g.human_turn():
            threading.Thread(target=bot_move, args=(g,), daemon=True).start()
        return g.to_json()

    def move(self, game, req):
        if game.over or not game.human_turn(): return {"error": "it is not your move"}
        try: mv = chess.Move.from_uci(str(req.get("uci", "")))
        except ValueError: return {"error": "bad move format"}
        if mv not in game.board.legal_moves:
            q = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
            if q in game.board.legal_moves: mv = q
            else: return {"error": "illegal move"}
        game.sans.append(game.board.san(mv)); game.board.push(mv); game.last = mv.uci()
        game.finish_if_over()
        if not game.over:
            threading.Thread(target=bot_move, args=(game,), daemon=True).start()
        return game.to_json()

    def undo(self, game):
        if len(game.board.move_stack) < 2: return {"error": "nothing to take back"}
        if game.over: return {"error": "the game is over"}
        for _ in range(2): game.board.pop(); game.sans.pop()
        game.last = game.board.peek().uci() if game.board.move_stack else None
        game.info = None
        return game.to_json()


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GNN Chess Bot Arena</title>
<style>
 * { box-sizing: border-box; }
 body {
   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
   background: #18181c;
   color: #eee;
   margin: 0;
   padding: 20px 14px;
 }
 .container { max-width: 960px; margin: 0 auto; }
 h1 { font-size: 22px; margin: 0 0 18px; text-align: center; color: #fff; letter-spacing: 0.5px; }
 .wrap { display: flex; flex-wrap: wrap; gap: 24px; justify-content: center; align-items: flex-start; }
 
 #board {
   display: grid;
   grid-template-columns: repeat(8, 1fr);
   grid-template-rows: repeat(8, 1fr);
   width: min(90vw, 520px);
   height: min(90vw, 520px);
   aspect-ratio: 1 / 1;
   border: 3px solid #333;
   border-radius: 4px;
   box-shadow: 0 8px 24px rgba(0,0,0,0.5);
   overflow: hidden;
 }
 .sq {
   position: relative;
   display: flex;
   align-items: center;
   justify-content: center;
   cursor: pointer;
   user-select: none;
   overflow: hidden;
   aspect-ratio: 1 / 1;
 }
 .light { background: #eadeca; }
 .dark { background: #b58863; }
 .sq.last { box-shadow: inset 0 0 0 100px rgba(255, 235, 59, 0.38); }
 .sq.sel { box-shadow: inset 0 0 0 100px rgba(80, 200, 120, 0.55); }
 .sq.check { box-shadow: inset 0 0 0 100px rgba(255, 60, 60, 0.65); }
 
 .piece {
   font-size: clamp(24px, 5.5vw, 42px);
   line-height: 1;
   z-index: 1;
 }
 .piece.w {
   color: #ffffff;
   text-shadow: 0 0 2px #000, 0 1px 3px rgba(0,0,0,0.8);
 }
 .piece.b {
   color: #111111;
   text-shadow: 0 0 1px #fff;
 }
 
 .dot { position: absolute; width: 26%; height: 26%; border-radius: 50%; background: rgba(0,0,0,0.25); z-index: 2; }
 .cap { position: absolute; inset: 6%; border: 4px solid rgba(0,0,0,0.3); border-radius: 50%; z-index: 2; }
 
 .coord { position: absolute; font-size: 10px; font-weight: 700; opacity: 0.65; z-index: 3; pointer-events: none; }
 .coord.r { left: 3px; top: 2px; }
 .coord.f { right: 3px; bottom: 2px; }
 .light .coord { color: #b58863; }
 .dark .coord { color: #eadeca; }

 .panel {
   width: min(90vw, 340px);
   background: #24242c;
   border-radius: 10px;
   padding: 16px;
   box-shadow: 0 4px 16px rgba(0,0,0,0.3);
 }
 label { display: block; margin: 10px 0 3px; font-size: 11px; color: #aaa; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
 select, input[type=text] {
   width: 100%;
   padding: 8px 10px;
   border-radius: 6px;
   border: 1px solid #444;
   background: #18181c;
   color: #eee;
   font-size: 13px;
 }
 select:focus, input[type=text]:focus { border-color: #3b82f6; outline: none; }
 button {
   margin: 10px 6px 0 0;
   padding: 8px 14px;
   border: 0;
   border-radius: 6px;
   background: #3b82f6;
   color: #fff;
   cursor: pointer;
   font-size: 13px;
   font-weight: 600;
   transition: background 0.15s;
 }
 button:hover { background: #2563eb; }
 button.alt { background: #444450; }
 button.alt:hover { background: #555565; }
 #status { font-size: 16px; font-weight: 600; margin: 4px 0 8px; color: #fff; }
 #hist {
   font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
   font-size: 12px;
   line-height: 1.5;
   max-height: 120px;
   overflow-y: auto;
   background: #18181c;
   padding: 8px 10px;
   border-radius: 6px;
   min-height: 40px;
   border: 1px solid #333;
 }
 .bar { height: 12px; background: #111; border-radius: 6px; overflow: hidden; margin: 8px 0; border: 1px solid #333; }
 #barfill { height: 100%; background: #eee; width: 50%; transition: width 0.4s ease-in-out; }
 small { color: #888; font-size: 12px; line-height: 1.4; }
 hr { border: 0; border-top: 1px solid #333; margin: 14px 0; }
</style></head><body>
<div class="container">
 <h1>GNN Chess Bot Arena</h1>
 <div class="wrap">
  <div><div id="board"></div></div>
  <div class="panel">
   <div id="status">Press "New game"</div>
   <div id="bar" class="bar"><div id="barfill"></div></div>
   <div id="botinfo"><small>Bot evaluation will appear here (white is the light side of the bar).</small></div>
   <div id="hist"></div>
   <button id="undo" class="alt" onclick="undo()">Take back</button><button class="alt" onclick="resign()">Resign</button><button class="alt" onclick="flip()">Flip board</button>
   <hr>
   <label>Your Name</label><input type="text" id="name" maxlength="30" placeholder="Player">
   <label>Opponent Bot</label><select id="bot"></select>
   <label>Search Depth</label>
   <select id="depth"><option value="2">2 - Quick</option><option value="3">3 - Normal</option><option value="4" selected>4 - Strongest (72.5% vs SF6)</option></select>
   <label style="text-transform:none; font-weight:normal; margin-top:8px;"><input type="checkbox" id="adaptive"> Adaptive depth (thinks deeper in tactical positions)</label>
   <label>You Play</label><select id="side"><option value="white">White</option><option value="black">Black</option><option value="random">Random</option></select>
   <button onclick="newGame()">New game</button>
   <p><small>Click a piece, then a highlighted square to play.</small></p>
  </div>
 </div>
</div>
<script>
const FILES='abcdefgh', GLYPH={K:'\u265A',Q:'\u265B',R:'\u265C',B:'\u265D',N:'\u265E',P:'\u265F'};
let game=null, sel=null, flipped=false, busy=false, pollTimer=null;

async function api(path, body){
  const r = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {})
  });
  return r.json();
}

function setStatus(t){ document.getElementById('status').textContent = t; }

function fenMap(fen){
  const m = {};
  const rows = fen.split(' ')[0].split('/');
  for(let i=0; i<8; i++){
    let f=0;
    for(const ch of rows[i]){
      if(/\d/.test(ch)){ f += parseInt(ch); }
      else { m[FILES[f] + (8-i)] = ch; f++; }
    }
  }
  return m;
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(async () => {
    if (!game || game.over) {
      stopPolling();
      return;
    }
    try {
      const r = await api('/api/state', { id: game.id });
      if (r && !r.error) {
        game = r;
        render();
        if (game.myturn || game.over) {
          stopPolling();
        }
      }
    } catch(e) {}
  }, 300);
}

function render(){
  const b = document.getElementById('board');
  b.innerHTML = '';
  if(!game){ return; }
  const pm = fenMap(game.fen), targets = (sel && game.legal[sel]) ? game.legal[sel].map(u => u.slice(2,4)) : [];
  for(let r=0; r<8; r++) for(let c=0; c<8; c++){
    const rank = flipped ? r : 7-r, file = flipped ? 7-c : c, sq = FILES[file] + (rank+1);
    const d = document.createElement('div');
    d.className = 'sq ' + (((file+rank)%2===0) ? 'dark' : 'light');
    if(game.last && (sq === game.last.slice(0,2) || sq === game.last.slice(2,4))) d.classList.add('last');
    if(sq === sel) d.classList.add('sel');
    if(sq === game.check) d.classList.add('check');
    if(c === 0){ const s = document.createElement('span'); s.className = 'coord r'; s.textContent = rank+1; d.appendChild(s); }
    if(r === 7){ const s = document.createElement('span'); s.className = 'coord f'; s.textContent = FILES[file]; d.appendChild(s); }
    const p = pm[sq];
    if(p){ const s = document.createElement('span'); s.className = 'piece ' + (p === p.toUpperCase() ? 'w' : 'b'); s.textContent = GLYPH[p.toUpperCase()] + '\uFE0E'; d.appendChild(s); }
    if(targets.includes(sq)){ const dot = document.createElement('div'); dot.className = p ? 'cap' : 'dot'; d.appendChild(dot); }
    d.onclick = () => clickSq(sq);
    b.appendChild(d);
  }
  setStatus(busy ? 'Bot is thinking...' : game.status);
  let h = '';
  for(let i=0; i<game.history.length; i+=2){
    h += (i/2+1) + '. ' + game.history[i] + ' ' + (game.history[i+1] || '') + '  ';
  }
  document.getElementById('hist').textContent = h;
  const hd = document.getElementById('hist');
  hd.scrollTop = hd.scrollHeight;
  
  const bi = document.getElementById('botinfo'), fill = document.getElementById('barfill');
  if(game.bot){
    let e = game.bot.mate_for ? ('mate for ' + game.bot.mate_for) : ((game.bot.eval_white >= 0 ? '+' : '') + (game.bot.eval_white*100).toFixed(0) + ' cp for white');
    bi.innerHTML = '<small>' + game.bot_label + '<br>Bot evaluation: <b>' + e + '</b><br>search depth ' + game.bot.depth + ', ' + game.bot.nodes + ' nodes, ' + game.bot.seconds + ' s</small>';
    fill.style.width = (game.bot.mate_for ? (game.bot.mate_for === 'white' ? 100 : 0) : 50 + 50*game.bot.eval_white) + '%';
  } else {
    bi.innerHTML = '<small>' + (game.bot_label || '') + '</small>';
    fill.style.width = '50%';
  }
}

async function clickSq(sq){
  if(!game || game.over || busy || !game.myturn) return;
  if(sel && game.legal[sel]){
    const c = game.legal[sel].filter(u => u.slice(2,4) === sq);
    if(c.length){
      const uci = c.find(u => u.length === 5 && u[4] === 'q') || c[0];
      sel = null;
      await play(uci);
      return;
    }
  }
  sel = game.legal[sq] ? sq : null;
  render();
}

async function play(uci){
  busy = true;
  render();
  let r;
  try {
    r = await api('/api/move', { id: game.id, uci: uci });
  } catch(e) {
    r = { error: 'connection problem' };
  }
  busy = false;
  if(r.error){
    alert(r.error);
  } else {
    game = r;
    sel = null;
    render();
    if (!game.over && !game.myturn) {
      startPolling();
    }
  }
}

async function newGame(){
  stopPolling();
  busy = true;
  sel = null;
  setStatus('Starting...');
  let r;
  try {
    r = await api('/api/new', {
      name: document.getElementById('name').value,
      bot: document.getElementById('bot').value,
      depth: document.getElementById('depth').value,
      adaptive: document.getElementById('adaptive').checked,
      side: document.getElementById('side').value
    });
  } catch(e) {
    r = { error: 'connection problem' };
  }
  busy = false;
  if(r.error){
    alert(r.error);
    return;
  }
  game = r;
  flipped = !game.human_white;
  render();
  if (!game.over && !game.myturn) {
    startPolling();
  }
}

async function undo(){
  if(!game || busy) return;
  stopPolling();
  const r = await api('/api/undo', { id: game.id });
  if(r.error){
    alert(r.error);
  } else {
    game = r;
    sel = null;
    render();
  }
}

async function resign(){
  if(!game || busy || game.over) return;
  if(!confirm('Resign this game?')) return;
  stopPolling();
  const r = await api('/api/resign', { id: game.id });
  if(!r.error){
    game = r;
    render();
  }
}

function flip(){
  flipped = !flipped;
  render();
}

fetch('/api/bots')
  .then(r => r.json())
  .then(list => {
    const s = document.getElementById('bot');
    list.forEach(o => {
      const e = document.createElement('option');
      e.value = o.key;
      e.textContent = o.label;
      s.appendChild(e);
    });
  });
</script></body></html>"""


def lan_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except OSError:
        return socket.gethostbyname(socket.gethostname())


if __name__ == "__main__":
    bots = available_bots()
    print("Bots available:", ", ".join(bots.values()))
    print(f"\nOpen on this computer:   http://localhost:{PORT}")
    print(f"Friends on your Wi-Fi:   http://{lan_address()}:{PORT}")
    print("(Windows may ask to allow Python through the firewall - allow private networks.)")
    print("Leave this command prompt open while playing. Press Ctrl+C to stop.\n")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()