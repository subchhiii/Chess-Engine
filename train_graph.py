"""Graph hybrid trainer: CNN + SE blocks + attention guided by the chess attack graph (see hybrid_model.py).
Stage 1 (pretrain on original + Lichess):  USE_LICHESS = True,  INIT_FROM = None,      OUT_NAME = "gph_a", EPOCHS = 20, LR = 5e-4
Stage 2 (fine-tune on original only):      USE_LICHESS = False, INIT_FROM = "gph_a",   OUT_NAME = "gph_b", EPOCHS = 40, LR = 1e-3
Run  python make_attacks.py  once before this.  Validation and test sets are never trained on."""
import os, sys, glob, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from hybrid_model import make_graph
from attack_graph import V_PERM, H_PERM

# ======================= SETTINGS =======================
DATA_DIR = r"C:\Work\Fen analysis"
QUICK_TEST = False            # True: small subset, 3 epochs.  False: full run
EPOCHS = 3 if QUICK_TEST else 40
OUT_NAME = "gph_b"
SEED = 62
USE_LICHESS = False
INIT_FROM = "gph_a"
LR = 1e-3
LICHESS_FILE = "lichess_extra_cnn.npz"
LICHESS_ATT = "lichess_extra_att.npy"
LICHESS_MAX = 3000000
OVERWRITE = False
# ========================================================
torch.manual_seed(SEED); np.random.seed(SEED)
best_path = os.path.join(DATA_DIR, OUT_NAME + ("_quick" if QUICK_TEST else "") + ".pt")
if os.path.exists(best_path) and not OVERWRITE:
    sys.exit(f"{best_path} already exists. Change OUT_NAME (or set OVERWRITE = True) so a good model is not overwritten.")
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("training on:", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU")

def load(name, as_float=True):
    z = np.load(glob.glob(os.path.join(DATA_DIR, f"*{name}_cnn.npz"))[0])
    X, y = torch.from_numpy(z["X"]), torch.from_numpy(z["y"]).float()
    att = torch.from_numpy(np.load(os.path.join(DATA_DIR, f"{name}_att.npy")))
    assert len(att) == len(X), f"{name}: attack file does not match the data - rerun make_attacks.py"
    return (X.float() if as_float else X), att, y

Xtr, Atr, ytr = load("train", as_float=False)
Xva, Ava, yva = load("validation"); Xte, Ate, yte = load("test")
if USE_LICHESS:
    z = np.load(os.path.join(DATA_DIR, LICHESS_FILE))
    Xex, yex = torch.from_numpy(z["X"]), torch.from_numpy(z["y"]).float()
    Aex = torch.from_numpy(np.load(os.path.join(DATA_DIR, LICHESS_ATT)))
    assert len(Aex) == len(Xex), "Lichess attack file does not match - rerun make_attacks.py"
    if LICHESS_MAX and len(Xex) > LICHESS_MAX:
        sel = torch.randperm(len(Xex))[:LICHESS_MAX]; Xex, Aex, yex = Xex[sel], Aex[sel], yex[sel]
    if QUICK_TEST: Xtr, Atr, ytr, Xex, Aex, yex = Xtr[:30000], Atr[:30000], ytr[:30000], Xex[:30000], Aex[:30000], yex[:30000]
    Xtr, Atr, ytr = torch.cat([Xtr, Xex]), torch.cat([Atr, Aex]), torch.cat([ytr, yex])
    print(f"training positions: {len(Xtr)} (original + {len(Xex)} from Lichess)")
    del Xex, Aex, yex
elif QUICK_TEST:
    Xtr, Atr, ytr = Xtr[:60000], Atr[:60000], ytr[:60000]

def color_flip(x):                                   # swap colours and mirror vertically (labels are side-to-move relative)
    pieces = torch.cat([x[:, 6:12], x[:, 0:6]], 1).flip(2)
    return torch.cat([pieces, 1 - x[:, 12:13], torch.cat([x[:, 15:17], x[:, 13:15]], 1), x[:, 17:18].flip(2)], 1)

VP = torch.as_tensor(V_PERM, device=dev); HP = torch.as_tensor(H_PERM, device=dev)

@torch.no_grad()
def evaluate(model, X, A, y):
    model.eval()
    p = torch.cat([model(X[i:i+1024].to(dev), A[i:i+1024].to(dev)).squeeze(1).cpu() for i in range(0, len(X), 1024)])
    err = p - y
    clear = y.abs() > 0.1
    return dict(MAE=round(err.abs().mean().item(), 4),
                MSE=round(err.pow(2).mean().item(), 5),
                RMSE=round(err.pow(2).mean().sqrt().item(), 4),
                R2=round(1 - (err**2).sum().item() / ((y - y.mean())**2).sum().item(), 4),
                corr=round(float(np.corrcoef(p.numpy(), y.numpy())[0, 1]), 4),
                sign_agree=round((torch.sign(p[clear]) == torch.sign(y[clear])).float().mean().item(), 4))

model = make_graph().to(dev)
if INIT_FROM:
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, INIT_FROM + ".pt"), map_location=dev))
    print("starting from", INIT_FROM)
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.MSELoss()
dl = DataLoader(TensorDataset(Xtr, Atr, ytr), batch_size=256, shuffle=True, pin_memory=(dev == "cuda"))
best = 9.0

for ep in range(EPOCHS):
    model.train(); t0 = time.time(); run_loss, seen = 0.0, 0
    for xb, ab, yb in dl:
        xb, yb = xb.to(dev).float(), yb.to(dev)
        A = model.unpack(ab.to(dev))                                        # (B, 64, 64) float
        n = len(xb)
        flip = torch.rand(n, device=dev) < 0.5                              # colour flip: board and attack graph together
        xb = torch.where(flip[:, None, None, None], color_flip(xb), xb)
        A = torch.where(flip[:, None, None], A[:, VP][:, :, VP], A)
        no_castle = xb[:, 13:17].sum(dim=(1, 2, 3)) == 0                    # left-right flip only without castling rights
        hf = (torch.rand(n, device=dev) < 0.5) & no_castle
        xb = torch.where(hf[:, None, None, None], xb.flip(3), xb)
        A = torch.where(hf[:, None, None], A[:, HP][:, :, HP], A)
        opt.zero_grad()
        loss = loss_fn(model(xb, A).squeeze(1), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.detach() * n; seen += n
    sched.step()
    r = evaluate(model, Xva, Ava, yva)
    mark = ""
    if r["MAE"] < best:
        best = r["MAE"]; torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path); mark = "  <- saved"
    print(f"epoch {ep+1}/{EPOCHS} ({time.time()-t0:.0f}s) train RMSE {(float(run_loss)/seen)**0.5:.4f} | val: {r}{mark}", flush=True)

model.load_state_dict(torch.load(best_path, map_location=dev))
print("TEST (best-validation checkpoint):", evaluate(model, Xte, Ate, yte))
print("for comparison, hybrid hyb_d: MAE 0.0793, RMSE 0.1320, corr 0.9511 (0.0736 / 0.1229 with TTA)")
